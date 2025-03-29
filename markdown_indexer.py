import os
import json
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from dotenv import load_dotenv, find_dotenv
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from openai import OpenAI
from supabase import Client, create_client
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('markdown_indexer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_env_vars():
    """Load and validate environment variables from .env file."""
    # Find .env file
    env_path = find_dotenv()
    if not env_path:
        raise FileNotFoundError("No .env file found. Please create one using .env.example as template.")
    
    # Load environment variables from .env file only
    load_dotenv(env_path, override=True)
    
    # Required environment variables
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key',
        'SUPABASE_URL': 'Supabase URL',
        'SUPABASE_ANON_KEY': 'Supabase anonymous key',
        'DOCUMENTS_TABLE': 'Supabase table name'
    }
    
    # Optional environment variables with defaults
    optional_vars = {
        'OPENAI_MODEL': 'text-embedding-3-small',
        'WATCH_DIR': './docs',
        'POLLING_INTERVAL': '300',
        'FILE_EXTENSIONS': '.md,.txt',
        'EXCLUDE_FOLDERS': '.git,node_modules,venv,__pycache__'
    }
    
    # Create a dictionary to store all environment variables
    env_vars = {}
    
    # First check required variables
    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing_vars.append(f"{var} ({description})")
        env_vars[var] = value
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables in .env file: {', '.join(missing_vars)}")
    
    # Then handle optional variables
    for var, default in optional_vars.items():
        value = os.getenv(var)
        env_vars[var] = value if value is not None else default
    
    # Log configuration (masking sensitive data)
    logger.info("Environment configuration loaded from .env file:")
    logger.info(f"OpenAI Model: {env_vars['OPENAI_MODEL']}")
    logger.info(f"Watch Directory: {os.path.abspath(env_vars['WATCH_DIR'])}")
    logger.info(f"Polling Interval: {env_vars['POLLING_INTERVAL']} seconds")
    logger.info(f"Table Name: {env_vars['DOCUMENTS_TABLE']}")
    logger.info(f"File Extensions: {env_vars['FILE_EXTENSIONS']}")
    logger.info(f"Excluded Folders: {env_vars['EXCLUDE_FOLDERS']}")
    logger.info(f"Supabase URL: {env_vars['SUPABASE_URL']}")
    logger.info(f"Supabase Key: {'*' * len(env_vars['SUPABASE_ANON_KEY'])}")
    logger.info(f"OpenAI Key: {'*' * len(env_vars['OPENAI_API_KEY'])}")
    
    return env_vars

class MarkdownProcessor:
    def __init__(self):
        # Load and validate environment variables
        env_vars = load_env_vars()
        
        # Get environment variables from the loaded dictionary
        self.supabase_url = env_vars['SUPABASE_URL']
        self.supabase_key = env_vars['SUPABASE_ANON_KEY']
        self.table_name = env_vars['DOCUMENTS_TABLE']
        self.openai_api_key = env_vars['OPENAI_API_KEY']
        self.openai_model = env_vars['OPENAI_MODEL']
        self.watch_dir = env_vars['WATCH_DIR']
        self.polling_interval = int(env_vars['POLLING_INTERVAL'])
        
        # Get allowed extensions from environment variable
        self.allowed_extensions = [ext.strip() for ext in env_vars['FILE_EXTENSIONS'].split(',')]
        
        # Get excluded folders from environment variable
        self.excluded_folders = [folder.strip() for folder in env_vars['EXCLUDE_FOLDERS'].split(',')]
        
        # Initialize Supabase client
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Count files in watch directory
        file_count = sum(1 for root, _, files in os.walk(self.watch_dir)
                        for file in files if self.is_allowed_file(os.path.join(root, file)))
        logger.info(f"Found {file_count} files in watch directory")
        
        logger.info(f"Initialized MarkdownProcessor to watch directory: {self.watch_dir}")
        self.check_table_exists()

    def check_table_exists(self):
        """Check if the table exists and log a warning if it doesn't."""
        try:
            # Check if table exists
            response = self.supabase.table(self.table_name).select('id').limit(1).execute()
            logger.info(f"Table {self.table_name} exists")
        except Exception as e:
            if "relation" in str(e).lower() and "does not exist" in str(e).lower():
                logger.warning(f"""
                Table {self.table_name} does not exist. Please create it using the following SQL in your Supabase SQL editor:
                
                -- Enable the vector extension if not already enabled
                CREATE EXTENSION IF NOT EXISTS vector;
                
                -- Create the documents table with proper structure
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id integer primary key generated always as identity,
                    content text not null,
                    metadata jsonb not null,
                    embedding halfvec(1536),
                    created_at timestamp with time zone default now()
                );
                
                -- Create HNSW index for vector search
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx ON {self.table_name} 
                USING hnsw (embedding halfvec_cosine_ops);
                
                Note: Make sure you have the vector extension enabled in your Supabase project.
                You can check this in the Dashboard under Database > Extensions.
                """)
                raise
            else:
                logger.error(f"Error checking table existence: {str(e)}")
                raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text using OpenAI API."""
        try:
            response = self.openai_client.embeddings.create(
                model=os.getenv('OPENAI_MODEL', 'text-embedding-3-small'),
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file content."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            raise

    def truncate_content(self, content: str, max_tokens: int = 7000) -> str:
        """Truncate content to approximately max_tokens (rough estimation)."""
        # Rough estimation: 1 token ≈ 4 characters for English text
        max_chars = max_tokens * 4
        
        if len(content) <= max_chars:
            return content
            
        # Truncate to max_chars and add a note
        truncated = content[:max_chars]
        truncated += f"\n\n[Content truncated. Original length: {len(content)} characters]"
        logger.warning(f"Content truncated from {len(content)} to {len(truncated)} characters")
        return truncated

    def process_markdown_file(self, file_path: str) -> Optional[Dict]:
        """Process a markdown file and return document data."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Truncate content if too long
            content = self.truncate_content(content)

            # Generate embedding
            embedding = self.generate_embedding(content)
            
            # Calculate file hash
            file_hash = self.calculate_file_hash(file_path)

            # Prepare metadata
            metadata = {
                # Original metadata fields
                'loc': None,  # Keep original field
                'source': 'file_system',  # Changed from 'test' to indicate source
                'file_id': file_hash,  # Use file hash as unique identifier
                'blobType': 'markdown',  # Changed from null to indicate content type
                
                # Additional metadata fields
                'filename': os.path.basename(file_path),
                'path': file_path,
                'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                'file_size': os.path.getsize(file_path),
                'content_length': len(content),
                'is_truncated': len(content) > 7000 * 4,
                'file_extension': os.path.splitext(file_path)[1],
                'directory': os.path.dirname(file_path),
                'created_at': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                'file_hash': file_hash,
                'processing_info': {
                    'processed_at': datetime.now().isoformat(),
                    'model': self.openai_model,
                    'embedding_dimension': 1536
                }
            }

            return {
                'content': content,
                'metadata': metadata,
                'embedding': embedding
            }
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None

    def upsert_document(self, file_path: str, doc_data: Dict):
        """Insert or update document in Supabase."""
        try:
            # Check if document exists
            response = self.supabase.table(self.table_name).select('id').eq('metadata->>path', file_path).execute()
            
            if response.data:
                # Update existing document
                self.supabase.table(self.table_name).update(doc_data).eq('id', response.data[0]['id']).execute()
                logger.info(f"Updated document for {file_path}")
            else:
                # Insert new document
                self.supabase.table(self.table_name).insert(doc_data).execute()
                logger.info(f"Inserted new document for {file_path}")
            
        except Exception as e:
            logger.error(f"Error upserting document for {file_path}: {str(e)}")
            raise

    def delete_document(self, file_path: str):
        """Delete document from Supabase."""
        try:
            self.supabase.table(self.table_name).delete().eq('metadata->>path', file_path).execute()
            logger.info(f"Successfully deleted document for {file_path}")
        except Exception as e:
            logger.error(f"Error deleting document for {file_path}: {str(e)}")
            raise

    def is_allowed_file(self, file_path: str) -> bool:
        """Check if file has allowed extension and is not in excluded folder"""
        # Check if file is in an excluded folder
        for folder in self.excluded_folders:
            if folder in file_path.split(os.sep):
                return False
                
        # Check file extension
        return any(file_path.lower().endswith(ext.lower()) for ext in self.allowed_extensions)

class MarkdownHandler(FileSystemEventHandler):
    def __init__(self, processor: MarkdownProcessor):
        self.processor = processor

    def on_created(self, event):
        if event.is_directory or not self.processor.is_allowed_file(event.src_path):
            return
        logger.info(f"New file detected: {event.src_path}")
        doc_data = self.processor.process_markdown_file(event.src_path)
        if doc_data:
            self.processor.upsert_document(event.src_path, doc_data)

    def on_modified(self, event):
        if event.is_directory or not self.processor.is_allowed_file(event.src_path):
            return
        logger.info(f"File modified: {event.src_path}")
        doc_data = self.processor.process_markdown_file(event.src_path)
        if doc_data:
            self.processor.upsert_document(event.src_path, doc_data)

    def on_deleted(self, event):
        if event.is_directory or not self.processor.is_allowed_file(event.src_path):
            return
        logger.info(f"File deleted: {event.src_path}")
        self.processor.delete_document(event.src_path)

    def on_moved(self, event):
        """Handle file moves/renames."""
        if event.is_directory:
            return
            
        # Handle source file (old path)
        if self.processor.is_allowed_file(event.src_path):
            logger.info(f"File moved/renamed from: {event.src_path}")
            self.processor.delete_document(event.src_path)
            
        # Handle destination file (new path)
        if self.processor.is_allowed_file(event.dest_path):
            logger.info(f"File moved/renamed to: {event.dest_path}")
            doc_data = self.processor.process_markdown_file(event.dest_path)
            if doc_data:
                self.processor.upsert_document(event.dest_path, doc_data)

def check_and_remove_deleted_files(processor):
    """Check for and remove database entries for files that no longer exist."""
    try:
        # Get all files from database
        response = processor.supabase.table(processor.table_name).select('metadata').execute()
        db_files = [doc['metadata']['path'] for doc in response.data]
        
        # Get all existing files in watch directory
        existing_files = []
        for root, _, files in os.walk(processor.watch_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if processor.is_allowed_file(file_path):
                    existing_files.append(file_path)
        
        # Find files that are in database but not in filesystem
        deleted_files = set(db_files) - set(existing_files)
        
        # Remove deleted files from database
        for file_path in deleted_files:
            logger.info(f"Removing deleted file from database: {file_path}")
            processor.delete_document(file_path)
            
        if deleted_files:
            logger.info(f"Removed {len(deleted_files)} deleted files from database")
    except Exception as e:
        logger.error(f"Error checking for deleted files: {str(e)}")

def main():
    try:
        processor = MarkdownProcessor()
        event_handler = MarkdownHandler(processor)
        
        # Initialize observer
        observer = PollingObserver(timeout=processor.polling_interval)
        observer.schedule(event_handler, processor.watch_dir, recursive=True)
        
        # Check for and remove deleted files from database
        check_and_remove_deleted_files(processor)
        
        # Process existing files with allowed extensions
        logger.info("Processing existing files...")
        for root, _, files in os.walk(processor.watch_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if processor.is_allowed_file(file_path):
                    # Check if file exists in Supabase and get its hash
                    try:
                        response = processor.supabase.table(processor.table_name).select('id, metadata').eq('metadata->>path', file_path).execute()
                        exists_in_supabase = bool(response.data)
                        stored_hash = response.data[0]['metadata'].get('file_hash') if exists_in_supabase else None
                    except Exception as e:
                        logger.error(f"Error checking file existence in Supabase: {str(e)}")
                        exists_in_supabase = False
                        stored_hash = None
                    
                    # Calculate current hash
                    current_hash = processor.calculate_file_hash(file_path)
                    
                    # Process if file is new, hash has changed, or doesn't exist in Supabase
                    if not exists_in_supabase or stored_hash != current_hash:
                        logger.info(f"Processing file: {file_path} (exists in Supabase: {exists_in_supabase}, hash changed: {stored_hash != current_hash if stored_hash else True})")
                        doc_data = processor.process_markdown_file(file_path)
                        if doc_data:
                            processor.upsert_document(file_path, doc_data)
                    else:
                        logger.debug(f"Skipping unchanged file: {file_path}")
        
        # Start monitoring
        observer.start()
        logger.info(f"Started monitoring {processor.watch_dir} for files with extensions: {', '.join(processor.allowed_extensions)}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            logger.info("Stopped monitoring")
        
        observer.join()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 