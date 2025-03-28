import os
import json
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from dotenv import load_dotenv
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

# Load environment variables
load_dotenv()

class MarkdownProcessor:
    def __init__(self):
        # Load environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        self.table_name = os.getenv('DOCUMENTS_TABLE')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_model = os.getenv('OPENAI_MODEL', 'text-embedding-3-small')
        self.watch_dir = os.getenv('WATCH_DIR', './docs')
        self.polling_interval = int(os.getenv('POLLING_INTERVAL', '300'))
        
        # Get allowed extensions from environment variable
        extensions_str = os.getenv('FILE_EXTENSIONS', '.md,.txt')
        self.allowed_extensions = [ext.strip() for ext in extensions_str.split(',')]
        
        # Get excluded folders from environment variable
        exclude_folders_str = os.getenv('EXCLUDE_FOLDERS', '.git,node_modules,venv,__pycache__')
        self.excluded_folders = [folder.strip() for folder in exclude_folders_str.split(',')]
        
        # Initialize Supabase client
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize OpenAI client
        if not self.openai_api_key:
            raise ValueError("OpenAI API key must be provided in environment variables")
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Load existing hashes
        self.processed_files = self.load_file_hashes()
        
        # Log startup configuration
        logger.info("Starting Markdown Indexer with configuration:")
        logger.info(f"Supabase URL: {self.supabase_url}")
        logger.info(f"Supabase Key: {'*' * len(self.supabase_key)}")
        logger.info(f"Watch Directory: {os.path.abspath(self.watch_dir)}")
        logger.info(f"Polling Interval: {self.polling_interval} seconds")
        logger.info(f"Table Name: {self.table_name}")
        logger.info(f"File Extensions: {', '.join(self.allowed_extensions)}")
        logger.info(f"OpenAI Model: {self.openai_model}")
        logger.info(f"Excluded Folders: {', '.join(self.excluded_folders)}")
        
        # Count files in watch directory
        file_count = sum(1 for root, _, files in os.walk(self.watch_dir)
                        for file in files if self.is_allowed_file(os.path.join(root, file)))
        logger.info(f"Found {file_count} files in watch directory")
        logger.info(f"Loaded {len(self.processed_files)} existing hashes from file_hashes.json")
        
        # Initialize processed files dictionary and load existing hashes
        self.processed_files: Dict[str, str] = {}  # file_path -> hash mapping
        self.hashes_file = Path('file_hashes.json')
        self.load_file_hashes()
        
        logger.info(f"Initialized MarkdownProcessor to watch directory: {self.watch_dir}")
        self.check_table_exists()

    def load_file_hashes(self):
        """Load existing file hashes from JSON file."""
        try:
            if os.path.exists('file_hashes.json'):
                with open('file_hashes.json', 'r') as f:
                    return json.load(f)
            return {}  # Return empty dict if file doesn't exist
        except Exception as e:
            logger.error(f"Error loading file hashes: {str(e)}")
            return {}  # Return empty dict on error

    def save_file_hashes(self):
        """Save file hashes to disk."""
        try:
            with open(self.hashes_file, 'w') as f:
                json.dump(self.processed_files, f)
            logger.info(f"Saved {len(self.processed_files)} file hashes to disk")
        except Exception as e:
            logger.error(f"Error saving file hashes: {str(e)}")

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
            # Check if we already have a hash for this file
            if file_path in self.processed_files:
                return self.processed_files[file_path]

            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            hash_value = sha256_hash.hexdigest()
            
            # Save the new hash
            self.processed_files[file_path] = hash_value
            self.save_file_hashes()
            
            return hash_value
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            raise

    def process_markdown_file(self, file_path: str) -> Optional[Dict]:
        """Process a markdown file and return document data."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Generate embedding
            embedding = self.generate_embedding(content)

            # Prepare metadata
            metadata = {
                'filename': os.path.basename(file_path),
                'path': file_path,
                'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                'file_size': os.path.getsize(file_path)
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
        self.processor.processed_files[event.src_path] = self.processor.calculate_file_hash(event.src_path)
        doc_data = self.processor.process_markdown_file(event.src_path)
        if doc_data:
            self.processor.upsert_document(event.src_path, doc_data)

    def on_modified(self, event):
        if event.is_directory or not self.processor.is_allowed_file(event.src_path):
            return
        logger.info(f"File modified: {event.src_path}")
        current_hash = self.processor.calculate_file_hash(event.src_path)
        if event.src_path not in self.processor.processed_files or \
           self.processor.processed_files[event.src_path] != current_hash:
            self.processor.processed_files[event.src_path] = current_hash
            doc_data = self.processor.process_markdown_file(event.src_path)
            if doc_data:
                self.processor.upsert_document(event.src_path, doc_data)

    def on_deleted(self, event):
        if event.is_directory or not self.processor.is_allowed_file(event.src_path):
            return
        logger.info(f"File deleted: {event.src_path}")
        self.processor.delete_document(event.src_path)
        if event.src_path in self.processor.processed_files:
            del self.processor.processed_files[event.src_path]

def main():
    try:
        processor = MarkdownProcessor()
        event_handler = MarkdownHandler(processor)
        
        # Initialize observer
        observer = PollingObserver(timeout=processor.polling_interval)
        observer.schedule(event_handler, processor.watch_dir, recursive=True)
        
        # Process existing files with allowed extensions
        logger.info("Processing existing files...")
        for root, _, files in os.walk(processor.watch_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if processor.is_allowed_file(file_path):
                    # Check if file exists in Supabase
                    try:
                        response = processor.supabase.table(processor.table_name).select('id').eq('metadata->>path', file_path).execute()
                        exists_in_supabase = bool(response.data)
                    except Exception as e:
                        logger.error(f"Error checking file existence in Supabase: {str(e)}")
                        exists_in_supabase = False
                    
                    # Calculate current hash
                    current_hash = processor.calculate_file_hash(file_path)
                    
                    # Process if file is new, hash has changed, or doesn't exist in Supabase
                    if not exists_in_supabase or file_path not in processor.processed_files or \
                       processor.processed_files[file_path] != current_hash:
                        logger.info(f"Processing file: {file_path} (exists in Supabase: {exists_in_supabase})")
                        doc_data = processor.process_markdown_file(file_path)
                        if doc_data:
                            processor.upsert_document(file_path, doc_data)
                    else:
                        logger.debug(f"Skipping unchanged file: {file_path}")
        
        # Save final state of hashes
        processor.save_file_hashes()
        
        # Start monitoring
        observer.start()
        logger.info(f"Started monitoring {processor.watch_dir} for files with extensions: {', '.join(processor.allowed_extensions)}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            logger.info("Stopped monitoring")
            # Save hashes before exiting
            processor.save_file_hashes()
        
        observer.join()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 