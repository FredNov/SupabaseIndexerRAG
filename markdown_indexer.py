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
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize Supabase client without proxy
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and key must be provided in environment variables")
        
        self.supabase = create_client(supabase_url, supabase_key)
        self.watch_dir = os.path.abspath(os.getenv('WATCH_DIR', '.'))
        self.polling_interval = int(os.getenv('POLLING_INTERVAL', '5'))
        self.table_name = os.getenv('DOCUMENTS_TABLE', 'documents')
        self.processed_files: Dict[str, str] = {}  # file_path -> hash mapping
        
        # Get allowed file extensions from environment
        extensions = os.getenv('FILE_EXTENSIONS', '.md')
        self.allowed_extensions = tuple(ext.strip() for ext in extensions.split(','))
        logger.info(f"Monitoring files with extensions: {self.allowed_extensions}")
        
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
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

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
        """Check if the file has an allowed extension."""
        return file_path.lower().endswith(self.allowed_extensions)

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
                    processor.processed_files[file_path] = processor.calculate_file_hash(file_path)
                    doc_data = processor.process_markdown_file(file_path)
                    if doc_data:
                        processor.upsert_document(file_path, doc_data)
        
        # Start monitoring
        observer.start()
        logger.info(f"Started monitoring {processor.watch_dir} for markdown files")
        
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