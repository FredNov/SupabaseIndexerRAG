import os
import logging
from dotenv import load_dotenv
from supabase import create_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleanup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def cleanup_table():
    """Delete all records from the specified Supabase table."""
    try:
        # Initialize Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')
        table_name = os.getenv('DOCUMENTS_TABLE', 'documents')

        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and key must be provided in environment variables")

        supabase = create_client(supabase_url, supabase_key)

        # Get count before deletion
        response = supabase.table(table_name).select('id', count='exact').execute()
        count_before = response.count

        if count_before == 0:
            logger.info(f"Table {table_name} is already empty")
            return

        # Delete all records
        supabase.table(table_name).delete().neq('id', 0).execute()
        
        logger.info(f"Successfully deleted {count_before} records from table {table_name}")
        
    except Exception as e:
        logger.error(f"Error cleaning up table: {str(e)}")
        raise

if __name__ == "__main__":
    cleanup_table() 