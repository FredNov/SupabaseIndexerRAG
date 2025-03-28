import os
import logging
from pathlib import Path
from typing import Optional
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

def find_env_file() -> Optional[Path]:
    """
    Find the .env file by checking multiple locations:
    1. Current working directory
    2. Script directory
    3. Parent directory of script
    """
    # Get the script's directory
    script_dir = Path(__file__).resolve().parent
    current_dir = Path.cwd()
    
    # List of possible .env file locations
    possible_locations = [
        current_dir / '.env',
        script_dir / '.env',
        script_dir.parent / '.env'
    ]
    
    # Log all locations we're checking
    logger.info(f"Checking for .env file in locations: {[str(p) for p in possible_locations]}")
    
    # Find the first existing .env file
    for env_path in possible_locations:
        if env_path.exists():
            logger.info(f"Found .env file at: {env_path}")
            return env_path
    
    logger.info("No .env file found in any of the checked locations")
    return None

# Try to load .env file if it exists
env_path = find_env_file()
if env_path:
    load_dotenv(env_path)
else:
    logger.info("Using system environment variables")

def get_env_var(key: str, default: Optional[str] = None) -> str:
    """Get environment variable with fallback to system environment."""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Required environment variable {key} not found in .env file or system environment")
    return value

def cleanup_table():
    """Delete all records from the specified Supabase table."""
    try:
        # Initialize Supabase client with environment variables
        supabase_url = get_env_var('SUPABASE_URL')
        supabase_key = get_env_var('SUPABASE_ANON_KEY')
        table_name = get_env_var('DOCUMENTS_TABLE', 'documents')

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