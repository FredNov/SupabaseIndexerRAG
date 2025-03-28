# Markdown File Indexer

A Python script that monitors a directory for Markdown files, processes their content, generates embeddings using OpenAI, and maintains a synchronized index in Supabase with vector search capabilities.

## Features

- Real-time monitoring of Markdown files
- Automatic processing of new, modified, and deleted files
- OpenAI embeddings generation
- Supabase vector storage integration
- Efficient change detection using file hashes
- Comprehensive error handling and logging
- Cross-platform compatibility

## Prerequisites

- Python 3.8 or higher
- Supabase account with pgvector extension enabled
- OpenAI API key
- Required Python packages (listed in requirements.txt)

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_ANON_KEY=your_supabase_anon_key
   WATCH_DIR=path_to_watch
   POLLING_INTERVAL=5
   OPENAI_MODEL=text-embedding-3-small
   ```

4. Set up your Supabase database with the following table:
   ```sql
   CREATE TABLE documents (
     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
     content TEXT NOT NULL,
     metadata JSONB NOT NULL,
     embedding vector(1536)
   );
   ```

## Usage

Run the script:
```bash
python markdown_indexer.py
```

The script will:
1. Start monitoring the specified directory for Markdown files
2. Process any existing Markdown files
3. Watch for new, modified, or deleted files
4. Generate embeddings and update the Supabase database accordingly

## Logging

Logs are written to both:
- Console output
- `markdown_indexer.log` file

## Error Handling

The script includes:
- Retry mechanism for API calls
- Comprehensive error logging
- Graceful handling of file access errors
- Rate limiting for OpenAI API calls

## Performance Considerations

- Uses file hashing to avoid unnecessary updates
- Implements connection pooling for Supabase
- Efficient batch processing for multiple file changes
- Configurable polling interval

## Contributing

Feel free to submit issues and enhancement requests! 