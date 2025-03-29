-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create the notes table
CREATE TABLE IF NOT EXISTS notes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Create an index on the embedding column for faster similarity searches
CREATE INDEX IF NOT EXISTS notes_embedding_idx ON notes USING ivfflat (embedding vector_cosine_ops);

-- Create an index on the metadata JSONB for faster queries
CREATE INDEX IF NOT EXISTS notes_metadata_idx ON notes USING gin (metadata);

-- Enable Row Level Security
ALTER TABLE notes ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations with anon key
CREATE POLICY "Allow all operations with anon key" ON notes
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Grant necessary permissions
GRANT ALL ON notes TO anon;
GRANT USAGE ON SCHEMA public TO anon;

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = TIMEZONE('utc'::text, NOW());
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_notes_updated_at
    BEFORE UPDATE ON notes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create function for similarity search
CREATE OR REPLACE FUNCTION match_notes(
    query_embedding vector(1536),
    match_threshold float,
    match_count int
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        notes.id,
        notes.content,
        notes.metadata,
        1 - (notes.embedding <=> query_embedding) as similarity
    FROM notes
    WHERE 1 - (notes.embedding <=> query_embedding) > match_threshold
    ORDER BY notes.embedding <=> query_embedding
    LIMIT match_count;
END;
$$; 