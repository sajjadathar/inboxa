-- Runs once, the first time the database starts.
-- Turns on pgvector so we can store embeddings for the knowledge base.
CREATE EXTENSION IF NOT EXISTS vector;
