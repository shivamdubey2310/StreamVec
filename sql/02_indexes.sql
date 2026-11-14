-- Dense vector index (HNSW with cosine distance, tuned for 8GB RAM environments)
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Sparse keyword index (GIN over generated tsvector for BM25-style ranking)
CREATE INDEX IF NOT EXISTS idx_chunks_gin
    ON document_chunks
    USING gin (search_tsv);

-- Fast foreign-key lookup for ACID document updates & tombstone deletes
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
    ON document_chunks (doc_id);

-- Fast time-range filtering for >30-day cold Parquet archival
CREATE INDEX IF NOT EXISTS idx_chunks_indexed_at
    ON document_chunks (indexed_at);