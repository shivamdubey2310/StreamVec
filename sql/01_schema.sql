-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Parent table: tracks document-level state, SHA-256 hash, and lifecycle
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    source          VARCHAR(64) NOT NULL,
    title           TEXT NOT NULL,
    content_hash    CHAR(64) NOT NULL,
    published_at    TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Child table: stores semantic chunks, 384d embeddings, and generated full-text search vectors
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id                BIGSERIAL PRIMARY KEY,
    doc_id                  TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    source                  VARCHAR(64) NOT NULL,
    title                   TEXT NOT NULL,
    chunk_index             INTEGER NOT NULL,
    chunk_text              TEXT NOT NULL,
    embedding               vector(384) NOT NULL,
    embedding_model_version VARCHAR(64) NOT NULL DEFAULT 'bge-small-en-v1.5-384d',
    search_tsv              tsvector GENERATED ALWAYS AS (
                                to_tsvector('english', coalesce(title, '') || ' ' || coalesce(chunk_text, ''))
                            ) STORED,
    published_at            TIMESTAMPTZ NOT NULL,
    indexed_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (doc_id, chunk_index)
);

-- Governance table: stores Airflow 3 golden-dataset evaluation runs (MRR & Hit@5)
CREATE TABLE IF NOT EXISTS retrieval_eval_logs (
    eval_id         BIGSERIAL PRIMARY KEY,
    run_timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    search_mode     VARCHAR(32) NOT NULL, -- 'hybrid' or 'vector_only'
    num_queries     INTEGER NOT NULL,
    hit_at_5        NUMERIC(5, 4) NOT NULL,
    mrr             NUMERIC(5, 4) NOT NULL,
    avg_latency_ms  NUMERIC(8, 2) NOT NULL
);