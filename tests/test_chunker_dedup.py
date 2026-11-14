import pytest

from worker.chunker import semantic_chunk_text
from worker.dedup import compute_content_hash


def test_semantic_chunker_preserves_sentence_boundaries_and_bounds():
    text = (
        "Apache Kafka KRaft removes Zookeeper overhead. "
        "PostgreSQL pgvector supports HNSW cosine indexes. "
        "Reciprocal Rank Fusion merges dense and sparse rankings. "
        "SHA-256 deduplication skips redundant ONNX inference."
    )
    chunks = semantic_chunk_text(text, chunk_size=110, chunk_overlap=35)

    assert len(chunks) >= 2
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert len(chunk.chunk_text) <= 110


def test_semantic_chunker_handles_ultra_long_single_sentence():
    long_sentence = " ".join([f"token{i}" for i in range(120)])
    chunks = semantic_chunk_text(long_sentence, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 5
    for chunk in chunks:
        assert len(chunk.chunk_text) <= 100


def test_semantic_chunker_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="chunk_overlap"):
        semantic_chunk_text("Sample text.", chunk_size=100, chunk_overlap=100)


def test_compute_content_hash_normalizes_whitespace():
    h1 = compute_content_hash("HNSW   Indexing", "Line one.\n\n   Line two.  ")
    h2 = compute_content_hash("HNSW Indexing", "Line one. Line two.")
    h3 = compute_content_hash("HNSW Indexing", "Line one. Line three.")

    assert len(h1) == 64
    assert h1 == h2
    assert h1 != h3
