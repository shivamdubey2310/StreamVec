import time
from typing import Any, Dict, List, Optional, Tuple

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from api.models import SearchHit


HYBRID_RRF_SQL = """
WITH vector_candidates AS (
    SELECT
        chunk_id,
        ROW_NUMBER() OVER (ORDER BY embedding <=> %(query_vec)s::vector) AS vec_rank
    FROM document_chunks
    WHERE embedding_model_version = %(model_version)s
      AND (%(source_filter)s::text IS NULL OR source = %(source_filter)s)
    ORDER BY embedding <=> %(query_vec)s::vector
    LIMIT 20
),
keyword_candidates AS (
    SELECT
        chunk_id,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(search_tsv, websearch_to_tsquery('english', %(query_text)s)) DESC
        ) AS kw_rank
    FROM document_chunks
    WHERE embedding_model_version = %(model_version)s
      AND (%(source_filter)s::text IS NULL OR source = %(source_filter)s)
      AND search_tsv @@ websearch_to_tsquery('english', %(query_text)s)
    ORDER BY ts_rank_cd(search_tsv, websearch_to_tsquery('english', %(query_text)s)) DESC
    LIMIT 20
),
fused AS (
    SELECT
        COALESCE(v.chunk_id, k.chunk_id) AS chunk_id,
        v.vec_rank AS vector_rank,
        k.kw_rank AS keyword_rank,
        (
            COALESCE(1.0 / (%(rrf_k)s + v.vec_rank), 0.0) +
            COALESCE(1.0 / (%(rrf_k)s + k.kw_rank), 0.0)
        ) AS rrf_score
    FROM vector_candidates v
    FULL OUTER JOIN keyword_candidates k ON v.chunk_id = k.chunk_id
)
SELECT
    dc.chunk_id,
    dc.doc_id,
    dc.source,
    dc.title,
    dc.chunk_index,
    dc.chunk_text,
    ROUND(f.rrf_score::numeric, 6)::float AS rrf_score,
    f.vector_rank::int AS vector_rank,
    f.keyword_rank::int AS keyword_rank,
    dc.indexed_at
FROM fused f
JOIN document_chunks dc ON dc.chunk_id = f.chunk_id
ORDER BY f.rrf_score DESC, dc.chunk_id ASC
LIMIT %(top_k)s;
"""

VECTOR_ONLY_SQL = """
WITH vector_candidates AS (
    SELECT
        chunk_id,
        ROW_NUMBER() OVER (ORDER BY embedding <=> %(query_vec)s::vector) AS vec_rank
    FROM document_chunks
    WHERE embedding_model_version = %(model_version)s
      AND (%(source_filter)s::text IS NULL OR source = %(source_filter)s)
    ORDER BY embedding <=> %(query_vec)s::vector
    LIMIT %(top_k)s
)
SELECT
    dc.chunk_id,
    dc.doc_id,
    dc.source,
    dc.title,
    dc.chunk_index,
    dc.chunk_text,
    ROUND((1.0 / (%(rrf_k)s + v.vec_rank))::numeric, 6)::float AS rrf_score,
    v.vec_rank::int AS vector_rank,
    NULL::int AS keyword_rank,
    dc.indexed_at
FROM vector_candidates v
JOIN document_chunks dc ON dc.chunk_id = v.chunk_id
ORDER BY v.vec_rank ASC;
"""


def _configure_conn(conn) -> None:
    register_vector(conn)


class HybridRetriever:
    def __init__(self, dsn: str, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k
        self.pool = ConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=5,
            configure=_configure_conn,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )

    def close(self) -> None:
        self.pool.close()

    def search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
        mode: str = "hybrid",
        source_filter: Optional[str] = None,
        model_version: str = "bge-small-en-v1.5-384d",
    ) -> Tuple[List[SearchHit], float]:
        sql = HYBRID_RRF_SQL if mode == "hybrid" else VECTOR_ONLY_SQL
        params: Dict[str, Any] = {
            "query_text": query_text,
            "query_vec": query_embedding,
            "top_k": top_k,
            "rrf_k": self.rrf_k,
            "source_filter": source_filter,
            "model_version": model_version,
        }

        t0 = time.perf_counter()
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        db_ms = (time.perf_counter() - t0) * 1000.0

        hits = [SearchHit(**row) for row in rows]
        return hits, round(db_ms, 2)

    def explain_plan(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
        model_version: str = "bge-small-en-v1.5-384d",
    ) -> List[str]:
        explain_sql = f"EXPLAIN (ANALYZE, BUFFERS)\n{HYBRID_RRF_SQL}"
        params: Dict[str, Any] = {
            "query_text": query_text,
            "query_vec": query_embedding,
            "top_k": top_k,
            "rrf_k": self.rrf_k,
            "source_filter": None,
            "model_version": model_version,
        }
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(explain_sql, params)
                return [row["QUERY PLAN"] for row in cur.fetchall()]
