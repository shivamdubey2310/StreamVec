from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence

import psycopg
from pgvector.psycopg import register_vector

from worker.chunker import TextChunk


@dataclass
class PreparedDocumentWrite:
    doc_id: str
    source: str
    title: str
    content_hash: str
    published_at: datetime
    chunks: List[TextChunk]
    embeddings: List[List[float]]
    model_version: str


class PostgresVectorWriter:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.conn = psycopg.connect(self.dsn, autocommit=True)
        register_vector(self.conn)

    def close(self) -> None:
        if not self.conn.closed:
            self.conn.close()

    def get_existing_hashes(self, doc_ids: Iterable[str]) -> Dict[str, str]:
        ids = list(doc_ids)
        if not ids:
            return {}
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, content_hash FROM documents WHERE doc_id = ANY(%s)",
                (ids,),
            )
            return {row[0]: row[1] for row in cur.fetchall()}

    def apply_batch(
        self,
        writes: Sequence[PreparedDocumentWrite],
        tombstone_doc_ids: Sequence[str],
    ) -> Dict[str, int]:
        """
        Atomically applies document inserts, ACID chunk replacements, and tombstone deletes
        within a single PostgreSQL transaction.
        """
        inserted = 0
        updated = 0
        deleted = 0

        with self.conn.transaction():
            with self.conn.cursor() as cur:
                # 1. Process tombstone deletions (ON DELETE CASCADE cleans document_chunks)
                if tombstone_doc_ids:
                    cur.execute(
                        "DELETE FROM documents WHERE doc_id = ANY(%s) RETURNING doc_id",
                        (list(tombstone_doc_ids),),
                    )
                    deleted = len(cur.fetchall())

                # 2. Process new documents & ACID updates
                for item in writes:
                    cur.execute(
                        "SELECT 1 FROM documents WHERE doc_id = %s",
                        (item.doc_id,),
                    )
                    existed = cur.fetchone() is not None

                    cur.execute(
                        """
                        INSERT INTO documents (doc_id, source, title, content_hash, published_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (doc_id) DO UPDATE SET
                            source = EXCLUDED.source,
                            title = EXCLUDED.title,
                            content_hash = EXCLUDED.content_hash,
                            published_at = EXCLUDED.published_at,
                            updated_at = NOW()
                        """,
                        (item.doc_id, item.source, item.title, item.content_hash, item.published_at),
                    )

                    if existed:
                        # Wipe stale chunks before inserting revised chunks (prevents orphan vectors)
                        cur.execute("DELETE FROM document_chunks WHERE doc_id = %s", (item.doc_id,))
                        updated += 1
                    else:
                        inserted += 1

                    if item.chunks:
                        rows = [
                            (
                                item.doc_id,
                                item.source,
                                item.title,
                                chunk.chunk_index,
                                chunk.chunk_text,
                                emb,
                                item.model_version,
                                item.published_at,
                            )
                            for chunk, emb in zip(item.chunks, item.embeddings)
                        ]
                        cur.executemany(
                            """
                            INSERT INTO document_chunks (
                                doc_id, source, title, chunk_index, chunk_text,
                                embedding, embedding_model_version, published_at, indexed_at
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                            """,
                            rows,
                        )

        return {"inserted": inserted, "updated": updated, "deleted": deleted}
