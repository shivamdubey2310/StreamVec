import argparse
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_ARROW_SCHEMA = pa.schema([
    ("chunk_id", pa.int64()),
    ("doc_id", pa.string()),
    ("source", pa.string()),
    ("title", pa.string()),
    ("chunk_index", pa.int32()),
    ("chunk_text", pa.string()),
    ("embedding", pa.list_(pa.float32(), 384)),
    ("embedding_model_version", pa.string()),
    ("published_at", pa.timestamp("us", tz="UTC")),
    ("indexed_at", pa.timestamp("us", tz="UTC")),
])


def archive_cold_chunks(
    older_than_days: int = 30,
    archive_root: str = "archive",
    batch_limit: int = 10000,
) -> Dict[str, Any]:
    pg_dsn = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'streamvec')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'streamvec_pass')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'streamvec')}"
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)

    with psycopg.connect(pg_dsn, autocommit=True, row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    chunk_id, doc_id, source, title, chunk_index, chunk_text,
                    embedding, embedding_model_version, published_at, indexed_at
                FROM document_chunks
                WHERE indexed_at < %s
                ORDER BY indexed_at ASC
                LIMIT %s
                """,
                (cutoff, batch_limit),
            )
            rows = cur.fetchall()

        if not rows:
            logger.info("No chunks older than %d days (cutoff=%s). Nothing to archive.", older_than_days, cutoff.isoformat())
            return {"archived_chunks": 0, "files_written": 0, "pruned_empty_docs": 0}

        # Group records by Hive partition: year=YYYY/month=MM
        partitions: Dict[tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        archived_ids: List[int] = []
        for row in rows:
            row["embedding"] = [float(x) for x in row["embedding"]]
            ts: datetime = row["indexed_at"]
            partitions[(ts.year, ts.month)].append(row)
            archived_ids.append(row["chunk_id"])

        run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        files_written = 0

        for (year, month), part_rows in partitions.items():
            part_dir = Path(archive_root) / f"year={year:04d}" / f"month={month:02d}"
            part_dir.mkdir(parents=True, exist_ok=True)
            out_file = part_dir / f"chunks_{run_tag}.parquet"

            table = pa.Table.from_pylist(part_rows, schema=ARCHIVE_ARROW_SCHEMA)
            pq.write_table(table, out_file, compression="zstd")
            files_written += 1
            logger.info("Wrote %d cold chunks to %s", len(part_rows), out_file)

        # Prune archived chunks from hot PostgreSQL storage only after Parquet flush succeeds
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE chunk_id = ANY(%s)", (archived_ids,))
                cur.execute(
                    """
                    DELETE FROM documents d
                    WHERE NOT EXISTS (
                        SELECT 1 FROM document_chunks dc WHERE dc.doc_id = d.doc_id
                    )
                    """
                )
                pruned_docs = cur.rowcount

    summary = {
        "archived_chunks": len(archived_ids),
        "files_written": files_written,
        "pruned_empty_docs": pruned_docs,
    }
    logger.info("Cold archival complete: %s", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StreamVec Cold-Tier Parquet Archiver")
    parser.add_argument("--older-than-days", type=int, default=30)
    parser.add_argument("--archive-root", default="archive")
    parser.add_argument("--batch-limit", type=int, default=10000)
    args = parser.parse_args()
    archive_cold_chunks(
        older_than_days=args.older_than_days,
        archive_root=args.archive_root,
        batch_limit=args.batch_limit,
    )
