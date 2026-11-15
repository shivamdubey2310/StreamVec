import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psycopg
from confluent_kafka import Producer
from dotenv import load_dotenv

from producer.dlq import DLQRouter
from producer.topics import ensure_topics_exist
from worker.consumer import StreamVecWorker
from worker.embedder import LocalBatchEmbedder

load_dotenv()

TOPICS = [
    ("HNSW Graph Indexing in pgvector", "Hierarchical Navigable Small World (HNSW) graphs accelerate approximate nearest neighbor search in PostgreSQL pgvector using multi-layer proximity graphs with ef_construction=64 and m=16."),
    ("Reciprocal Rank Fusion for Hybrid RAG", "Reciprocal Rank Fusion (RRF) merges dense cosine similarity rankings with sparse BM25 tsvector GIN keyword rankings using score = 1/(k + rank_vec) + 1/(k + rank_kw) with k=60."),
    ("Kafka KRaft Metadata Architecture", "Apache Kafka KRaft mode eliminates Zookeeper dependencies by embedding an event-driven Raft consensus controller quorum directly inside broker nodes, saving 500MB RAM."),
    ("Mitigating CVE-2024-3094 Supply Chain Risks", "Security patch analysis for CVE-2024-3094 highlights deterministic SHA-256 artifact verification, isolated build sandboxes, and automated dependency hash pinning."),
    ("Idempotent Micro-Batch Stream Processing", "Computing SHA-256 content hashes against a low-latency Redis cache prevents redundant ONNX embedding inference when upstream CDC streams replay unchanged records."),
]


def benchmark_microbatch_speedup(sample_count: int = 256) -> dict:
    texts = [
        f"Technical benchmark passage #{i}: HNSW vector search with pgvector and BM25 GIN keyword fusion over streaming Kafka micro-batches."
        for i in range(sample_count)
    ]
    single_embedder = LocalBatchEmbedder(batch_size=1)
    t0 = time.perf_counter()
    for t in texts:
        single_embedder.embed_passages([t])
    single_sec = time.perf_counter() - t0
    single_rate = sample_count / max(single_sec, 1e-6)

    batch_embedder = LocalBatchEmbedder(batch_size=32)
    t1 = time.perf_counter()
    batch_embedder.embed_passages(texts)
    batch_sec = time.perf_counter() - t1
    batch_rate = sample_count / max(batch_sec, 1e-6)

    return {
        "single_docs_per_sec": round(single_rate, 1),
        "batch32_docs_per_sec": round(batch_rate, 1),
        "speedup_x": round(batch_rate / max(single_rate, 1e-6), 2),
    }


def produce_load_stream(total_events: int, duplicate_ratio: float = 0.40, poison_count: int = 100) -> int:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    raw_topic = os.getenv("KAFKA_RAW_TOPIC", "raw_documents")
    dlq_topic = os.getenv("KAFKA_DLQ_TOPIC", "documents_dlq")
    ensure_topics_exist(bootstrap, raw_topic=raw_topic, dlq_topic=dlq_topic)

    producer = Producer({
        "bootstrap.servers": bootstrap,
        "acks": "all",
        "linger.ms": 15,
        "batch.num.messages": 1000,
        "compression.type": "lz4",
    })
    dlq = DLQRouter(producer=producer, dlq_topic=dlq_topic)

    unique_pool = []
    for i in range(total_events):
        if unique_pool and random.random() < duplicate_ratio:
            base = random.choice(unique_pool)
            payload = {**base, "published_at": datetime.now(timezone.utc).isoformat()}
        else:
            title_base, body_base = TOPICS[i % len(TOPICS)]
            payload = {
                "doc_id": f"bench-doc-{i}",
                "source": "arxiv",
                "title": f"{title_base} — Study #{i}",
                "body": (
                    f"{body_base} Document bench-doc-{i} evaluates high-throughput streaming vector ingestion, "
                    f"sub-25ms Reciprocal Rank Fusion latency, and deterministic SHA-256 deduplication across "
                    f"multi-chunk technical abstracts in PostgreSQL 16 with pgvector HNSW and GIN indexes."
                ),
                "published_at": datetime.now(timezone.utc).isoformat(),
                "is_deleted": False,
            }
            if len(unique_pool) < 5000:
                unique_pool.append(payload)

        event, ok = dlq.validate_or_quarantine(payload, source_stage="benchmark_producer")
        if ok and event:
            producer.produce(
                topic=raw_topic,
                key=event.doc_id.encode("utf-8"),
                value=event.to_kafka_bytes(),
            )
        if i % 1000 == 0:
            producer.poll(0)

    # Inject malformed payloads to verify DLQ fault isolation at scale
    for p in range(poison_count):
        dlq.validate_or_quarantine(b'{"doc_id": "corrupt-bench", broken', source_stage="benchmark_dlq_test")

    producer.flush(timeout=30)
    return dlq.quarantined_count


def query_postgres_metrics(pg_dsn: str) -> dict:
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("ANALYZE document_chunks;")
            cur.execute("SELECT COUNT(*) FROM documents;")
            total_docs = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            total_chunks = cur.fetchone()[0]

            cur.execute(
                """
                SELECT
                    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (
                        ORDER BY EXTRACT(EPOCH FROM (indexed_at - published_at)) * 1000
                    )::numeric, 1) AS p50_ms,
                    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (
                        ORDER BY EXTRACT(EPOCH FROM (indexed_at - published_at)) * 1000
                    )::numeric, 1) AS p95_ms
                FROM document_chunks;
                """
            )
            p50_ms, p95_ms = cur.fetchone()

    return {
        "total_docs": total_docs,
        "total_chunks": total_chunks,
        "p50_ms": float(p50_ms or 0.0),
        "p95_ms": float(p95_ms or 0.0),
    }


def query_api_benchmarks(api_url: str = "http://localhost:8000") -> dict | None:
    try:
        with httpx.Client(base_url=api_url, timeout=15.0) as client:
            health = client.get("/health")
            if health.status_code != 200:
                return None

            explain_resp = client.post("/explain", json={"query": "HNSW pgvector CVE-2024-3094", "top_k": 5})
            plan_lines = explain_resp.json().get("plan", [])
            exec_line = next((l for l in plan_lines if "Execution Time:" in l), "Execution Time: N/A")
            uses_hnsw = any("idx_chunks_hnsw" in l for l in plan_lines)
            uses_gin = any("idx_chunks_gin" in l for l in plan_lines)

            from worker.evaluator import run_evaluation
            eval_summary = run_evaluation(api_url=api_url, min_hybrid_mrr=0.50)

            return {
                "explain_execution": exec_line.strip(),
                "uses_hnsw": uses_hnsw,
                "uses_gin": uses_gin,
                "eval": eval_summary,
            }
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="StreamVec 50K Load & Latency Benchmark")
    parser.add_argument("--docs", type=int, default=50000, help="Total events to stream through Kafka")
    parser.add_argument("--dup-ratio", type=float, default=0.40, help="Fraction of duplicate replay events")
    parser.add_argument("--poison", type=int, default=100, help="Malformed DLQ payloads to inject")
    args = parser.parse_args()

    pg_dsn = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'streamvec')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'streamvec_pass')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'streamvec')}"
    )

    print(f"[1/4] Benchmarking single-record vs batch-32 ONNX inference...")
    speedup = benchmark_microbatch_speedup(sample_count=256)

    print(f"[2/4] Producing {args.docs:,} events ({int(args.dup_ratio*100)}% duplicates) + {args.poison} DLQ pills...")
    dlq_count = produce_load_stream(total_events=args.docs, duplicate_ratio=args.dup_ratio, poison_count=args.poison)

    print(f"[3/4] Draining Kafka micro-batches into PostgreSQL + pgvector...")
    worker = StreamVecWorker()
    worker.run(idle_exit_sec=4.0)

    consumed = max(worker.metrics["events_consumed"], 1)
    dedup_pct = round((worker.metrics["events_skipped_dedup"] / consumed) * 100.0, 1)

    print(f"[4/4] Capturing PostgreSQL percentiles & Hybrid RRF query plan...")
    pg_stats = query_postgres_metrics(pg_dsn)
    api_stats = query_api_benchmarks()

    report = {
        "microbatch": speedup,
        "worker_metrics": worker.metrics,
        "dedup_saved_pct": dedup_pct,
        "dlq_quarantined": dlq_count,
        "postgres": pg_stats,
        "api_and_eval": api_stats,
    }
    print("\n=== STREAMVEC BENCHMARK REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
