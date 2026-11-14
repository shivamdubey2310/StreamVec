import argparse
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Iterator

import httpx
from confluent_kafka import Producer
from dotenv import load_dotenv

from producer.schemas import DocumentEvent
from producer.topics import ensure_topics_exist

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_TOPICS = [
    ("HNSW Graph Indexing in pgvector", "Hierarchical Navigable Small World (HNSW) graphs accelerate approximate nearest neighbor search in PostgreSQL pgvector using multi-layer proximity graphs with ef_construction=64 and m=16."),
    ("Reciprocal Rank Fusion for Hybrid RAG", "Reciprocal Rank Fusion (RRF) merges dense cosine similarity rankings with sparse BM25 tsvector GIN keyword rankings using score = 1/(k + rank_vec) + 1/(k + rank_kw) with k=60."),
    ("Kafka KRaft Metadata Architecture", "Apache Kafka KRaft mode eliminates Zookeeper dependencies by embedding an event-driven Raft consensus controller quorum directly inside broker nodes, saving 500MB RAM."),
    ("Mitigating CVE-2024-3094 Supply Chain Risks", "Security patch analysis for CVE-2024-3094 highlights deterministic SHA-256 artifact verification, isolated build sandboxes, and automated dependency hash pinning."),
    ("Idempotent Micro-Batch Stream Processing", "Computing SHA-256 content hashes against a low-latency Redis cache prevents redundant ONNX embedding inference when upstream CDC streams replay unchanged records."),
]


def generate_synthetic_events(count: int, duplicate_ratio: float = 0.2, update_ratio: float = 0.15) -> Iterator[DocumentEvent]:
    seen_docs: list[DocumentEvent] = []
    for i in range(count):
        roll = random.random()
        if seen_docs and roll < duplicate_ratio:
            # Exact duplicate event (refreshed timestamp, identical title + body)
            orig = random.choice(seen_docs)
            yield DocumentEvent(
                doc_id=orig.doc_id,
                source=orig.source,
                title=orig.title,
                body=orig.body,
                published_at=datetime.now(timezone.utc),
            )
        elif seen_docs and roll < (duplicate_ratio + update_ratio):
            # Revised document body (triggers ACID chunk replacement downstream)
            orig = random.choice(seen_docs)
            updated = DocumentEvent(
                doc_id=orig.doc_id,
                source=orig.source,
                title=orig.title,
                body=f"{orig.body} Revision {i}: updated benchmark telemetry confirms sub-20ms p95 query latency under concurrent load.",
                published_at=datetime.now(timezone.utc),
            )
            seen_docs.append(updated)
            yield updated
        else:
            title_base, body_base = random.choice(SAMPLE_TOPICS)
            doc = DocumentEvent(
                doc_id=f"arxiv-2026.{10000 + i}",
                source="arxiv",
                title=f"{title_base} — Study #{i}",
                body=f"{body_base} Document identifier arxiv-2026.{10000 + i} evaluates throughput, recall@5, and memory bounds on CPU hardware.",
                published_at=datetime.now(timezone.utc),
            )
            seen_docs.append(doc)
            yield doc


def stream_wikimedia_events(limit: int) -> Iterator[DocumentEvent]:
    url = "https://stream.wikimedia.org/v2/stream/recentchange"
    headers = {"User-Agent": "StreamVec-DataEngineering-Pipeline/1.0"}
    emitted = 0
    with httpx.stream("GET", url, headers=headers, timeout=30.0) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if payload.get("wiki") != "enwiki" or payload.get("type") not in {"edit", "new", "log"}:
                continue
            is_deleted = payload.get("type") == "log" and payload.get("log_action") == "delete"
            comment = payload.get("comment") or ""
            title = payload.get("title") or "Untitled"
            body = f"Wikipedia revision on {title} by {payload.get('user', 'unknown')}. Edit summary: {comment}. Server: {payload.get('server_name', 'en.wikipedia.org')}."
            if not is_deleted and len(body.strip()) < 20:
                continue
            yield DocumentEvent(
                doc_id=f"wiki-{payload.get('page_id', payload.get('id'))}",
                source="wikimedia",
                title=title,
                body="" if is_deleted else body,
                published_at=datetime.now(timezone.utc),
                is_deleted=is_deleted,
            )
            emitted += 1
            if emitted >= limit:
                break


def run_producer(mode: str, count: int) -> None:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    raw_topic = os.getenv("KAFKA_RAW_TOPIC", "raw_documents")
    dlq_topic = os.getenv("KAFKA_DLQ_TOPIC", "documents_dlq")

    ensure_topics_exist(bootstrap, raw_topic=raw_topic, dlq_topic=dlq_topic)

    producer = Producer({
        "bootstrap.servers": bootstrap,
        "acks": "all",
        "linger.ms": 10,
        "compression.type": "lz4",
    })

    events = generate_synthetic_events(count) if mode == "synthetic" else stream_wikimedia_events(count)
    delivered = 0

    def on_delivery(err, msg):
        nonlocal delivered
        if err is not None:
            logger.error("Delivery failed for key=%s: %s", msg.key(), err)
        else:
            delivered += 1

    start = time.perf_counter()
    for event in events:
        producer.produce(
            topic=raw_topic,
            key=event.doc_id.encode("utf-8"),
            value=event.to_kafka_bytes(),
            on_delivery=on_delivery,
        )
        producer.poll(0)

    producer.flush(timeout=15)
    elapsed = time.perf_counter() - start
    logger.info("Produced %d events to '%s' in %.2fs (%.1f events/s)", delivered, raw_topic, elapsed, delivered / max(elapsed, 1e-6))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StreamVec Kafka Event Producer")
    parser.add_argument("--mode", choices=["synthetic", "wikimedia"], default="synthetic")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    run_producer(mode=args.mode, count=args.count)