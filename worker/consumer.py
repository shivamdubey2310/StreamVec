import argparse
import json
import logging
import os
import signal
import time
from typing import Dict, List

from confluent_kafka import Consumer, KafkaError, Producer
from dotenv import load_dotenv

from producer.dlq import DLQRouter
from producer.schemas import DocumentEvent
from producer.topics import ensure_topics_exist
from worker.chunker import semantic_chunk_text
from worker.db_writer import PostgresVectorWriter, PreparedDocumentWrite
from worker.dedup import RedisDedupCache, compute_content_hash
from worker.embedder import LocalBatchEmbedder

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StreamVecWorker:
    def __init__(self) -> None:
        self.running = True
        self.bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.raw_topic = os.getenv("KAFKA_RAW_TOPIC", "raw_documents")
        self.dlq_topic = os.getenv("KAFKA_DLQ_TOPIC", "documents_dlq")
        self.group_id = os.getenv("KAFKA_CONSUMER_GROUP", "streamvec-workers")

        self.batch_size = int(os.getenv("WORKER_BATCH_SIZE", "32"))
        self.batch_timeout = float(os.getenv("WORKER_BATCH_TIMEOUT_SEC", "1.0"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE_CHARS", "800"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP_CHARS", "120"))
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
        self.model_version = os.getenv("EMBEDDING_MODEL_VERSION", "bge-small-en-v1.5-384d")

        pg_dsn = (
            f"postgresql://{os.getenv('POSTGRES_USER', 'streamvec')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'streamvec_pass')}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB', 'streamvec')}"
        )

        ensure_topics_exist(self.bootstrap, raw_topic=self.raw_topic, dlq_topic=self.dlq_topic)

        self.consumer = Consumer({
            "bootstrap.servers": self.bootstrap,
            "group.id": self.group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": 300000,
        })
        self.dlq_producer = Producer({"bootstrap.servers": self.bootstrap, "acks": "all"})
        self.dlq = DLQRouter(producer=self.dlq_producer, dlq_topic=self.dlq_topic)

        self.dedup = RedisDedupCache(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            ttl_seconds=int(os.getenv("REDIS_DEDUP_TTL_SECONDS", "2592000")),
        )
        self.embedder = LocalBatchEmbedder(model_name=self.model_name, batch_size=self.batch_size)
        self.writer = PostgresVectorWriter(dsn=pg_dsn)

        self.metrics = {
            "events_consumed": 0,
            "events_skipped_dedup": 0,
            "events_inserted": 0,
            "events_updated_acid": 0,
            "events_deleted_tombstone": 0,
            "events_quarantined_dlq": 0,
            "chunks_indexed": 0,
        }

    def stop(self, *_) -> None:
        logger.info("Shutdown signal received; finishing current micro-batch...")
        self.running = False

    def process_micro_batch(self, raw_messages) -> None:
        valid_events: List[DocumentEvent] = []
        for msg in raw_messages:
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error("Kafka consumer error: %s", msg.error())
                continue
            self.metrics["events_consumed"] += 1
            event, ok = self.dlq.validate_or_quarantine(msg.value(), source_stage="worker_consumer")
            if not ok or event is None:
                self.metrics["events_quarantined_dlq"] += 1
                continue
            valid_events.append(event)

        if not valid_events:
            self.dlq_producer.flush(5)
            self.consumer.commit(asynchronous=False)
            return

        # Coalesce multiple updates for the same doc_id within the same micro-batch (latest wins)
        latest_by_doc: Dict[str, DocumentEvent] = {}
        for ev in valid_events:
            latest_by_doc[ev.doc_id] = ev

        tombstones: List[str] = []
        candidates: List[tuple[DocumentEvent, str]] = []
        for doc_id, ev in latest_by_doc.items():
            if ev.is_deleted:
                tombstones.append(doc_id)
            else:
                candidates.append((ev, compute_content_hash(ev.title, ev.body)))

        # Step 1: Fast batch lookup in Redis, with Postgres fallback for evicted keys
        candidate_ids = [ev.doc_id for ev, _ in candidates]
        redis_hashes = self.dedup.get_hashes_batch(candidate_ids)
        missing_in_redis = [doc_id for doc_id, h in redis_hashes.items() if h is None]
        pg_hashes = self.writer.get_existing_hashes(missing_in_redis) if missing_in_redis else {}

        to_embed: List[tuple[DocumentEvent, str]] = []
        intra_batch_dedup = len(valid_events) - len(latest_by_doc)
        skipped_cache = 0

        for ev, new_hash in candidates:
            known_hash = redis_hashes.get(ev.doc_id) or pg_hashes.get(ev.doc_id)
            if known_hash == new_hash:
                skipped_cache += 1
                # Re-warm Redis if hit came from Postgres fallback
                if redis_hashes.get(ev.doc_id) is None:
                    self.dedup.set_hash(ev.doc_id, new_hash)
            else:
                to_embed.append((ev, new_hash))

        self.metrics["events_skipped_dedup"] += intra_batch_dedup + skipped_cache

        # Step 2: Semantic chunking + single vectorized batch ONNX inference
        prepared_writes: List[PreparedDocumentWrite] = []
        all_passages: List[str] = []
        doc_chunk_ranges: List[tuple[DocumentEvent, str, list, int, int]] = []

        for ev, new_hash in to_embed:
            chunks = semantic_chunk_text(
                ev.body,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            start_idx = len(all_passages)
            for ch in chunks:
                all_passages.append(f"{ev.title}: {ch.chunk_text}")
            end_idx = len(all_passages)
            doc_chunk_ranges.append((ev, new_hash, chunks, start_idx, end_idx))

        all_embeddings = self.embedder.embed_passages(all_passages) if all_passages else []

        for ev, new_hash, chunks, start_idx, end_idx in doc_chunk_ranges:
            prepared_writes.append(
                PreparedDocumentWrite(
                    doc_id=ev.doc_id,
                    source=ev.source,
                    title=ev.title,
                    content_hash=new_hash,
                    published_at=ev.published_at,
                    chunks=chunks,
                    embeddings=all_embeddings[start_idx:end_idx],
                    model_version=self.model_version,
                )
            )

        # Step 3: Commit ACID transaction to PostgreSQL, then update Redis & commit Kafka offsets
        db_counts = self.writer.apply_batch(writes=prepared_writes, tombstone_doc_ids=tombstones)

        for item in prepared_writes:
            self.dedup.set_hash(item.doc_id, item.content_hash)
        for doc_id in tombstones:
            self.dedup.delete_hash(doc_id)

        self.dlq_producer.flush(5)
        self.consumer.commit(asynchronous=False)

        self.metrics["events_inserted"] += db_counts["inserted"]
        self.metrics["events_updated_acid"] += db_counts["updated"]
        self.metrics["events_deleted_tombstone"] += db_counts["deleted"]
        self.metrics["chunks_indexed"] += len(all_passages)

        logger.info("Micro-batch committed | %s", json.dumps(self.metrics))

    def run(self, idle_exit_sec: float | None = None) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        self.consumer.subscribe([self.raw_topic])
        logger.info(
            "StreamVec worker subscribed to '%s' (batch_size=%d, model=%s)",
            self.raw_topic,
            self.batch_size,
            self.model_name,
        )

        last_msg_time = time.monotonic()
        try:
            while self.running:
                messages = self.consumer.consume(num_messages=self.batch_size, timeout=self.batch_timeout)
                if not messages:
                    if idle_exit_sec is not None and (time.monotonic() - last_msg_time) >= idle_exit_sec:
                        logger.info("Idle timeout (%.1fs) reached; exiting cleanly.", idle_exit_sec)
                        break
                    continue
                last_msg_time = time.monotonic()
                self.process_micro_batch(messages)
        finally:
            self.consumer.close()
            self.writer.close()
            logger.info("Final worker metrics | %s", json.dumps(self.metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StreamVec Micro-Batch Stream Processor")
    parser.add_argument(
        "--idle-exit-sec",
        type=float,
        default=None,
        help="Exit automatically after N seconds with no new Kafka messages (useful for batch/test runs)",
    )
    args = parser.parse_args()
    StreamVecWorker().run(idle_exit_sec=args.idle_exit_sec)
