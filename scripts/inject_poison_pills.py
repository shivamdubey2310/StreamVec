import os
from confluent_kafka import Producer
from dotenv import load_dotenv

from producer.dlq import DLQRouter
from producer.topics import ensure_topics_exist

load_dotenv()


def main() -> None:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    raw_topic = os.getenv("KAFKA_RAW_TOPIC", "raw_documents")
    dlq_topic = os.getenv("KAFKA_DLQ_TOPIC", "documents_dlq")

    ensure_topics_exist(bootstrap, raw_topic=raw_topic, dlq_topic=dlq_topic)

    producer = Producer({"bootstrap.servers": bootstrap, "acks": "all"})
    dlq = DLQRouter(producer=producer, dlq_topic=dlq_topic)

    poison_samples = [
        b'{"doc_id": "broken-json-1", "title": unquoted}',
        b"",
        {"doc_id": "missing-body-1", "source": "arxiv", "title": "Valid Title", "body": ""},
        {"doc_id": "", "source": "arxiv", "title": "Empty ID", "body": "A sufficiently long body text."},
        {"unexpected_schema": True, "foo": "bar"},
    ]

    for i in range(50):
        sample = poison_samples[i % len(poison_samples)]
        dlq.validate_or_quarantine(sample, source_stage="fault_injection_test")

    producer.flush(timeout=10)
    print(f"Injected and quarantined {dlq.quarantined_count} poison-pill events to '{dlq_topic}'.")


if __name__ == "__main__":
    main()
