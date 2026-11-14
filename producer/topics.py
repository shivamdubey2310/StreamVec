import logging
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import KafkaException

logger = logging.getLogger(__name__)


def ensure_topics_exist(
    bootstrap_servers: str,
    raw_topic: str = "raw_documents",
    dlq_topic: str = "documents_dlq",
    raw_partitions: int = 3,
) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    existing = admin.list_topics(timeout=10).topics

    desired_topics = []
    if raw_topic not in existing:
        desired_topics.append(
            NewTopic(
                topic=raw_topic,
                num_partitions=raw_partitions,
                replication_factor=1,
                config={"retention.ms": str(7 * 24 * 3600 * 1000)},
            )
        )
    if dlq_topic not in existing:
        desired_topics.append(
            NewTopic(
                topic=dlq_topic,
                num_partitions=1,
                replication_factor=1,
                config={"retention.ms": str(14 * 24 * 3600 * 1000)},
            )
        )

    if not desired_topics:
        logger.info("Kafka topics already provisioned: %s, %s", raw_topic, dlq_topic)
        return

    futures = admin.create_topics(desired_topics)
    for topic_name, future in futures.items():
        try:
            future.result()
            logger.info("Provisioned Kafka topic: %s", topic_name)
        except KafkaException as exc:
            # Handle concurrent creation race safely
            if "TOPIC_ALREADY_EXISTS" not in str(exc):
                raise
