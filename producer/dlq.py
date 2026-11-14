import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from confluent_kafka import Producer
from pydantic import ValidationError

from producer.schemas import DocumentEvent

logger = logging.getLogger(__name__)


class DLQRouter:
    def __init__(self, producer: Producer, dlq_topic: str = "documents_dlq") -> None:
        self.producer = producer
        self.dlq_topic = dlq_topic
        self.quarantined_count = 0

    def send_to_dlq(
        self,
        raw_payload: bytes | str,
        error_reason: str,
        source_stage: str = "producer_validation",
        key: Optional[str] = None,
    ) -> None:
        payload_bytes = raw_payload.encode("utf-8", errors="replace") if isinstance(raw_payload, str) else raw_payload
        headers = [
            ("error_reason", error_reason[:512].encode("utf-8")),
            ("source_stage", source_stage.encode("utf-8")),
            ("failed_at", datetime.now(timezone.utc).isoformat().encode("utf-8")),
        ]
        self.producer.produce(
            topic=self.dlq_topic,
            key=key.encode("utf-8") if key else b"unknown-key",
            value=payload_bytes,
            headers=headers,
        )
        self.producer.poll(0)
        self.quarantined_count += 1
        logger.warning("Quarantined malformed event to %s | reason=%s", self.dlq_topic, error_reason)

    def validate_or_quarantine(
        self,
        raw_record: bytes | str | dict[str, Any],
        source_stage: str = "producer_validation",
    ) -> Tuple[Optional[DocumentEvent], bool]:
        """
        Validates a raw record into a DocumentEvent.
        Returns (event, True) on success, or (None, False) after routing to DLQ on failure.
        """
        try:
            if isinstance(raw_record, bytes):
                parsed = json.loads(raw_record.decode("utf-8"))
            elif isinstance(raw_record, str):
                parsed = json.loads(raw_record)
            elif isinstance(raw_record, dict):
                parsed = raw_record
            else:
                raise TypeError(f"Unsupported payload type: {type(raw_record).__name__}")

            if not isinstance(parsed, dict):
                raise ValueError("JSON payload must be an object/dictionary")

            event = DocumentEvent.model_validate(parsed)
            return event, True

        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raw_bytes = raw_record if isinstance(raw_record, bytes) else str(raw_record).encode("utf-8")
            self.send_to_dlq(raw_bytes, error_reason=f"JSONDecodeError: {exc}", source_stage=source_stage)
            return None, False

        except (ValidationError, ValueError, TypeError) as exc:
            extracted_key = parsed.get("doc_id") if isinstance(locals().get("parsed"), dict) else None
            raw_bytes = (
                json.dumps(raw_record).encode("utf-8")
                if isinstance(raw_record, dict)
                else (raw_record if isinstance(raw_record, bytes) else str(raw_record).encode("utf-8"))
            )
            self.send_to_dlq(
                raw_bytes,
                error_reason=f"SchemaValidationError: {exc}",
                source_stage=source_stage,
                key=str(extracted_key) if extracted_key else None,
            )
            return None, False
