from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class DocumentEvent(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=256, description="Unique document identifier")
    source: str = Field(..., min_length=1, max_length=64, description="Upstream origin e.g. arxiv, wikimedia")
    title: str = Field(default="", max_length=1024)
    body: str = Field(default="")
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_deleted: bool = Field(default=False, description="Tombstone flag for downstream vector deletion")

    @model_validator(mode="after")
    def validate_content_unless_deleted(self) -> "DocumentEvent":
        if not self.is_deleted:
            if not self.title.strip():
                raise ValueError("title must be non-empty when is_deleted is False")
            if len(self.body.strip()) < 20:
                raise ValueError("body must contain at least 20 non-whitespace characters when is_deleted is False")
        return self

    def to_kafka_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")
