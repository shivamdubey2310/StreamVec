import hashlib
import re
from typing import Dict, Iterable, Optional

import redis


def compute_content_hash(title: str, body: str) -> str:
    """
    Computes a deterministic 64-character SHA-256 hex digest over normalized title + body.
    """
    norm_title = re.sub(r"\s+", " ", title).strip()
    norm_body = re.sub(r"\s+", " ", body).strip()
    canonical = f"{norm_title}\n{norm_body}".encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class RedisDedupCache:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        ttl_seconds: int = 2592000,
        key_prefix: str = "streamvec:doc_hash:",
    ) -> None:
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix

    def _key(self, doc_id: str) -> str:
        return f"{self.key_prefix}{doc_id}"

    def get_hashes_batch(self, doc_ids: Iterable[str]) -> Dict[str, Optional[str]]:
        ids = list(doc_ids)
        if not ids:
            return {}
        keys = [self._key(doc_id) for doc_id in ids]
        values = self.client.mget(keys)
        return dict(zip(ids, values))

    def set_hash(self, doc_id: str, content_hash: str) -> None:
        self.client.set(self._key(doc_id), content_hash, ex=self.ttl_seconds)

    def delete_hash(self, doc_id: str) -> None:
        self.client.delete(self._key(doc_id))
