from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language or exact-identifier query")
    top_k: int = Field(default=5, ge=1, le=50)
    mode: Literal["hybrid", "vector_only"] = Field(default="hybrid")
    source_filter: Optional[str] = Field(default=None, description="Optional filter e.g. 'arxiv' or 'wikimedia'")
    model_version: str = Field(default="bge-small-en-v1.5-384d")


class SearchHit(BaseModel):
    chunk_id: int
    doc_id: str
    source: str
    title: str
    chunk_index: int
    chunk_text: str
    rrf_score: float
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    indexed_at: datetime


class SearchResponse(BaseModel):
    query: str
    mode: str
    total_hits: int
    embed_latency_ms: float
    db_latency_ms: float
    total_latency_ms: float
    results: List[SearchHit]
