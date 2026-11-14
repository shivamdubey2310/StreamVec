import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from api.models import SearchRequest, SearchResponse
from api.retrieval import HybridRetriever
from worker.embedder import LocalBatchEmbedder

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pg_dsn = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'streamvec')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'streamvec_pass')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'streamvec')}"
    )
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    rrf_k = int(os.getenv("RRF_K", "60"))

    app.state.embedder = LocalBatchEmbedder(model_name=model_name, batch_size=1)
    app.state.retriever = HybridRetriever(dsn=pg_dsn, rrf_k=rrf_k)
    yield
    app.state.retriever.close()


app = FastAPI(
    title="StreamVec Hybrid RAG Retrieval API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "streamvec-api"}


@app.post("/search", response_model=SearchResponse)
def search_chunks(payload: SearchRequest) -> SearchResponse:
    total_start = time.perf_counter()

    embed_start = time.perf_counter()
    query_vec = app.state.embedder.embed_query(payload.query)
    embed_ms = round((time.perf_counter() - embed_start) * 1000.0, 2)

    hits, db_ms = app.state.retriever.search(
        query_text=payload.query,
        query_embedding=query_vec,
        top_k=payload.top_k,
        mode=payload.mode,
        source_filter=payload.source_filter,
        model_version=payload.model_version,
    )

    total_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
    return SearchResponse(
        query=payload.query,
        mode=payload.mode,
        total_hits=len(hits),
        embed_latency_ms=embed_ms,
        db_latency_ms=db_ms,
        total_latency_ms=total_ms,
        results=hits,
    )


@app.post("/explain")
def explain_search(payload: SearchRequest):
    query_vec = app.state.embedder.embed_query(payload.query)
    plan_lines = app.state.retriever.explain_plan(
        query_text=payload.query,
        query_embedding=query_vec,
        top_k=payload.top_k,
        model_version=payload.model_version,
    )
    return {"query": payload.query, "plan": plan_lines}
