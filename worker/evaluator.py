import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import httpx
import psycopg
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_mrr_and_hit_at_k(
    ranked_results: List[Dict[str, Any]],
    expected_match: str,
    k: int = 5,
) -> tuple[float, float]:
    """
    Computes Reciprocal Rank (1/rank) and Hit@K (1.0 or 0.0) for a single query.
    """
    target = expected_match.lower()
    for idx, hit in enumerate(ranked_results[:k], start=1):
        haystack = f"{hit.get('doc_id', '')} {hit.get('title', '')} {hit.get('chunk_text', '')}".lower()
        if target in haystack:
            return 1.0 / idx, 1.0
    return 0.0, 0.0


def evaluate_search_mode(
    api_base_url: str,
    golden_queries: List[Dict[str, str]],
    mode: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    reciprocal_ranks: List[float] = []
    hits_at_k: List[float] = []
    latencies_ms: List[float] = []

    with httpx.Client(base_url=api_base_url, timeout=15.0) as client:
        for item in golden_queries:
            resp = client.post(
                "/search",
                json={"query": item["query"], "top_k": top_k, "mode": mode},
            )
            resp.raise_for_status()
            payload = resp.json()

            rr, hit = compute_mrr_and_hit_at_k(payload["results"], item["expected_match"], k=top_k)
            reciprocal_ranks.append(rr)
            hits_at_k.append(hit)
            latencies_ms.append(payload["total_latency_ms"])

    num_q = len(golden_queries)
    mrr = round(sum(reciprocal_ranks) / max(num_q, 1), 4)
    hit_rate = round(sum(hits_at_k) / max(num_q, 1), 4)
    avg_lat = round(sum(latencies_ms) / max(num_q, 1), 2)

    return {
        "search_mode": mode,
        "num_queries": num_q,
        "hit_at_5": hit_rate,
        "mrr": mrr,
        "avg_latency_ms": avg_lat,
    }


def log_eval_result_to_postgres(pg_dsn: str, metrics: Dict[str, Any]) -> None:
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO retrieval_eval_logs (search_mode, num_queries, hit_at_5, mrr, avg_latency_ms)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    metrics["search_mode"],
                    metrics["num_queries"],
                    metrics["hit_at_5"],
                    metrics["mrr"],
                    metrics["avg_latency_ms"],
                ),
            )


def run_evaluation(
    golden_path: str = "tests/golden_queries.json",
    api_url: str = "http://localhost:8000",
    min_hybrid_mrr: float = 0.75,
) -> Dict[str, Any]:
    pg_dsn = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'streamvec')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'streamvec_pass')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'streamvec')}"
    )

    golden_queries = json.loads(Path(golden_path).read_text(encoding="utf-8"))

    vector_metrics = evaluate_search_mode(api_url, golden_queries, mode="vector_only", top_k=5)
    hybrid_metrics = evaluate_search_mode(api_url, golden_queries, mode="hybrid", top_k=5)

    log_eval_result_to_postgres(pg_dsn, vector_metrics)
    log_eval_result_to_postgres(pg_dsn, hybrid_metrics)

    mrr_lift_pct = round(
        ((hybrid_metrics["mrr"] - vector_metrics["mrr"]) / max(vector_metrics["mrr"], 1e-6)) * 100.0,
        2,
    )

    summary = {
        "vector_only": vector_metrics,
        "hybrid_rrf": hybrid_metrics,
        "mrr_lift_pct": mrr_lift_pct,
    }
    logger.info("Evaluation Summary: %s", json.dumps(summary, indent=2))

    if hybrid_metrics["mrr"] < min_hybrid_mrr:
        raise RuntimeError(
            f"Retrieval Drift Gate FAILED: Hybrid MRR {hybrid_metrics['mrr']} < threshold {min_hybrid_mrr}"
        )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StreamVec Golden-Set Retrieval Evaluator")
    parser.add_argument("--golden-path", default="tests/golden_queries.json")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--min-mrr", type=float, default=0.75)
    args = parser.parse_args()
    run_evaluation(golden_path=args.golden_path, api_url=args.api_url, min_hybrid_mrr=args.min_mrr)
