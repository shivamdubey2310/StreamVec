import os
import sys
from datetime import datetime, timedelta

from airflow.sdk import DAG, task

PROJECT_ROOT = "/opt/airflow/project"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from worker.archiver import archive_cold_chunks
from worker.evaluator import run_evaluation

default_args = {
    "owner": "streamvec-data-eng",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="streamvec_governance_and_archival",
    description="Daily retrieval drift check (MRR & Hit@5) + >30d cold Parquet archival",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["streamvec", "rag-governance", "cold-archival"],
) as dag:

    @task(task_id="retrieval_quality_drift_gate")
    def evaluate_retrieval_quality():
        api_url = os.getenv("STREAMVEC_API_URL", "http://host.docker.internal:8000")
        golden_path = os.path.join(PROJECT_ROOT, "tests", "golden_queries.json")
        return run_evaluation(
            golden_path=golden_path,
            api_url=api_url,
            min_hybrid_mrr=0.75,
        )

    @task(task_id="archive_cold_vectors_to_parquet")
    def run_cold_archival(eval_summary: dict):
        archive_dir = os.path.join(PROJECT_ROOT, "archive")
        result = archive_cold_chunks(
            older_than_days=30,
            archive_root=archive_dir,
            batch_limit=10000,
        )
        return {"eval_mrr": eval_summary["hybrid_rrf"]["mrr"], "archival": result}

    eval_metrics = evaluate_retrieval_quality()
    run_cold_archival(eval_metrics)
