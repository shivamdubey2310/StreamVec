from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from producer.dlq import DLQRouter
from producer.schemas import DocumentEvent
from worker.evaluator import compute_mrr_and_hit_at_k


def test_document_event_enforces_non_empty_content_unless_deleted():
    with pytest.raises(ValidationError):
        DocumentEvent(doc_id="doc-1", source="arxiv", title="", body="Valid length body text over twenty chars.")

    with pytest.raises(ValidationError):
        DocumentEvent(doc_id="doc-2", source="arxiv", title="Valid Title", body="short")

    # Tombstone events allow empty title and body
    tombstone = DocumentEvent(doc_id="doc-3", source="arxiv", is_deleted=True)
    assert tombstone.is_deleted is True


def test_dlq_router_quarantines_malformed_json_and_schema_errors():
    mock_producer = MagicMock()
    router = DLQRouter(producer=mock_producer, dlq_topic="documents_dlq")

    ev1, ok1 = router.validate_or_quarantine(b'{"doc_id": "bad", broken_json', source_stage="unit_test")
    ev2, ok2 = router.validate_or_quarantine(
        {"doc_id": "doc-empty", "source": "arxiv", "title": "T", "body": ""},
        source_stage="unit_test",
    )
    ev3, ok3 = router.validate_or_quarantine(
        {
            "doc_id": "doc-valid",
            "source": "arxiv",
            "title": "Valid Title",
            "body": "Sufficiently long document body for schema validation.",
        },
        source_stage="unit_test",
    )

    assert ok1 is False and ev1 is None
    assert ok2 is False and ev2 is None
    assert ok3 is True and ev3 is not None
    assert router.quarantined_count == 2
    assert mock_producer.produce.call_count == 2

    # Verify diagnostic headers were attached to DLQ messages
    call_kwargs = mock_producer.produce.call_args_list[0].kwargs
    header_keys = {k for k, _ in call_kwargs["headers"]}
    assert {"error_reason", "source_stage", "failed_at"}.issubset(header_keys)


def test_rrf_formula_ranks_dual_match_above_single_modality_match():
    rrf_k = 60
    # Doc A ranks #2 in vector and #1 in keyword -> 1/62 + 1/61
    rrf_dual = (1.0 / (rrf_k + 2)) + (1.0 / (rrf_k + 1))
    # Doc B ranks #1 in vector only -> 1/61
    rrf_vector_only = 1.0 / (rrf_k + 1)

    assert rrf_dual > rrf_vector_only


def test_compute_mrr_and_hit_at_k():
    results = [
        {"doc_id": "d1", "title": "Unrelated Paper", "chunk_text": "Nothing here"},
        {"doc_id": "d2", "title": "Patch Analysis CVE-2024-3094", "chunk_text": "Details"},
        {"doc_id": "d3", "title": "Other", "chunk_text": "More text"},
    ]
    rr, hit = compute_mrr_and_hit_at_k(results, expected_match="CVE-2024-3094", k=5)
    assert rr == pytest.approx(0.5)
    assert hit == 1.0

    rr_miss, hit_miss = compute_mrr_and_hit_at_k(results, expected_match="NON_EXISTENT_ID", k=5)
    assert rr_miss == 0.0
    assert hit_miss == 0.0
