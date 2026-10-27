"""Tests for Passage-Level NLI Claim Verification, Digest-Bound HITL Queue, and FastAPI Endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hybrid_rag.api import WorkbenchRuntime, create_app
from hybrid_rag.claim_verifier import (
    decompose_into_atomic_claims,
    extract_citation_ids,
    verify_answer_claims,
)
from hybrid_rag.hitl import HITLReviewQueue, compute_evidence_digest
from hybrid_rag.ingest import load_corpus_from_directory
from hybrid_rag.retriever import HybridRetriever
from hybrid_rag.schemas import (
    HITLDecisionRequest,
    HITLReviewStatus,
    HITLSubmitRequest,
    RetrievalMode,
    RoleEnum,
)


def test_atomic_claim_verifier_distinguishes_supported_and_unsupported_claims() -> None:
    corpus = load_corpus_from_directory()
    retriever = HybridRetriever(corpus=corpus)
    res = retriever.retrieve(
        "Under release gate GATE-AI-104, what minimum NDCG@10 score and RBAC Leakage Rate are required?",
        user_role=RoleEnum.ENGINEERING,
        mode=RetrievalMode.HYBRID_RRF_RERANK,
    )

    valid_answer = (
        "Under Enterprise Release Gate GATE-AI-104, promotion requires a minimum NDCG@10 score "
        "of 0.82 and a pre-retrieval RBAC Leakage Rate of 0.0 [DOC-AI-SOP-001#P1]."
    )
    claims, support_rate = verify_answer_claims(valid_answer, res.ranked_chunks)
    assert len(claims) == 1
    assert claims[0].is_supported is True
    assert claims[0].citation_ids_valid is True
    assert support_rate == 1.0

    # Unsupported claim with fabricated numbers and missing citation tag
    hallucinated_answer = (
        "Under release gate GATE-AI-104, the minimum NDCG@10 is 0.45 and leakage up to 12% is allowed."
    )
    bad_claims, bad_rate = verify_answer_claims(hallucinated_answer, res.ranked_chunks)
    assert len(bad_claims) == 1
    assert bad_claims[0].citation_ids_valid is False
    assert bad_claims[0].is_supported is False
    assert bad_rate == 0.0

    assert extract_citation_ids("Fact one [DOC-AI-SOP-001#P1, DOC-AI-SOP-001#P2].") == [
        "DOC-AI-SOP-001#P1",
        "DOC-AI-SOP-001#P2",
    ]
    assert len(decompose_into_atomic_claims("First sentence [DOC-1#P1]. Second sentence [DOC-1#P2].")) == 2


def test_hitl_sqlite_queue_approval_and_upstream_edit_invalidation(tmp_path: Path) -> None:
    db_file = tmp_path / "hitl_test.sqlite"
    queue = HITLReviewQueue(db_path=db_file)

    orig_chunk_digests = {"DOC-MED-005#P2": "a" * 64}
    record = queue.submit(
        HITLSubmitRequest(
            query="What does ISO-13485-CL-7.5.8 require when a clinical SOP is updated?",
            user_role=RoleEnum.ENGINEERING,
            retrieval_mode=RetrievalMode.HYBRID_RRF_RERANK,
            answer_text="Every clinical summary must bind to SHA-256 content digests [DOC-MED-005#P2].",
            cited_chunk_ids=["DOC-MED-005#P2"],
            chunk_digests=orig_chunk_digests,
            flagged_reason="Regulated clinical compliance boundary check.",
        )
    )
    assert record.status == HITLReviewStatus.PENDING_REVIEW

    # Mismatched expected digest must fail closed
    with pytest.raises(ValueError, match="Evidence digest mismatch"):
        queue.record_decision(
            HITLDecisionRequest(
                review_id=record.review_id,
                decision="approve",
                expected_evidence_digest="f" * 64,
                reviewer_id="clinical-qa@enterprise.local",
                reviewer_notes="Attempt with wrong digest",
            )
        )

    # Valid approval bound to exact evidence_digest succeeds
    approved = queue.record_decision(
        HITLDecisionRequest(
            review_id=record.review_id,
            decision="approve",
            expected_evidence_digest=record.evidence_digest,
            reviewer_id="clinical-qa@enterprise.local",
            reviewer_notes="Verified against ISO 13485 SOP v1.8",
        ),
        current_corpus_chunk_digests=orig_chunk_digests,
    )
    assert approved.status == HITLReviewStatus.APPROVED

    # Verification with unchanged chunk digest remains APPROVED
    verify_ok = queue.verify_digest(record.review_id, current_chunk_digests=orig_chunk_digests)
    assert verify_ok.digest_matches is True
    assert verify_ok.current_status == HITLReviewStatus.APPROVED

    # Upstream document edit alters DOC-MED-005#P2 SHA-256 -> automatically invalidates approval!
    edited_chunk_digests = {"DOC-MED-005#P2": "b" * 64}
    verify_invalid = queue.verify_digest(record.review_id, current_chunk_digests=edited_chunk_digests)
    assert verify_invalid.digest_matches is False
    assert verify_invalid.previous_status == HITLReviewStatus.APPROVED
    assert verify_invalid.current_status == HITLReviewStatus.INVALIDATED_BY_UPSTREAM_EDIT
    assert verify_invalid.modified_chunk_ids == ["DOC-MED-005#P2"]
    assert compute_evidence_digest("q", orig_chunk_digests, "a") != compute_evidence_digest(
        "q", edited_chunk_digests, "a"
    )


def test_fastapi_endpoints_and_static_workbench_ui() -> None:
    rt = WorkbenchRuntime(hitl_db_path=":memory:")
    client = TestClient(create_app(rt))

    # 1. Health & Corpus endpoints
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["governance"]["pre_retrieval_rbac_enforced"] is True

    corpus_resp = client.get("/api/corpus")
    assert corpus_resp.status_code == 200
    assert len(corpus_resp.json()) == 12

    report_resp = client.get("/api/eval-report")
    assert report_resp.status_code == 200
    assert report_resp.json()["total_eval_cases"] == 50

    # 2. Query endpoint: stale source triggers HITL flag and auto-enqueue
    q_resp = client.post(
        "/api/query",
        json={
            "query": "Under guideline OPS-GPU-2024-09, can engineers rent g5.12xlarge spot GPU nodes in us-east-1?",
            "user_role": "engineering",
            "retrieval_mode": "hybrid_rrf_rerank",
            "top_k": 5,
            "as_of_date": "2026-10-27",
            "enqueue_hitl_if_flagged": True,
        },
    )
    assert q_resp.status_code == 200
    q_body = q_resp.json()
    assert q_body["requires_hitl_review"] is True
    assert q_body["hitl_queue_id"] is not None
    assert q_body["retrieved_chunks"][0]["is_stale"] is True

    # 3. HITL queue, decision, and digest invalidation endpoints
    review_id = q_body["hitl_queue_id"]
    ev_digest = q_body["evidence_digest"]

    dec_resp = client.post(
        "/api/hitl/decision",
        json={
            "review_id": review_id,
            "decision": "approve",
            "expected_evidence_digest": ev_digest,
            "reviewer_id": "sec-lead@enterprise.local",
            "reviewer_notes": "Temporary exception reviewed",
        },
    )
    assert dec_resp.status_code == 200
    assert dec_resp.json()["status"] == "approved"

    ver_resp = client.post(
        "/api/hitl/verify-digest",
        json={
            "review_id": review_id,
            "current_chunk_digests": {"DOC-STALE-010#P1": "0" * 64},
        },
    )
    assert ver_resp.status_code == 200
    assert ver_resp.json()["current_status"] == "invalidated_by_upstream_edit"

    # 4. Root HTML workbench
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "Enterprise Hybrid RAG" in root_resp.text
