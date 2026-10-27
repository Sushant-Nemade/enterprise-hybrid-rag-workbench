"""Unit tests for Page-Aware Ingestion, SHA-256 Provenance, Pre-Retrieval RBAC, and Freshness Gating."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hybrid_rag.governance import (
    build_rbac_audit_summary,
    evaluate_chunk_freshness,
    filter_authorized_chunks,
    is_role_authorized,
    verify_zero_rbac_leakage,
)
from hybrid_rag.ingest import (
    compute_chunk_sha256,
    load_corpus_from_directory,
    load_eval_cases,
)
from hybrid_rag.schemas import DocumentMetadata, RoleEnum


def test_corpus_ingestion_counts_and_sha256_determinism() -> None:
    corpus_a = load_corpus_from_directory()
    corpus_b = load_corpus_from_directory()

    assert len(corpus_a.documents) == 12
    assert len(corpus_a.chunks) == 30
    assert corpus_a.corpus_sha256 == corpus_b.corpus_sha256
    assert len(corpus_a.corpus_sha256) == 64

    cases, dataset_sha = load_eval_cases()
    assert len(cases) == 50
    assert len(dataset_sha) == 64


def test_chunk_sha256_changes_on_single_character_edit() -> None:
    digest_1 = compute_chunk_sha256(
        doc_id="DOC-AI-SOP-001",
        page_number=1,
        section_heading="1. Release Gate",
        content="Minimum NDCG@10 of 0.82 is required.",
        allowed_roles=["engineering"],
        effective_date="2026-08-15",
        max_staleness_days=180,
    )
    digest_2 = compute_chunk_sha256(
        doc_id="DOC-AI-SOP-001",
        page_number=1,
        section_heading="1. Release Gate",
        content="Minimum NDCG@10 of 0.85 is required.",
        allowed_roles=["engineering"],
        effective_date="2026-08-15",
        max_staleness_days=180,
    )
    assert digest_1 != digest_2


def test_pydantic_metadata_validation_rejects_invalid_role_and_date() -> None:
    with pytest.raises(ValidationError):
        DocumentMetadata(
            doc_id="DOC-BAD-001",
            title="Invalid Role Document",
            sha256_digest="a" * 64,
            allowed_roles=["unauthorized_hacker_role"],
            effective_date="2026-08-15",
            max_staleness_days=90,
        )

    with pytest.raises(ValidationError):
        DocumentMetadata(
            doc_id="DOC-BAD-002",
            title="Invalid Date Document",
            sha256_digest="b" * 64,
            allowed_roles=["engineering"],
            effective_date="2026-13-45",
            max_staleness_days=90,
        )


def test_pre_retrieval_rbac_filtering_across_roles() -> None:
    corpus = load_corpus_from_directory()

    # Public role must never see HR, Legal, Procurement, GDPR, or Medical restricted docs
    pub_auth, pub_blocked = filter_authorized_chunks(corpus.chunks, RoleEnum.PUBLIC)
    pub_doc_ids = {c.doc_id for c in pub_auth}
    assert "DOC-HR-008" not in pub_doc_ids
    assert "DOC-LEGAL-009" not in pub_doc_ids
    assert "DOC-PROC-004" not in pub_doc_ids
    assert "DOC-GDPR-003" not in pub_doc_ids
    assert "DOC-MED-005" not in pub_doc_ids
    assert len(pub_auth) + len(pub_blocked) == 30

    # Engineering role sees GDPR and Medical, but never HR-008, LEGAL-009, or PROC-004
    eng_auth, _ = filter_authorized_chunks(corpus.chunks, RoleEnum.ENGINEERING)
    eng_doc_ids = {c.doc_id for c in eng_auth}
    assert "DOC-GDPR-003" in eng_doc_ids
    assert "DOC-MED-005" in eng_doc_ids
    assert "DOC-HR-008" not in eng_doc_ids
    assert "DOC-LEGAL-009" not in eng_doc_ids
    assert "DOC-PROC-004" not in eng_doc_ids

    # Executive role has clearance across all 12 documents (30 chunks)
    exec_auth, exec_blocked = filter_authorized_chunks(corpus.chunks, RoleEnum.EXECUTIVE)
    assert len(exec_auth) == 30
    assert len(exec_blocked) == 0


def test_freshness_gate_detects_stale_and_overdue_documents() -> None:
    corpus = load_corpus_from_directory()

    fresh_chunk = corpus.chunks_by_id["DOC-AI-SOP-001#P1"]
    fresh_eval = evaluate_chunk_freshness(fresh_chunk, as_of_date="2026-10-27")
    assert fresh_eval.is_stale is False
    assert fresh_eval.days_overdue == 0

    stale_gpu_chunk = corpus.chunks_by_id["DOC-STALE-010#P1"]
    stale_eval = evaluate_chunk_freshness(stale_gpu_chunk, as_of_date="2026-10-27")
    assert stale_eval.is_stale is True
    assert stale_eval.age_days == 650
    assert stale_eval.days_overdue == 470

    # Future date makes DOC-MED-005 stale
    med_chunk = corpus.chunks_by_id["DOC-MED-005#P1"]
    future_eval = evaluate_chunk_freshness(med_chunk, as_of_date="2027-06-15")
    assert future_eval.is_stale is True
    assert future_eval.days_overdue > 0

    audit = build_rbac_audit_summary(corpus.chunks, RoleEnum.ENGINEERING, as_of_date="2026-10-27")
    assert "DOC-STALE-010" in audit.stale_doc_ids_flagged
    assert "DOC-STALE-011" in audit.stale_doc_ids_flagged
    assert "DOC-HR-008" in audit.restricted_doc_ids_blocked
    assert is_role_authorized("engineering", ["engineering", "executive"]) is True
    assert verify_zero_rbac_leakage([], RoleEnum.PUBLIC) is True
