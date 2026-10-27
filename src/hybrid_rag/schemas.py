"""Strict Pydantic v2 data contracts for Enterprise Hybrid RAG, RBAC, and Evaluation."""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

UNSAFE_INPUT_PATTERN = re.compile(
    r"(?:<\s*script|javascript:|onerror\s*=|onload\s*=|[\x00-\x08\x0b\x0c\x0e-\x1f])",
    re.IGNORECASE,
)


def _reject_unsafe_text(value: str, field_name: str) -> str:
    if UNSAFE_INPUT_PATTERN.search(value):
        raise ValueError(f"{field_name} contains disallowed script/HTML injection or ASCII control characters")
    return value


class RoleEnum(StrEnum):
    """Enterprise RBAC role tiers."""

    PUBLIC = "public"
    ENGINEERING = "engineering"
    PROCUREMENT = "procurement"
    LEGAL_COMPLIANCE = "legal_compliance"
    HR_RESTRICTED = "hr_restricted"
    EXECUTIVE = "executive"


class RetrievalMode(StrEnum):
    """Four retrieval modes evaluated in the ablation benchmark."""

    LEXICAL_FTS5 = "lexical_fts5"
    DENSE_VECTOR = "dense_vector"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RRF_RERANK = "hybrid_rrf_rerank"


RETRIEVAL_MODE_LABELS: dict[RetrievalMode, str] = {
    RetrievalMode.LEXICAL_FTS5: "FTS5 Lexical Only",
    RetrievalMode.DENSE_VECTOR: "Dense Only",
    RetrievalMode.HYBRID_RRF: "Hybrid RRF",
    RetrievalMode.HYBRID_RRF_RERANK: "Hybrid RRF + Reranker",
}


class EvalCaseCategory(StrEnum):
    """Evaluation slices in the 50-case enterprise benchmark."""

    LEXICAL_FACTUAL = "lexical_factual"
    SEMANTIC_MULTICHUNK = "semantic_multichunk"
    RBAC_RESTRICTED = "rbac_restricted"
    STALE_SOURCE = "stale_source"
    UNANSWERABLE_ABSTENTION = "unanswerable_abstention"


class HITLReviewStatus(StrEnum):
    """Lifecycle states for digest-bound human-in-the-loop review items."""

    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    INVALIDATED_BY_UPSTREAM_EDIT = "invalidated_by_upstream_edit"


class DocumentMetadata(BaseModel):
    """Governance metadata attached to every enterprise document and chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(..., min_length=3, max_length=80, pattern=r"^[A-Za-z0-9_\-]+$")
    title: str = Field(..., min_length=3, max_length=200)
    sha256_digest: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    allowed_roles: list[str] = Field(..., min_length=1)
    effective_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    max_staleness_days: int = Field(..., ge=1, le=3650)
    domain: str = Field(default="enterprise_sop", min_length=2, max_length=64)
    version: str = Field(default="1.0", min_length=1, max_length=32)

    @field_validator("effective_date")
    @classmethod
    def _validate_iso_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value

    @field_validator("allowed_roles")
    @classmethod
    def _validate_roles(cls, value: list[str]) -> list[str]:
        valid = {r.value for r in RoleEnum}
        cleaned: list[str] = []
        for role in value:
            norm = role.strip().lower()
            if norm not in valid:
                raise ValueError(f"Unsupported RBAC role {role!r}; allowed: {sorted(valid)}")
            if norm not in cleaned:
                cleaned.append(norm)
        if not cleaned:
            raise ValueError("allowed_roles must contain at least one valid role")
        return cleaned


class PageSectionInput(BaseModel):
    """A page-scoped section within a raw enterprise governance document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: int = Field(..., ge=1, le=500)
    section_heading: str = Field(..., min_length=2, max_length=160)
    content: str = Field(..., min_length=20, max_length=8000)


class RawCorpusDocument(BaseModel):
    """Structured enterprise document representation stored in data/corpus/."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(..., min_length=3, max_length=80, pattern=r"^[A-Za-z0-9_\-]+$")
    title: str = Field(..., min_length=3, max_length=200)
    allowed_roles: list[str] = Field(..., min_length=1)
    effective_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    max_staleness_days: int = Field(..., ge=1, le=3650)
    domain: str = Field(default="enterprise_sop", min_length=2, max_length=64)
    version: str = Field(default="1.0", min_length=1, max_length=32)
    pages: list[PageSectionInput] = Field(..., min_length=1)

    @field_validator("effective_date")
    @classmethod
    def _validate_iso_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value


class DocumentChunk(BaseModel):
    """Page-aware chunk with strict RBAC and SHA-256 provenance metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(..., min_length=4, max_length=120)
    doc_id: str = Field(..., min_length=3, max_length=80)
    title: str = Field(..., min_length=3, max_length=200)
    page_number: int = Field(..., ge=1, le=500)
    section_heading: str = Field(..., min_length=2, max_length=160)
    text: str = Field(..., min_length=10, max_length=4000)
    sha256_digest: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    doc_sha256_digest: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    allowed_roles: list[str] = Field(..., min_length=1)
    effective_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    max_staleness_days: int = Field(..., ge=1, le=3650)
    domain: str = Field(default="enterprise_sop", min_length=2, max_length=64)


class FreshnessAssessment(BaseModel):
    """Result of evaluating a document or chunk against max_staleness_days."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    chunk_id: str
    effective_date: str
    max_staleness_days: int
    as_of_date: str
    age_days: int
    is_stale: bool
    days_overdue: int = Field(..., ge=0)
    reason: str


class RBACAuditSummary(BaseModel):
    """Audit trail proving pre-retrieval RBAC enforcement and freshness checks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requesting_role: str
    total_corpus_chunks: int = Field(..., ge=0)
    authorized_chunks: int = Field(..., ge=0)
    restricted_chunks_filtered_pre_retrieval: int = Field(..., ge=0)
    restricted_doc_ids_blocked: list[str]
    stale_chunks_flagged: int = Field(..., ge=0)
    stale_doc_ids_flagged: list[str]


class ScoredChunk(BaseModel):
    """Retrieved chunk annotated with multi-stage ranking scores and governance status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    doc_id: str
    title: str
    page_number: int
    section_heading: str
    text: str
    sha256_digest: str
    allowed_roles: list[str]
    effective_date: str
    max_staleness_days: int
    is_stale: bool
    days_overdue: int = 0
    lexical_rank: int | None = None
    lexical_bm25_score: float = 0.0
    dense_rank: int | None = None
    dense_cosine_score: float = 0.0
    rrf_rank: int | None = None
    rrf_score: float = 0.0
    reranker_score: float | None = None
    final_rank: int = Field(..., ge=1)
    final_score: float


class AtomicClaimSupport(BaseModel):
    """Passage-level NLI & citation support assessment for a single atomic claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_index: int = Field(..., ge=1)
    claim_text: str = Field(..., min_length=1)
    cited_chunk_ids: list[str]
    citation_ids_valid: bool
    best_supporting_chunk_id: str | None = None
    nli_entailment_score: float = Field(..., ge=0.0, le=1.0)
    lexical_overlap_score: float = Field(..., ge=0.0, le=1.0)
    combined_support_score: float = Field(..., ge=0.0, le=1.0)
    is_supported: bool
    verdict: str


class QueryRequest(BaseModel):
    """API request payload for hybrid RAG retrieval and grounded synthesis."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=3, max_length=600)
    user_role: RoleEnum = Field(default=RoleEnum.ENGINEERING)
    retrieval_mode: RetrievalMode = Field(default=RetrievalMode.HYBRID_RRF_RERANK)
    top_k: int = Field(default=5, ge=1, le=10)
    as_of_date: str = Field(default="2026-10-27", pattern=r"^\d{4}-\d{2}-\d{2}$")
    enqueue_hitl_if_flagged: bool = Field(default=False)

    @field_validator("query")
    @classmethod
    def _validate_safe_query(cls, value: str) -> str:
        return _reject_unsafe_text(value, "query")

    @field_validator("as_of_date")
    @classmethod
    def _validate_as_of_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value


class QueryResponse(BaseModel):
    """API response containing retrieved evidence, RBAC audit, and claim faithfulness."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    query: str
    user_role: str
    retrieval_mode: str
    retrieval_mode_label: str
    as_of_date: str
    abstained: bool
    abstention_reason: str | None = None
    requires_hitl_review: bool
    hitl_reason: str | None = None
    answer_text: str
    evidence_digest: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    retrieved_chunks: list[ScoredChunk]
    rbac_audit: RBACAuditSummary
    claims: list[AtomicClaimSupport]
    passage_claim_support_rate: float = Field(..., ge=0.0, le=1.0)
    hitl_queue_id: str | None = None


class EvalCase(BaseModel):
    """Single labelled test case in data/benchmark/eval_cases.json."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(..., pattern=r"^CASE-\d{3}$")
    category: EvalCaseCategory
    query: str = Field(..., min_length=5, max_length=500)
    user_role: RoleEnum
    as_of_date: str = Field(default="2026-10-27", pattern=r"^\d{4}-\d{2}-\d{2}$")
    expected_doc_ids: list[str]
    expected_chunk_ids: list[str]
    forbidden_doc_ids: list[str] = Field(default_factory=list)
    expect_abstention: bool = False
    expect_stale_flag: bool = False
    expect_hitl_flag: bool = False
    reference_answer: str
    expected_key_terms: list[str] = Field(default_factory=list)


class EvalDatasetFile(BaseModel):
    """Top-level wrapper for data/benchmark/eval_cases.json."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    schema_version: str
    default_as_of_date: str
    description: str
    cases: list[EvalCase] = Field(..., min_length=50, max_length=50)


class CaseEvaluationDetail(BaseModel):
    """Per-case evaluation output recorded in the benchmark report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    category: EvalCaseCategory
    query: str
    user_role: str
    retrieved_chunk_ids_top5: list[str]
    retrieved_doc_ids_top5: list[str]
    recall_at_5: float
    mrr_at_10: float
    ndcg_at_10: float
    passage_claim_support_rate: float
    rbac_leaked: bool
    abstained: bool
    abstention_correct: bool
    stale_flag_triggered: bool
    evidence_digest: str


class ModeEvaluationMetrics(BaseModel):
    """Aggregated evaluation metrics for one of the 4 retrieval modes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: RetrievalMode
    mode_label: str
    total_cases: int = Field(..., ge=1)
    answerable_cases: int = Field(..., ge=0)
    Recall_at_5: float = Field(..., ge=0.0, le=1.0)
    MRR_at_10: float = Field(..., ge=0.0, le=1.0)
    NDCG_at_10: float = Field(..., ge=0.0, le=1.0)
    Passage_Claim_Support_Rate: float = Field(..., ge=0.0, le=1.0)
    RBAC_Leakage_Rate: float = Field(..., ge=0.0, le=1.0)
    Unanswerable_Abstention_Accuracy: float = Field(..., ge=0.0, le=1.0)
    Stale_Source_Detection_Rate: float = Field(..., ge=0.0, le=1.0)
    mean_latency_ms: float = Field(..., ge=0.0)
    category_recall_at_5: dict[str, float]


class ModelArtifactInfo(BaseModel):
    """Cryptographic provenance record for a local inference model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    architecture: str
    runtime: str
    sha256_digest: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    embedding_dim: int | None = None
    parameters_summary: str


class AblationDeltaSummary(BaseModel):
    """Measurable delta between FTS5 Lexical Only and Hybrid RRF + Reranker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_mode: str
    target_mode: str
    recall_at_5_delta: float
    mrr_at_10_delta: float
    ndcg_at_10_delta: float
    claim_support_delta: float
    rbac_leakage_rate_both: float


class RAGBenchmarkReport(BaseModel):
    """Machine-readable benchmark artifact stored at artifacts/rag_eval_report.json."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0.0"
    benchmark_name: str = "enterprise-hybrid-rag-workbench-50case-ablation"
    generated_at_utc: str
    as_of_date: str
    dataset_sha256: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    total_documents: int
    total_chunks: int
    total_eval_cases: int
    rrf_k: int = 60
    models: dict[str, ModelArtifactInfo]
    modes: dict[str, ModeEvaluationMetrics]
    ablation_delta: AblationDeltaSummary
    per_mode_case_results: dict[str, list[CaseEvaluationDetail]]


class HITLReviewRecord(BaseModel):
    """Persisted SQLite record for digest-bound human-in-the-loop review."""

    model_config = ConfigDict(extra="forbid")

    review_id: str
    query: str
    user_role: str
    retrieval_mode: str
    answer_text: str
    cited_chunk_ids: list[str]
    chunk_digests: dict[str, str]
    answer_sha256: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    evidence_digest: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    status: HITLReviewStatus
    flagged_reason: str
    reviewer_id: str | None = None
    reviewer_notes: str | None = None
    created_at: str
    reviewed_at: str | None = None


class HITLSubmitRequest(BaseModel):
    """Request to enqueue a response in the HITL review queue."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=3, max_length=600)
    user_role: RoleEnum = Field(default=RoleEnum.ENGINEERING)
    retrieval_mode: RetrievalMode = Field(default=RetrievalMode.HYBRID_RRF_RERANK)
    answer_text: str = Field(..., min_length=5, max_length=4000)
    cited_chunk_ids: list[str] = Field(..., min_length=1)
    chunk_digests: dict[str, str] = Field(..., min_length=1)
    flagged_reason: str = Field(..., min_length=3, max_length=300)

    @field_validator("query", "answer_text", "flagged_reason")
    @classmethod
    def _validate_safe_submit_fields(cls, value: str) -> str:
        return _reject_unsafe_text(value, "HITLSubmitRequest field")


class HITLDecisionRequest(BaseModel):
    """Human reviewer decision bound to the expected evidence SHA-256 digest."""

    model_config = ConfigDict(extra="forbid")

    review_id: str = Field(..., min_length=4, max_length=80)
    decision: str = Field(..., pattern=r"^(approve|reject)$")
    expected_evidence_digest: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    reviewer_id: str = Field(..., min_length=2, max_length=80)
    reviewer_notes: str = Field(default="", max_length=500)

    @field_validator("reviewer_id", "reviewer_notes")
    @classmethod
    def _validate_safe_decision_fields(cls, value: str) -> str:
        return _reject_unsafe_text(value, "HITLDecisionRequest field")


class HITLVerifyDigestRequest(BaseModel):
    """Request to verify whether an approved HITL record is still valid against current chunk hashes."""

    model_config = ConfigDict(extra="forbid")

    review_id: str = Field(..., min_length=4, max_length=80)
    current_chunk_digests: dict[str, str] | None = None


class HITLVerifyDigestResponse(BaseModel):
    """Digest verification result showing whether upstream edits invalidated approval."""

    model_config = ConfigDict(extra="forbid")

    review_id: str
    previous_status: HITLReviewStatus
    current_status: HITLReviewStatus
    stored_evidence_digest: str
    recomputed_evidence_digest: str
    digest_matches: bool
    modified_chunk_ids: list[str]
    record: HITLReviewRecord
    meta: dict[str, Any] = Field(default_factory=dict)
