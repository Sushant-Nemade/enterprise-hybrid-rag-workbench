"""Pre-Retrieval Role-Based Access Control (RBAC) and Document Freshness Governance Gates."""

from __future__ import annotations

from datetime import date

from hybrid_rag.schemas import (
    DocumentChunk,
    FreshnessAssessment,
    RBACAuditSummary,
    RoleEnum,
    ScoredChunk,
)


def normalize_role(user_role: str | RoleEnum) -> str:
    """Validate and normalize a role string against RoleEnum."""
    if isinstance(user_role, RoleEnum):
        return user_role.value
    cleaned = user_role.strip().lower()
    valid = {r.value for r in RoleEnum}
    if cleaned not in valid:
        raise ValueError(f"Invalid role {user_role!r}; must be one of {sorted(valid)}")
    return cleaned


def is_role_authorized(user_role: str | RoleEnum, allowed_roles: list[str]) -> bool:
    """Return True if and only if user_role is explicitly listed in allowed_roles."""
    role = normalize_role(user_role)
    allowed_normalized = {r.strip().lower() for r in allowed_roles}
    return role in allowed_normalized


def filter_authorized_chunks(
    chunks: list[DocumentChunk],
    user_role: str | RoleEnum,
) -> tuple[list[DocumentChunk], list[DocumentChunk]]:
    """Enforce pre-retrieval RBAC filtering over corpus chunks.

    Returns (authorized_chunks, restricted_chunks_blocked) so neither SQLite FTS5
    nor dense vector similarity ever evaluates unauthorized passages.
    """
    role = normalize_role(user_role)
    authorized: list[DocumentChunk] = []
    blocked: list[DocumentChunk] = []
    for chunk in chunks:
        if is_role_authorized(role, chunk.allowed_roles):
            authorized.append(chunk)
        else:
            blocked.append(chunk)
    return authorized, blocked


def evaluate_chunk_freshness(
    chunk: DocumentChunk,
    as_of_date: str = "2026-10-27",
) -> FreshnessAssessment:
    """Evaluate whether a chunk has exceeded its max_staleness_days as of as_of_date."""
    eval_dt = date.fromisoformat(as_of_date)
    effective_dt = date.fromisoformat(chunk.effective_date)
    age_days = (eval_dt - effective_dt).days

    if age_days > chunk.max_staleness_days:
        days_overdue = age_days - chunk.max_staleness_days
        return FreshnessAssessment(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            effective_date=chunk.effective_date,
            max_staleness_days=chunk.max_staleness_days,
            as_of_date=as_of_date,
            age_days=age_days,
            is_stale=True,
            days_overdue=days_overdue,
            reason=(
                f"Document {chunk.doc_id} (effective {chunk.effective_date}) is {age_days} days old "
                f"as of {as_of_date}, exceeding max_staleness_days={chunk.max_staleness_days} "
                f"by {days_overdue} days."
            ),
        )

    return FreshnessAssessment(
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        effective_date=chunk.effective_date,
        max_staleness_days=chunk.max_staleness_days,
        as_of_date=as_of_date,
        age_days=age_days,
        is_stale=False,
        days_overdue=0,
        reason=(
            f"Document {chunk.doc_id} is within freshness bounds ({age_days} <= {chunk.max_staleness_days} days)."
        ),
    )


def build_rbac_audit_summary(
    all_chunks: list[DocumentChunk],
    user_role: str | RoleEnum,
    as_of_date: str = "2026-10-27",
) -> RBACAuditSummary:
    """Build a complete pre-retrieval RBAC and freshness audit trail."""
    role = normalize_role(user_role)
    authorized, blocked = filter_authorized_chunks(all_chunks, role)

    blocked_doc_ids = sorted({c.doc_id for c in blocked})
    stale_chunks = [
        c for c in authorized if evaluate_chunk_freshness(c, as_of_date=as_of_date).is_stale
    ]
    stale_doc_ids = sorted({c.doc_id for c in stale_chunks})

    return RBACAuditSummary(
        requesting_role=role,
        total_corpus_chunks=len(all_chunks),
        authorized_chunks=len(authorized),
        restricted_chunks_filtered_pre_retrieval=len(blocked),
        restricted_doc_ids_blocked=blocked_doc_ids,
        stale_chunks_flagged=len(stale_chunks),
        stale_doc_ids_flagged=stale_doc_ids,
    )


def verify_zero_rbac_leakage(
    retrieved_chunks: list[ScoredChunk],
    user_role: str | RoleEnum,
) -> bool:
    """Assert that every retrieved chunk is authorized for user_role (0.0 leakage check)."""
    role = normalize_role(user_role)
    return all(is_role_authorized(role, chunk.allowed_roles) for chunk in retrieved_chunks)
