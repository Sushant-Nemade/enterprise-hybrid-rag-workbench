"""Transactional SQLite Human-in-the-Loop (HITL) Review Queue with Cryptographic Digest Binding."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from hybrid_rag.ingest import sha256_hex
from hybrid_rag.schemas import (
    HITLDecisionRequest,
    HITLReviewRecord,
    HITLReviewStatus,
    HITLSubmitRequest,
    HITLVerifyDigestResponse,
)


def normalize_query_for_digest(query: str) -> str:
    """Canonicalize query whitespace and casing for deterministic SHA-256 binding."""
    return re.sub(r"\s+", " ", query.strip().lower())


def compute_evidence_digest(
    query: str,
    chunk_digests: dict[str, str],
    answer_text: str,
) -> str:
    """Compute composite SHA-256 evidence digest binding query, sorted chunk SHA-256s, and answer text."""
    q_norm = normalize_query_for_digest(query)
    sorted_pairs = "|".join(
        f"{cid}:{chunk_digests[cid]}" for cid in sorted(chunk_digests.keys())
    )
    answer_sha = sha256_hex(answer_text.strip())
    payload = f"query={q_norm}||chunks={sorted_pairs}||answer_sha256={answer_sha}"
    return sha256_hex(payload)


class HITLReviewQueue:
    """Transactional SQLite review queue that invalidates human approvals on upstream document edits."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hitl_reviews (
                    review_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    retrieval_mode TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    cited_chunk_ids_json TEXT NOT NULL,
                    chunk_digests_json TEXT NOT NULL,
                    answer_sha256 TEXT NOT NULL,
                    evidence_digest TEXT NOT NULL,
                    status TEXT NOT NULL,
                    flagged_reason TEXT NOT NULL,
                    reviewer_id TEXT,
                    reviewer_notes TEXT,
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT
                );
                """
            )

    def _row_to_record(self, row: sqlite3.Row) -> HITLReviewRecord:
        return HITLReviewRecord(
            review_id=str(row["review_id"]),
            query=str(row["query"]),
            user_role=str(row["user_role"]),
            retrieval_mode=str(row["retrieval_mode"]),
            answer_text=str(row["answer_text"]),
            cited_chunk_ids=json.loads(str(row["cited_chunk_ids_json"])),
            chunk_digests=json.loads(str(row["chunk_digests_json"])),
            answer_sha256=str(row["answer_sha256"]),
            evidence_digest=str(row["evidence_digest"]),
            status=HITLReviewStatus(str(row["status"])),
            flagged_reason=str(row["flagged_reason"]),
            reviewer_id=str(row["reviewer_id"]) if row["reviewer_id"] is not None else None,
            reviewer_notes=str(row["reviewer_notes"]) if row["reviewer_notes"] is not None else None,
            created_at=str(row["created_at"]),
            reviewed_at=str(row["reviewed_at"]) if row["reviewed_at"] is not None else None,
        )

    def submit(self, req: HITLSubmitRequest) -> HITLReviewRecord:
        """Persist a low-confidence, stale-source, or policy-sensitive answer for human review."""
        review_id = f"HITL-{uuid.uuid4().hex[:10].upper()}"
        answer_sha = sha256_hex(req.answer_text.strip())
        ev_digest = compute_evidence_digest(req.query, req.chunk_digests, req.answer_text)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO hitl_reviews (
                    review_id, query, user_role, retrieval_mode, answer_text,
                    cited_chunk_ids_json, chunk_digests_json, answer_sha256,
                    evidence_digest, status, flagged_reason, reviewer_id,
                    reviewer_notes, created_at, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, NULL);
                """,
                (
                    review_id,
                    req.query,
                    req.user_role.value,
                    req.retrieval_mode.value,
                    req.answer_text,
                    json.dumps(req.cited_chunk_ids),
                    json.dumps(req.chunk_digests, sort_keys=True),
                    answer_sha,
                    ev_digest,
                    HITLReviewStatus.PENDING_REVIEW.value,
                    req.flagged_reason,
                    now_iso,
                ),
            )
        return self.get(review_id)

    def get(self, review_id: str) -> HITLReviewRecord:
        """Fetch a single HITL review record by ID."""
        row = self._conn.execute(
            "SELECT * FROM hitl_reviews WHERE review_id = ?;",
            (review_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"HITL review record {review_id!r} not found")
        return self._row_to_record(row)

    def list_records(self, status: HITLReviewStatus | None = None) -> list[HITLReviewRecord]:
        """List all persisted HITL review records, optionally filtered by status."""
        if status is not None:
            rows = self._conn.execute(
                "SELECT * FROM hitl_reviews WHERE status = ? ORDER BY created_at DESC, review_id ASC;",
                (status.value,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM hitl_reviews ORDER BY created_at DESC, review_id ASC;"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def record_decision(
        self,
        req: HITLDecisionRequest,
        current_corpus_chunk_digests: dict[str, str] | None = None,
    ) -> HITLReviewRecord:
        """Record a human approval or rejection strictly bound to expected_evidence_digest."""
        existing = self.get(req.review_id)

        if existing.evidence_digest != req.expected_evidence_digest:
            raise ValueError(
                f"Evidence digest mismatch for {req.review_id}: expected "
                f"{req.expected_evidence_digest}, stored {existing.evidence_digest}"
            )

        # If live corpus chunk digests are supplied, verify no upstream edit happened before approval
        if current_corpus_chunk_digests is not None:
            active_digests = {
                cid: current_corpus_chunk_digests.get(cid, "")
                for cid in existing.chunk_digests
            }
            recomputed = compute_evidence_digest(
                existing.query, active_digests, existing.answer_text
            )
            if recomputed != existing.evidence_digest:
                now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                with self._conn:
                    self._conn.execute(
                        """
                        UPDATE hitl_reviews
                           SET status = ?, reviewed_at = ?, reviewer_notes = ?
                         WHERE review_id = ?;
                        """,
                        (
                            HITLReviewStatus.INVALIDATED_BY_UPSTREAM_EDIT.value,
                            now_iso,
                            "Upstream chunk SHA-256 digest changed prior to approval.",
                            req.review_id,
                        ),
                    )
                raise ValueError(
                    f"Approval blocked for {req.review_id}: upstream document edit invalidated evidence digest."
                )

        new_status = (
            HITLReviewStatus.APPROVED
            if req.decision == "approve"
            else HITLReviewStatus.REJECTED
        )
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        with self._conn:
            self._conn.execute(
                """
                UPDATE hitl_reviews
                   SET status = ?,
                       reviewer_id = ?,
                       reviewer_notes = ?,
                       reviewed_at = ?
                 WHERE review_id = ?;
                """,
                (
                    new_status.value,
                    req.reviewer_id,
                    req.reviewer_notes,
                    now_iso,
                    req.review_id,
                ),
            )
        return self.get(req.review_id)

    def verify_digest(
        self,
        review_id: str,
        current_chunk_digests: dict[str, str],
    ) -> HITLVerifyDigestResponse:
        """Verify whether a previously approved record remains valid against current chunk SHA-256s."""
        record = self.get(review_id)
        previous_status = record.status

        modified_chunks: list[str] = []
        active_chunk_map: dict[str, str] = {}
        for cid, original_sha in record.chunk_digests.items():
            current_sha = current_chunk_digests.get(cid)
            if current_sha is None or current_sha != original_sha:
                modified_chunks.append(cid)
                active_chunk_map[cid] = current_sha or "MISSING_CHUNK"
            else:
                active_chunk_map[cid] = current_sha

        recomputed_digest = compute_evidence_digest(
            record.query,
            active_chunk_map,
            record.answer_text,
        )
        digest_matches = (recomputed_digest == record.evidence_digest) and not modified_chunks

        if not digest_matches and record.status == HITLReviewStatus.APPROVED:
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE hitl_reviews
                       SET status = ?, reviewed_at = ?
                     WHERE review_id = ?;
                    """,
                    (
                        HITLReviewStatus.INVALIDATED_BY_UPSTREAM_EDIT.value,
                        now_iso,
                        review_id,
                    ),
                )
            record = self.get(review_id)

        return HITLVerifyDigestResponse(
            review_id=review_id,
            previous_status=previous_status,
            current_status=record.status,
            stored_evidence_digest=record.evidence_digest,
            recomputed_evidence_digest=recomputed_digest,
            digest_matches=digest_matches,
            modified_chunk_ids=sorted(modified_chunks),
            record=record,
            meta={
                "policy": "HITL-DIGEST-801",
                "formula": "SHA256(normalized_query || sorted_chunk_sha256_digests || answer_sha256)",
            },
        )
