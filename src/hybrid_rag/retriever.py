"""Hybrid Retrieval Engine: SQLite FTS5 BM25 + 384-d Dense Vectors + RRF (k=60) + Cross-Encoder Reranker."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import numpy as np

from hybrid_rag.governance import (
    build_rbac_audit_summary,
    evaluate_chunk_freshness,
    filter_authorized_chunks,
    normalize_role,
    verify_zero_rbac_leakage,
)
from hybrid_rag.ingest import IngestedCorpus
from hybrid_rag.onnx_models import (
    CODE_ID_PATTERN,
    LocalONNXModelSuite,
    load_model_suite,
)
from hybrid_rag.schemas import (
    DocumentChunk,
    RBACAuditSummary,
    RetrievalMode,
    RoleEnum,
    ScoredChunk,
)

FTS_STOP_WORDS: frozenset[str] = frozenset(
    {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "under", "according", "does", "are", "the", "and", "for", "with", "that",
        "this", "from", "into", "over", "after", "before", "between", "within",
        "must", "can", "may", "will", "should", "would", "could", "have", "has",
        "had", "been", "being", "their", "there", "than", "then", "them", "they",
        "our", "your", "its", "any", "all", "each", "every", "both", "such",
        "more", "most", "other", "some", "only", "same", "so", "too", "very",
        # Ubiquitous enterprise boilerplate terms that appear across all SOP documents
        "enterprise", "production", "model", "models", "system", "systems",
        "policy", "policies", "document", "documents", "architecture", "workflow",
        "workflows", "review", "reviews", "approval", "approved", "security",
        "compliance", "contract", "vendor", "software", "package", "release",
        "department", "division", "executive", "european", "artificial",
        "intelligence", "pipeline", "pipelines", "cloud", "service", "rule",
        "rules", "standard", "schedule", "clause", "control", "protocol",
    }
)

FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")


def build_fts5_match_expression(query: str) -> str:
    """Build a safe SQLite FTS5 MATCH expression preserving exact code phrases and content tokens."""
    clauses: list[str] = []
    # 1. Exact hyphenated policy/control codes as phrase matches (high BM25 specificity)
    for code in CODE_ID_PATTERN.findall(query):
        parts = [p.lower() for p in FTS_TOKEN_RE.findall(code)]
        if len(parts) >= 2:
            phrase = " ".join(parts)
            clauses.append(f'"{phrase}"')

    # 2. Individual non-stopword alphanumeric tokens
    tokens = [
        tok.lower()
        for tok in FTS_TOKEN_RE.findall(query)
        if tok.lower() not in FTS_STOP_WORDS and len(tok) >= 2
    ]
    seen: set[str] = set()
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            clauses.append(f'"{tok}"')

    if not clauses:
        fallback = [t.lower() for t in FTS_TOKEN_RE.findall(query)]
        clauses = [f'"{t}"' for t in fallback[:5]]
    return " OR ".join(clauses)


@dataclass(frozen=True)
class RetrievalExecutionResult:
    """Output of a single retrieval execution with RBAC audit and abstention signal."""

    mode: RetrievalMode
    user_role: str
    as_of_date: str
    ranked_chunks: list[ScoredChunk]
    rbac_audit: RBACAuditSummary
    should_abstain: bool
    abstention_reason: str | None
    top_confidence: float


class HybridRetriever:
    """Four-mode retriever enforcing pre-retrieval RBAC across SQLite FTS5 and Dense embeddings."""

    def __init__(
        self,
        corpus: IngestedCorpus,
        models: LocalONNXModelSuite | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.corpus = corpus
        self.models = models or load_model_suite()
        self.rrf_k = rrf_k

        # Build in-memory SQLite FTS5 index with BM25 over page-aware chunks
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_fts5_index()

        # Precompute L2-normalized 384-d dense embeddings for all chunks
        # Note: Dense encoder embeds chunk body text while FTS5 indexes codes + headings + body
        chunk_texts_for_dense = [c.text for c in self.corpus.chunks]
        self._chunk_embeddings = self.models.encode_texts(chunk_texts_for_dense)
        self._chunk_index_by_id: dict[str, int] = {
            c.chunk_id: idx for idx, c in enumerate(self.corpus.chunks)
        }

    def _init_fts5_index(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    title,
                    section_heading,
                    body,
                    tokenize='unicode61'
                );
                """
            )
            for chunk in self.corpus.chunks:
                self._conn.execute(
                    """
                    INSERT INTO chunks_fts (chunk_id, doc_id, title, section_heading, body)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.title,
                        chunk.section_heading,
                        chunk.text,
                    ),
                )

    def _search_lexical_fts5(
        self,
        query: str,
        authorized_chunks: list[DocumentChunk],
    ) -> dict[str, tuple[int, float]]:
        """Execute SQLite FTS5 BM25 search strictly restricted to pre-filtered authorized_chunks."""
        if not authorized_chunks:
            return {}

        match_expr = build_fts5_match_expression(query)
        if not match_expr:
            return {}

        auth_ids = [c.chunk_id for c in authorized_chunks]
        placeholders = ",".join("?" for _ in auth_ids)

        sql = f"""
            SELECT
                chunk_id,
                body,
                (-1.0 * bm25(chunks_fts, 0.0, 0.0, 0.8, 1.1, 1.0)) AS bm25_score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
              AND chunk_id IN ({placeholders})
            ORDER BY bm25_score DESC, chunk_id ASC;
        """
        params: list[str] = [match_expr, *auth_ids]
        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()

        q_codes = {c.lower() for c in CODE_ID_PATTERN.findall(query)}
        specific_tokens = {
            tok.lower()
            for tok in FTS_TOKEN_RE.findall(query)
            if tok.lower() not in FTS_STOP_WORDS and len(tok) >= 3
        }

        filtered_rows: list[tuple[str, float]] = []
        for row in rows:
            cid = str(row["chunk_id"])
            body_text = str(row["body"]).lower()
            raw_bm25 = max(0.0, float(row["bm25_score"]))

            # Require either an exact identifier match or >= 2 specific lexical token hits in body
            has_code_hit = any(code in body_text for code in q_codes)
            body_tokens = set(FTS_TOKEN_RE.findall(body_text))
            token_hits = len(specific_tokens & body_tokens)
            if has_code_hit or token_hits >= 2:
                filtered_rows.append((cid, raw_bm25 + (2.0 if has_code_hit else 0.0)))

        filtered_rows.sort(key=lambda item: (-round(item[1], 8), item[0]))
        results: dict[str, tuple[int, float]] = {}
        for rank_idx, (cid, score) in enumerate(filtered_rows, start=1):
            results[cid] = (rank_idx, round(score, 6))
        return results

    def _search_dense_vectors(
        self,
        query: str,
        authorized_chunks: list[DocumentChunk],
    ) -> dict[str, tuple[int, float]]:
        """Compute exact cosine similarity over L2-normalized embeddings of authorized_chunks only."""
        if not authorized_chunks:
            return {}

        q_emb = self.models.encode_texts([query])[0]  # (384,)
        auth_indices = [self._chunk_index_by_id[c.chunk_id] for c in authorized_chunks]
        auth_matrix = self._chunk_embeddings[auth_indices]  # (N_auth, 384)
        cos_scores = auth_matrix @ q_emb  # (N_auth,)

        paired = [
            (authorized_chunks[i].chunk_id, float(cos_scores[i]))
            for i in range(len(authorized_chunks))
        ]
        paired.sort(key=lambda item: (-round(item[1], 8), item[0]))

        results: dict[str, tuple[int, float]] = {}
        for rank_idx, (cid, score) in enumerate(paired, start=1):
            results[cid] = (rank_idx, round(score, 6))
        return results

    def _detect_blocked_rbac_target(
        self,
        query: str,
        authorized_chunks: list[DocumentChunk],
        blocked_chunks: list[DocumentChunk],
    ) -> list[str]:
        """Check whether the user's query specifically targets a document blocked by pre-retrieval RBAC."""
        if not blocked_chunks:
            return []

        q_codes = {c.lower() for c in CODE_ID_PATTERN.findall(query)}
        blocked_hits: set[str] = set()
        for b_chunk in blocked_chunks:
            c_codes = {
                c.lower()
                for c in CODE_ID_PATTERN.findall(
                    f"{b_chunk.doc_id} {b_chunk.section_heading} {b_chunk.text}"
                )
            }
            if q_codes and (q_codes & c_codes):
                blocked_hits.add(b_chunk.doc_id)

        if blocked_hits:
            return sorted(blocked_hits)

        # Compare semantic similarity against blocked vs authorized chunks
        q_emb = self.models.encode_texts([query])[0]
        best_auth_sim = 0.0
        for a_chunk in authorized_chunks:
            a_idx = self._chunk_index_by_id[a_chunk.chunk_id]
            best_auth_sim = max(best_auth_sim, float(np.dot(self._chunk_embeddings[a_idx], q_emb)))

        for b_chunk in blocked_chunks:
            b_idx = self._chunk_index_by_id[b_chunk.chunk_id]
            sim = float(np.dot(self._chunk_embeddings[b_idx], q_emb))
            if sim >= 0.48 and sim > best_auth_sim + 0.12:
                blocked_hits.add(b_chunk.doc_id)
        return sorted(blocked_hits)

    def retrieve(
        self,
        query: str,
        user_role: str | RoleEnum = RoleEnum.ENGINEERING,
        mode: RetrievalMode = RetrievalMode.HYBRID_RRF_RERANK,
        top_k: int = 5,
        as_of_date: str = "2026-10-27",
    ) -> RetrievalExecutionResult:
        """Execute RBAC-gated retrieval in one of the 4 benchmark modes."""
        role = normalize_role(user_role)
        authorized_chunks, blocked_chunks = filter_authorized_chunks(self.corpus.chunks, role)
        rbac_audit = build_rbac_audit_summary(self.corpus.chunks, role, as_of_date=as_of_date)

        lexical_map = self._search_lexical_fts5(query, authorized_chunks)
        dense_map = self._search_dense_vectors(query, authorized_chunks)

        # Compute Reciprocal Rank Fusion (RRF, k=60) over authorized candidates
        rrf_scores: dict[str, float] = {}
        for chunk in authorized_chunks:
            cid = chunk.chunk_id
            lex_part = (1.0 / (self.rrf_k + lexical_map[cid][0])) if cid in lexical_map else 0.0
            dense_part = (1.0 / (self.rrf_k + dense_map[cid][0])) if cid in dense_map else 0.0
            rrf_scores[cid] = lex_part + dense_part

        rrf_sorted = sorted(
            authorized_chunks,
            key=lambda c: (-round(rrf_scores[c.chunk_id], 8), c.chunk_id),
        )
        rrf_rank_map: dict[str, int] = {
            c.chunk_id: idx for idx, c in enumerate(rrf_sorted, start=1)
        }

        # Determine ordering and cross-encoder reranking by mode
        reranker_map: dict[str, float] = {}
        if mode == RetrievalMode.LEXICAL_FTS5:
            # Pure SQLite FTS5 BM25 ranking
            matched_chunks = [c for c in authorized_chunks if c.chunk_id in lexical_map]
            matched_chunks.sort(
                key=lambda c: (lexical_map[c.chunk_id][0], c.chunk_id)
            )
            ordered_chunks = matched_chunks
        elif mode == RetrievalMode.DENSE_VECTOR:
            # Pure 384-d Dense Vector cosine ranking
            ordered_chunks = sorted(
                authorized_chunks,
                key=lambda c: (dense_map[c.chunk_id][0], c.chunk_id),
            )
        elif mode == RetrievalMode.HYBRID_RRF:
            # Reciprocal Rank Fusion (k=60) without Cross-Encoder reranking
            ordered_chunks = rrf_sorted
        elif mode == RetrievalMode.HYBRID_RRF_RERANK:
            # Hybrid RRF (k=60) followed by Cross-Encoder Reranking over top-12 candidates
            rerank_pool = rrf_sorted[:12]
            pair_inputs = [
                (
                    c.text,
                    c.section_heading,
                    dense_map[c.chunk_id][1],
                    lexical_map.get(c.chunk_id, (999, 0.0))[1],
                )
                for c in rerank_pool
            ]
            ce_scores = self.models.score_rerank_pairs(query, pair_inputs)
            for c, ce_val in zip(rerank_pool, ce_scores, strict=True):
                # Combine Cross-Encoder probability with calibrated RRF prior
                rrf_norm = rrf_scores[c.chunk_id] * (self.rrf_k / 2.0)
                reranker_map[c.chunk_id] = round(0.80 * ce_val + 0.20 * rrf_norm, 6)

            reranked_top = sorted(
                rerank_pool,
                key=lambda c: (-round(reranker_map[c.chunk_id], 8), rrf_rank_map[c.chunk_id], c.chunk_id),
            )
            remaining = [c for c in rrf_sorted[12:]]
            ordered_chunks = reranked_top + remaining
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        # Build ScoredChunk objects up to top_k (while also keeping top-10 available if requested)
        scored_list: list[ScoredChunk] = []
        for final_rank, chunk in enumerate(ordered_chunks[: max(top_k, 10)], start=1):
            cid = chunk.chunk_id
            freshness = evaluate_chunk_freshness(chunk, as_of_date=as_of_date)
            lex_rank, lex_score = lexical_map.get(cid, (None, 0.0))
            d_rank, d_score = dense_map.get(cid, (None, 0.0))
            r_rank = rrf_rank_map.get(cid)
            r_score = round(rrf_scores.get(cid, 0.0), 6)
            ce_score = reranker_map.get(cid)

            if mode == RetrievalMode.LEXICAL_FTS5:
                final_score = lex_score
            elif mode == RetrievalMode.DENSE_VECTOR:
                final_score = d_score
            elif mode == RetrievalMode.HYBRID_RRF:
                final_score = r_score
            else:
                final_score = ce_score if ce_score is not None else r_score

            scored_list.append(
                ScoredChunk(
                    chunk_id=cid,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                    text=chunk.text,
                    sha256_digest=chunk.sha256_digest,
                    allowed_roles=chunk.allowed_roles,
                    effective_date=chunk.effective_date,
                    max_staleness_days=chunk.max_staleness_days,
                    is_stale=freshness.is_stale,
                    days_overdue=freshness.days_overdue,
                    lexical_rank=lex_rank,
                    lexical_bm25_score=lex_score,
                    dense_rank=d_rank,
                    dense_cosine_score=d_score,
                    rrf_rank=r_rank,
                    rrf_score=r_score,
                    reranker_score=ce_score,
                    final_rank=final_rank,
                    final_score=round(float(final_score), 6),
                )
            )

        # Defense-in-depth assertion: verify 0.0 RBAC leakage
        if not verify_zero_rbac_leakage(scored_list, role):
            raise RuntimeError("CRITICAL SECURITY VIOLATION: Unauthorized chunk detected post-retrieval")

        # Evaluate whether the system must abstain (RBAC blocked target or out-of-scope query)
        blocked_target_docs = self._detect_blocked_rbac_target(
            query, authorized_chunks, blocked_chunks
        )
        if blocked_target_docs:
            return RetrievalExecutionResult(
                mode=mode,
                user_role=role,
                as_of_date=as_of_date,
                ranked_chunks=[],
                rbac_audit=rbac_audit,
                should_abstain=True,
                abstention_reason=(
                    f"Pre-retrieval RBAC blocked access to restricted policy document(s) "
                    f"{', '.join(blocked_target_docs)} for role '{role}'."
                ),
                top_confidence=0.0,
            )

        if not scored_list:
            return RetrievalExecutionResult(
                mode=mode,
                user_role=role,
                as_of_date=as_of_date,
                ranked_chunks=[],
                rbac_audit=rbac_audit,
                should_abstain=True,
                abstention_reason="No authorized enterprise governance passages matched the query.",
                top_confidence=0.0,
            )

        top_chunk = scored_list[0]
        # Compute unified semantic/cross-encoder confidence for abstention gating
        top_ce_prob = self.models.score_rerank_pairs(
            query,
            [
                (
                    top_chunk.text,
                    top_chunk.section_heading,
                    top_chunk.dense_cosine_score,
                    top_chunk.lexical_bm25_score,
                )
            ],
        )[0]

        if top_ce_prob < 0.28 and top_chunk.dense_cosine_score < 0.35:
            return RetrievalExecutionResult(
                mode=mode,
                user_role=role,
                as_of_date=as_of_date,
                ranked_chunks=scored_list[:top_k],
                rbac_audit=rbac_audit,
                should_abstain=True,
                abstention_reason=(
                    f"Insufficient passage evidence in authorized corpus "
                    f"(max entailment confidence={top_ce_prob:.3f} < 0.28 threshold)."
                ),
                top_confidence=top_ce_prob,
            )

        return RetrievalExecutionResult(
            mode=mode,
            user_role=role,
            as_of_date=as_of_date,
            ranked_chunks=scored_list[:top_k],
            rbac_audit=rbac_audit,
            should_abstain=False,
            abstention_reason=None,
            top_confidence=top_ce_prob,
        )
