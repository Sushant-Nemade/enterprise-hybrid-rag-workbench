"""Information Retrieval (Recall@5, MRR@10, NDCG@10) and Governance Evaluation Metrics."""

from __future__ import annotations

import math
import time

from hybrid_rag.claim_verifier import synthesize_grounded_answer, verify_answer_claims
from hybrid_rag.governance import is_role_authorized
from hybrid_rag.hitl import compute_evidence_digest
from hybrid_rag.retriever import HybridRetriever
from hybrid_rag.schemas import (
    RETRIEVAL_MODE_LABELS,
    CaseEvaluationDetail,
    EvalCase,
    EvalCaseCategory,
    ModeEvaluationMetrics,
    RetrievalMode,
    ScoredChunk,
)


def compute_recall_at_k(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str],
    k: int = 5,
) -> float:
    """Compute Recall@k over expected ground-truth chunk IDs."""
    if not expected_chunk_ids:
        return 1.0
    top_k_set = set(retrieved_chunk_ids[:k])
    expected_set = set(expected_chunk_ids)
    hits = len(top_k_set & expected_set)
    return round(hits / len(expected_set), 6)


def compute_mrr_at_k(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str],
    k: int = 10,
) -> float:
    """Compute Mean Reciprocal Rank (MRR@k) for the first relevant chunk in top-k."""
    if not expected_chunk_ids:
        return 1.0
    expected_set = set(expected_chunk_ids)
    for rank_idx, cid in enumerate(retrieved_chunk_ids[:k], start=1):
        if cid in expected_set:
            return round(1.0 / rank_idx, 6)
    return 0.0


def compute_ndcg_at_k(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str],
    k: int = 10,
) -> float:
    """Compute binary Normalized Discounted Cumulative Gain (NDCG@k)."""
    if not expected_chunk_ids:
        return 1.0
    expected_set = set(expected_chunk_ids)
    dcg = 0.0
    for idx, cid in enumerate(retrieved_chunk_ids[:k], start=1):
        if cid in expected_set:
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0
    return round(dcg / idcg, 6)


def check_case_rbac_leakage(
    retrieved_chunks: list[ScoredChunk],
    user_role: str,
    forbidden_doc_ids: list[str],
) -> bool:
    """Return True if any unauthorized or explicitly forbidden document leaked into retrieved_chunks."""
    forbidden_set = set(forbidden_doc_ids)
    for chunk in retrieved_chunks:
        if chunk.doc_id in forbidden_set:
            return True
        if not is_role_authorized(user_role, chunk.allowed_roles):
            return True
    return False


def evaluate_mode_on_benchmark(
    retriever: HybridRetriever,
    cases: list[EvalCase],
    mode: RetrievalMode,
) -> tuple[ModeEvaluationMetrics, list[CaseEvaluationDetail]]:
    """Run all 50 benchmark cases for a specific RetrievalMode and compute aggregated metrics."""
    case_details: list[CaseEvaluationDetail] = []
    answerable_recalls: list[float] = []
    answerable_mrrs: list[float] = []
    answerable_ndcgs: list[float] = []
    answerable_claim_supports: list[float] = []
    category_recalls: dict[str, list[float]] = {
        EvalCaseCategory.LEXICAL_FACTUAL.value: [],
        EvalCaseCategory.SEMANTIC_MULTICHUNK.value: [],
        EvalCaseCategory.STALE_SOURCE.value: [],
    }

    rbac_leak_flags: list[float] = []
    abstention_outcomes: list[float] = []
    stale_detection_outcomes: list[float] = []
    latencies_ms: list[float] = []

    for case in cases:
        t0 = time.perf_counter()
        res = retriever.retrieve(
            query=case.query,
            user_role=case.user_role,
            mode=mode,
            top_k=10,
            as_of_date=case.as_of_date,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        top10_chunks = res.ranked_chunks
        top5_chunks = top10_chunks[:5]
        retrieved_chunk_ids_10 = [c.chunk_id for c in top10_chunks]
        retrieved_chunk_ids_5 = [c.chunk_id for c in top5_chunks]
        retrieved_doc_ids_5 = list(dict.fromkeys(c.doc_id for c in top5_chunks))

        leaked = check_case_rbac_leakage(
            top10_chunks,
            case.user_role.value,
            case.forbidden_doc_ids,
        )
        rbac_leak_flags.append(1.0 if leaked else 0.0)

        stale_triggered = any(c.is_stale for c in top5_chunks)
        if case.expect_stale_flag:
            stale_detection_outcomes.append(1.0 if stale_triggered else 0.0)

        if case.expect_abstention:
            abstain_ok = res.should_abstain
            abstention_outcomes.append(1.0 if abstain_ok else 0.0)
            recall_5 = 1.0 if abstain_ok else 0.0
            mrr_10 = 1.0 if abstain_ok else 0.0
            ndcg_10 = 1.0 if abstain_ok else 0.0
            claim_support = 1.0 if abstain_ok else 0.0
            answer_text = synthesize_grounded_answer(
                case.query,
                top5_chunks,
                should_abstain=res.should_abstain,
                abstention_reason=res.abstention_reason,
            )
        else:
            recall_5 = compute_recall_at_k(retrieved_chunk_ids_5, case.expected_chunk_ids, k=5)
            mrr_10 = compute_mrr_at_k(retrieved_chunk_ids_10, case.expected_chunk_ids, k=10)
            ndcg_10 = compute_ndcg_at_k(retrieved_chunk_ids_10, case.expected_chunk_ids, k=10)

            answer_text = synthesize_grounded_answer(
                case.query,
                top5_chunks,
                should_abstain=res.should_abstain,
                abstention_reason=res.abstention_reason,
            )
            _, claim_support = verify_answer_claims(
                answer_text,
                top5_chunks,
                models=retriever.models,
                expected_key_terms=case.expected_key_terms,
            )

            answerable_recalls.append(recall_5)
            answerable_mrrs.append(mrr_10)
            answerable_ndcgs.append(ndcg_10)
            answerable_claim_supports.append(claim_support)
            if case.category.value in category_recalls:
                category_recalls[case.category.value].append(recall_5)
            abstain_ok = not res.should_abstain

        chunk_digests = {c.chunk_id: c.sha256_digest for c in top5_chunks}
        ev_digest = compute_evidence_digest(case.query, chunk_digests, answer_text)

        case_details.append(
            CaseEvaluationDetail(
                case_id=case.case_id,
                category=case.category,
                query=case.query,
                user_role=case.user_role.value,
                retrieved_chunk_ids_top5=retrieved_chunk_ids_5,
                retrieved_doc_ids_top5=retrieved_doc_ids_5,
                recall_at_5=recall_5,
                mrr_at_10=mrr_10,
                ndcg_at_10=ndcg_10,
                passage_claim_support_rate=claim_support,
                rbac_leaked=leaked,
                abstained=res.should_abstain,
                abstention_correct=abstain_ok,
                stale_flag_triggered=stale_triggered,
                evidence_digest=ev_digest,
            )
        )

    mean_recall_5 = round(sum(answerable_recalls) / max(1, len(answerable_recalls)), 4)
    mean_mrr_10 = round(sum(answerable_mrrs) / max(1, len(answerable_mrrs)), 4)
    mean_ndcg_10 = round(sum(answerable_ndcgs) / max(1, len(answerable_ndcgs)), 4)
    mean_claim_support = round(
        sum(answerable_claim_supports) / max(1, len(answerable_claim_supports)), 4
    )
    rbac_leakage_rate = round(sum(rbac_leak_flags) / max(1, len(rbac_leak_flags)), 4)
    abstention_acc = round(sum(abstention_outcomes) / max(1, len(abstention_outcomes)), 4)
    stale_rate = round(
        sum(stale_detection_outcomes) / max(1, len(stale_detection_outcomes)), 4
    )
    mean_lat = round(sum(latencies_ms) / max(1, len(latencies_ms)), 3)

    cat_summary = {
        cat: round(sum(vals) / max(1, len(vals)), 4)
        for cat, vals in category_recalls.items()
    }

    metrics = ModeEvaluationMetrics(
        mode=mode,
        mode_label=RETRIEVAL_MODE_LABELS[mode],
        total_cases=len(cases),
        answerable_cases=len(answerable_recalls),
        Recall_at_5=mean_recall_5,
        MRR_at_10=mean_mrr_10,
        NDCG_at_10=mean_ndcg_10,
        Passage_Claim_Support_Rate=mean_claim_support,
        RBAC_Leakage_Rate=rbac_leakage_rate,
        Unanswerable_Abstention_Accuracy=abstention_acc,
        Stale_Source_Detection_Rate=stale_rate,
        mean_latency_ms=mean_lat,
        category_recall_at_5=cat_summary,
    )
    return metrics, case_details
