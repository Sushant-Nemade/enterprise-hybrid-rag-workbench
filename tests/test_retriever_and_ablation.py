"""Tests for 4-Mode Hybrid Retriever, RRF Fusion, ONNX Checksums, and 50-Case Ablation Benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybrid_rag.evaluate import main as evaluate_cli_main
from hybrid_rag.evaluate import run_ablation_benchmark
from hybrid_rag.ingest import load_corpus_from_directory
from hybrid_rag.onnx_models import build_and_save_onnx_models, load_model_suite
from hybrid_rag.retriever import HybridRetriever
from hybrid_rag.schemas import RetrievalMode, RoleEnum


def test_onnx_bi_encoder_produces_unit_l2_normalized_384d_vectors() -> None:
    suite = load_model_suite()
    vecs = suite.encode_texts(
        [
            "Shadow deployment telemetry and PSI drift ceiling of 0.08",
            "German Steuer-ID and KVNR pseudonymization under GDPR Article 28",
        ]
    )
    assert vecs.shape == (2, 384)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_onnx_checksum_verification_rejects_tampered_weights(tmp_path: Path) -> None:
    build_and_save_onnx_models(tmp_path)
    bi_file = tmp_path / "bi_encoder_minilm_l6_v2_384d.onnx"
    raw = bytearray(bi_file.read_bytes())
    raw[-1] ^= 0xFF
    bi_file.write_bytes(bytes(raw))

    with pytest.raises(ValueError, match="SHA-256 checksum mismatch"):
        load_model_suite(tmp_path)


def test_semantic_paraphrase_retrieval_progression_across_modes() -> None:
    corpus = load_corpus_from_directory()
    retriever = HybridRetriever(corpus=corpus)

    # CASE-013 semantic paraphrase query (uses "dark launch", "statistical divergence", "slowdown", "revert")
    query = (
        "During a dark launch of a candidate language model in production, "
        "what statistical divergence threshold or response time slowdown triggers an automatic revert?"
    )
    lex_res = retriever.retrieve(query, user_role=RoleEnum.ENGINEERING, mode=RetrievalMode.LEXICAL_FTS5)
    rrf_rerank_res = retriever.retrieve(
        query, user_role=RoleEnum.ENGINEERING, mode=RetrievalMode.HYBRID_RRF_RERANK
    )

    lex_ids = [c.chunk_id for c in lex_res.ranked_chunks]
    assert "DOC-AI-SOP-001#P2" not in lex_ids
    assert rrf_rerank_res.ranked_chunks[0].chunk_id == "DOC-AI-SOP-001#P2"


def test_pre_retrieval_rbac_blocks_restricted_documents_in_all_modes() -> None:
    corpus = load_corpus_from_directory()
    retriever = HybridRetriever(corpus=corpus)

    restricted_query = "What is the Band L7 Principal AI Engineer annual RSU equity ceiling under HR-COMP-901?"
    for mode in RetrievalMode:
        eng_res = retriever.retrieve(restricted_query, user_role=RoleEnum.ENGINEERING, mode=mode)
        assert eng_res.should_abstain is True
        assert all(c.doc_id != "DOC-HR-008" for c in eng_res.ranked_chunks)

    # Executive role has clearance and retrieves DOC-HR-008#P1 at Rank 1
    exec_res = retriever.retrieve(
        restricted_query,
        user_role=RoleEnum.EXECUTIVE,
        mode=RetrievalMode.HYBRID_RRF_RERANK,
    )
    assert exec_res.should_abstain is False
    assert exec_res.ranked_chunks[0].chunk_id == "DOC-HR-008#P1"


def test_50_case_ablation_benchmark_and_cli_check_mode() -> None:
    report = run_ablation_benchmark()
    assert report.total_documents == 12
    assert report.total_chunks == 30
    assert report.total_eval_cases == 50

    lex = report.modes[RetrievalMode.LEXICAL_FTS5.value]
    dense = report.modes[RetrievalMode.DENSE_VECTOR.value]
    rrf = report.modes[RetrievalMode.HYBRID_RRF.value]
    rerank = report.modes[RetrievalMode.HYBRID_RRF_RERANK.value]

    # Zero RBAC leakage across all 4 modes
    for m in (lex, dense, rrf, rerank):
        assert m.RBAC_Leakage_Rate == 0.0
        assert m.Unanswerable_Abstention_Accuracy == 1.0
        assert m.Stale_Source_Detection_Rate == 1.0

    # Hybrid RRF + Reranker outperforms FTS5 Lexical Only across all retrieval & faithfulness metrics
    assert rerank.Recall_at_5 > lex.Recall_at_5
    assert rerank.MRR_at_10 > lex.MRR_at_10
    assert rerank.NDCG_at_10 > lex.NDCG_at_10
    assert rerank.Passage_Claim_Support_Rate > lex.Passage_Claim_Support_Rate
    assert rerank.MRR_at_10 > rrf.MRR_at_10

    # Verify CLI --check passes against committed artifact
    assert evaluate_cli_main(["--check"]) == 0
