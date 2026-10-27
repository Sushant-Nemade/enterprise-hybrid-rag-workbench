"""Deterministic 4-Mode Ablation Benchmark Runner (`python -m hybrid_rag.evaluate`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hybrid_rag.evaluation import evaluate_mode_on_benchmark
from hybrid_rag.ingest import load_corpus_from_directory, load_eval_cases
from hybrid_rag.onnx_models import load_model_suite
from hybrid_rag.retriever import HybridRetriever
from hybrid_rag.schemas import (
    RETRIEVAL_MODE_LABELS,
    AblationDeltaSummary,
    CaseEvaluationDetail,
    ModeEvaluationMetrics,
    RAGBenchmarkReport,
    RetrievalMode,
)

_PKG_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = _PKG_ROOT if (_PKG_ROOT / "artifacts").is_dir() else Path.cwd()
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "corpus"
DEFAULT_EVAL_PATH = REPO_ROOT / "data" / "benchmark" / "eval_cases.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "artifacts" / "rag_eval_report.json"


def run_ablation_benchmark(
    corpus_dir: Path | None = DEFAULT_CORPUS_DIR,
    eval_path: Path | None = DEFAULT_EVAL_PATH,
    as_of_date: str = "2026-10-27",
) -> RAGBenchmarkReport:
    """Execute the 50-case evaluation benchmark across all 4 retrieval modes."""
    corpus = load_corpus_from_directory(corpus_dir)
    cases, dataset_sha256 = load_eval_cases(eval_path)
    models = load_model_suite()
    retriever = HybridRetriever(corpus=corpus, models=models, rrf_k=60)

    modes_order = [
        RetrievalMode.LEXICAL_FTS5,
        RetrievalMode.DENSE_VECTOR,
        RetrievalMode.HYBRID_RRF,
        RetrievalMode.HYBRID_RRF_RERANK,
    ]

    mode_metrics: dict[str, ModeEvaluationMetrics] = {}
    per_mode_details: dict[str, list[CaseEvaluationDetail]] = {}

    for mode in modes_order:
        metrics, details = evaluate_mode_on_benchmark(retriever, cases, mode)
        # Keep latency deterministic in committed artifacts while measuring real execution
        mode_metrics[mode.value] = metrics
        per_mode_details[mode.value] = details

    lex = mode_metrics[RetrievalMode.LEXICAL_FTS5.value]
    target = mode_metrics[RetrievalMode.HYBRID_RRF_RERANK.value]

    delta = AblationDeltaSummary(
        baseline_mode=RETRIEVAL_MODE_LABELS[RetrievalMode.LEXICAL_FTS5],
        target_mode=RETRIEVAL_MODE_LABELS[RetrievalMode.HYBRID_RRF_RERANK],
        recall_at_5_delta=round(target.Recall_at_5 - lex.Recall_at_5, 4),
        mrr_at_10_delta=round(target.MRR_at_10 - lex.MRR_at_10, 4),
        ndcg_at_10_delta=round(target.NDCG_at_10 - lex.NDCG_at_10, 4),
        claim_support_delta=round(
            target.Passage_Claim_Support_Rate - lex.Passage_Claim_Support_Rate, 4
        ),
        rbac_leakage_rate_both=max(lex.RBAC_Leakage_Rate, target.RBAC_Leakage_Rate),
    )

    return RAGBenchmarkReport(
        schema_version="1.0.0",
        benchmark_name="enterprise-hybrid-rag-workbench-50case-ablation",
        generated_at_utc="2026-10-27T00:00:00Z",
        as_of_date=as_of_date,
        dataset_sha256=dataset_sha256,
        corpus_sha256=corpus.corpus_sha256,
        total_documents=len(corpus.documents),
        total_chunks=len(corpus.chunks),
        total_eval_cases=len(cases),
        rrf_k=60,
        models=models.artifacts,
        modes=mode_metrics,
        ablation_delta=delta,
        per_mode_case_results=per_mode_details,
    )


def format_ablation_markdown_table(report: RAGBenchmarkReport) -> str:
    """Format a Markdown summary table of the 4-mode ablation study."""
    lines = [
        "| Retrieval Mode | Recall@5 | MRR@10 | NDCG@10 | Passage Claim Support | RBAC Leakage | Abstention Acc |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for mode_key in (
        RetrievalMode.LEXICAL_FTS5.value,
        RetrievalMode.DENSE_VECTOR.value,
        RetrievalMode.HYBRID_RRF.value,
        RetrievalMode.HYBRID_RRF_RERANK.value,
    ):
        m = report.modes[mode_key]
        lines.append(
            f"| **{m.mode_label}** | `{m.Recall_at_5:.4f}` | `{m.MRR_at_10:.4f}` | "
            f"`{m.NDCG_at_10:.4f}` | `{m.Passage_Claim_Support_Rate:.4f}` | "
            f"`{m.RBAC_Leakage_Rate:.4f}` | `{m.Unanswerable_Abstention_Accuracy:.4f}` |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the 50-case 4-mode Enterprise Hybrid RAG Ablation Benchmark."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Output path for rag_eval_report.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that freshly computed ablation metrics and SHA-256 hashes match existing report",
    )
    args = parser.parse_args(argv)

    report = run_ablation_benchmark()

    if args.check:
        if not args.output.is_file():
            print(f"ERROR: Report file not found at {args.output}", file=sys.stderr)
            return 1
        existing = RAGBenchmarkReport.model_validate_json(
            args.output.read_text(encoding="utf-8")
        )
        assert existing.dataset_sha256 == report.dataset_sha256, "dataset_sha256 mismatch"
        assert existing.corpus_sha256 == report.corpus_sha256, "corpus_sha256 mismatch"
        for mode_key, actual_m in report.modes.items():
            exp_m = existing.modes[mode_key]
            assert actual_m.Recall_at_5 == exp_m.Recall_at_5, f"{mode_key} Recall@5 mismatch"
            assert actual_m.MRR_at_10 == exp_m.MRR_at_10, f"{mode_key} MRR@10 mismatch"
            assert actual_m.NDCG_at_10 == exp_m.NDCG_at_10, f"{mode_key} NDCG@10 mismatch"
            assert actual_m.RBAC_Leakage_Rate == 0.0, f"{mode_key} RBAC leakage non-zero!"
        print("OK: Deterministic ablation benchmark verified against committed artifact.")
        print(format_ablation_markdown_table(report))
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote benchmark report to {args.output}")
    print(format_ablation_markdown_table(report))
    print(
        f"\nAblation Delta ({report.ablation_delta.baseline_mode} -> {report.ablation_delta.target_mode}): "
        f"Recall@5 {report.ablation_delta.recall_at_5_delta:+.4f}, "
        f"MRR@10 {report.ablation_delta.mrr_at_10_delta:+.4f}, "
        f"NDCG@10 {report.ablation_delta.ndcg_at_10_delta:+.4f}, "
        f"Claim Support {report.ablation_delta.claim_support_delta:+.4f}, "
        f"RBAC Leakage {report.ablation_delta.rbac_leakage_rate_both:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
