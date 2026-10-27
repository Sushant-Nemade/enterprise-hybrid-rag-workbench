"""Package static data assets into src/hybrid_rag/site/data/ for FastAPI and GitHub Pages."""

from __future__ import annotations

import json
from pathlib import Path

from hybrid_rag.api import WorkbenchRuntime
from hybrid_rag.evaluate import DEFAULT_REPORT_PATH, run_ablation_benchmark
from hybrid_rag.schemas import QueryRequest, RetrievalMode, RoleEnum

ROOT = Path(__file__).resolve().parents[1]
SITE_DATA_DIR = ROOT / "src" / "hybrid_rag" / "site" / "data"


def build_site_data() -> None:
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rt = WorkbenchRuntime()

    if DEFAULT_REPORT_PATH.is_file():
        report_text = DEFAULT_REPORT_PATH.read_text(encoding="utf-8")
    else:
        report = run_ablation_benchmark()
        report_text = json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
        DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_REPORT_PATH.write_text(report_text, encoding="utf-8")

    (SITE_DATA_DIR / "rag_eval_report.json").write_text(report_text, encoding="utf-8")

    # Precompute 6 representative interactive scenarios across all 4 modes for static GitHub Pages mode
    showcase_cases = [
        ("CASE-013", "Semantic Paraphrase: Dark Launch & PSI Drift Rollback", RoleEnum.ENGINEERING, "2026-10-27"),
        ("CASE-003", "Exact Identifier: SPEC-RET-208 BM25 & 384-d Embeddings", RoleEnum.ENGINEERING, "2026-10-27"),
        ("CASE-015", "Semantic GDPR: German Steuer-ID, KVNR & IBAN Scrubbing", RoleEnum.LEGAL_COMPLIANCE, "2026-10-27"),
        ("CASE-027", "Pre-Retrieval RBAC Block: Band L7 RSU Equity (Engineering Role)", RoleEnum.ENGINEERING, "2026-10-27"),
        ("CASE-035", "Freshness Gate: Expired 2024 Spot GPU Policy (DOC-STALE-010)", RoleEnum.ENGINEERING, "2026-10-27"),
        ("CASE-043", "Unanswerable Abstention: Erlangen Cafeteria Vegan Subsidy", RoleEnum.ENGINEERING, "2026-10-27"),
    ]
    cases_by_id = {c.case_id: c for c in rt.eval_cases}

    from fastapi.testclient import TestClient

    from hybrid_rag.api import create_app

    client = TestClient(create_app(rt))
    scenarios: list[dict[str, object]] = []
    for cid, label, default_role, as_of in showcase_cases:
        ec = cases_by_id[cid]
        mode_outputs: dict[str, object] = {}
        for mode in RetrievalMode:
            resp = client.post(
                "/api/query",
                json=QueryRequest(
                    query=ec.query,
                    user_role=default_role,
                    retrieval_mode=mode,
                    top_k=5,
                    as_of_date=as_of,
                ).model_dump(mode="json"),
            )
            mode_outputs[mode.value] = resp.json()

        # Also compute role comparison for CASE-027 (engineering blocked vs executive authorized)
        role_outputs: dict[str, object] = {}
        for r_enum in (RoleEnum.PUBLIC, RoleEnum.ENGINEERING, RoleEnum.PROCUREMENT, RoleEnum.LEGAL_COMPLIANCE, RoleEnum.EXECUTIVE):
            r_resp = client.post(
                "/api/query",
                json=QueryRequest(
                    query=ec.query,
                    user_role=r_enum,
                    retrieval_mode=RetrievalMode.HYBRID_RRF_RERANK,
                    top_k=5,
                    as_of_date=as_of,
                ).model_dump(mode="json"),
            )
            role_outputs[r_enum.value] = r_resp.json()

        scenarios.append(
            {
                "case_id": cid,
                "label": label,
                "category": ec.category.value,
                "query": ec.query,
                "default_role": default_role.value,
                "as_of_date": as_of,
                "expected_chunk_ids": ec.expected_chunk_ids,
                "modes": mode_outputs,
                "roles": role_outputs,
            }
        )

    bundle = {
        "documents": [d.model_dump(mode="json") for d in rt.corpus.documents],
        "chunks": [c.model_dump(mode="json") for c in rt.corpus.chunks],
        "scenarios": scenarios,
        "hitl_seed": [r.model_dump(mode="json") for r in rt.hitl_queue.list_records()],
    }
    (SITE_DATA_DIR / "workbench_bundle.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Packaged static workbench data into {SITE_DATA_DIR}")


if __name__ == "__main__":
    build_site_data()
