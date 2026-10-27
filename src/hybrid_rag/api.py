"""FastAPI Service with Typed Pydantic v2 Contracts, OpenAPI Docs, and Static Workbench Mounting."""

from __future__ import annotations

import os
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from hybrid_rag import __version__
from hybrid_rag.claim_verifier import synthesize_grounded_answer, verify_answer_claims
from hybrid_rag.evaluate import DEFAULT_REPORT_PATH, run_ablation_benchmark
from hybrid_rag.hitl import HITLReviewQueue, compute_evidence_digest
from hybrid_rag.ingest import IngestedCorpus, load_corpus_from_directory, load_eval_cases
from hybrid_rag.onnx_models import LocalONNXModelSuite, load_model_suite
from hybrid_rag.retriever import HybridRetriever
from hybrid_rag.schemas import (
    RETRIEVAL_MODE_LABELS,
    DocumentMetadata,
    HITLDecisionRequest,
    HITLReviewRecord,
    HITLReviewStatus,
    HITLSubmitRequest,
    HITLVerifyDigestRequest,
    HITLVerifyDigestResponse,
    QueryRequest,
    QueryResponse,
    RAGBenchmarkReport,
)

SITE_DIR = Path(__file__).resolve().parent / "site"


class WorkbenchRuntime:
    """Stateful container holding the ingested corpus, ONNX models, retriever, and HITL SQLite queue."""

    def __init__(
        self,
        corpus_dir: Path | None = None,
        eval_path: Path | None = None,
        hitl_db_path: str | Path = ":memory:",
    ) -> None:
        self.corpus: IngestedCorpus = load_corpus_from_directory(corpus_dir)
        self.eval_cases, self.dataset_sha256 = load_eval_cases(eval_path)
        self.models: LocalONNXModelSuite = load_model_suite()
        self.retriever = HybridRetriever(corpus=self.corpus, models=self.models, rrf_k=60)
        self.hitl_queue = HITLReviewQueue(db_path=hitl_db_path)
        self._cached_report: RAGBenchmarkReport | None = None
        self._seed_demo_hitl_record()

    def _seed_demo_hitl_record(self) -> None:
        """Seed one realistic HITL review item so the queue is inspectable immediately."""
        if self.hitl_queue.list_records():
            return
        chunk = self.corpus.chunks_by_id["DOC-STALE-010#P1"]
        self.hitl_queue.submit(
            HITLSubmitRequest(
                query="Under guideline OPS-GPU-2024-09, can engineers rent g5.12xlarge spot GPU nodes in us-east-1 up to USD 4,500/month?",
                user_role="engineering",  # type: ignore[arg-type]
                retrieval_mode="hybrid_rrf_rerank",  # type: ignore[arg-type]
                answer_text=(
                    "STALE POLICY WARNING: Document DOC-STALE-010 (effective 2025-01-15, "
                    "max_staleness_days=180) is overdue; it previously described provisioning "
                    "g5.12xlarge spot GPU instances in us-east-1 up to USD 4,500 per month [DOC-STALE-010#P1]."
                ),
                cited_chunk_ids=["DOC-STALE-010#P1"],
                chunk_digests={"DOC-STALE-010#P1": chunk.sha256_digest},
                flagged_reason="Cited policy DOC-STALE-010 exceeds max_staleness_days=180 (HITL-DIGEST-801).",
            )
        )

    def get_eval_report(self) -> RAGBenchmarkReport:
        if self._cached_report is not None:
            return self._cached_report
        if DEFAULT_REPORT_PATH.is_file():
            self._cached_report = RAGBenchmarkReport.model_validate_json(
                DEFAULT_REPORT_PATH.read_text(encoding="utf-8")
            )
            return self._cached_report
        self._cached_report = run_ablation_benchmark()
        return self._cached_report


def create_app(runtime: WorkbenchRuntime | None = None) -> FastAPI:
    """Create the FastAPI application with OpenAPI documentation and workbench routes."""
    rt = runtime or WorkbenchRuntime(
        hitl_db_path=os.environ.get("HYBRID_RAG_HITL_DB", ":memory:")
    )

    app = FastAPI(
        title="Enterprise Hybrid RAG, RBAC & Faithfulness Workbench",
        version=__version__,
        description=(
            "Evidence-led reference architecture upgrading lexical PDF QA into Hybrid Retrieval "
            "(SQLite FTS5 BM25 + 384-d Dense Vectors + Reciprocal Rank Fusion k=60 + Cross-Encoder Reranking), "
            "Pre-Retrieval RBAC & Freshness Governance, Passage-Level NLI Claim Support Evaluation, "
            "and Cryptographic Digest-Bound Human-in-the-Loop (HITL) Review."
        ),
    )
    app.state.runtime = rt

    @app.middleware("http")
    async def add_security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        return response

    @app.get("/api/health", tags=["system"])
    def get_health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "corpus_documents": len(rt.corpus.documents),
            "corpus_chunks": len(rt.corpus.chunks),
            "eval_cases": len(rt.eval_cases),
            "corpus_sha256": rt.corpus.corpus_sha256,
            "dataset_sha256": rt.dataset_sha256,
            "models": {
                k: v.model_dump(mode="json") for k, v in rt.models.artifacts.items()
            },
            "governance": {
                "pre_retrieval_rbac_enforced": True,
                "freshness_gate_enforced": True,
                "hitl_digest_binding_enforced": True,
                "external_llm_enabled": False,
            },
        }

    @app.get("/api/corpus", response_model=list[DocumentMetadata], tags=["governance"])
    def list_corpus_documents() -> list[DocumentMetadata]:
        return rt.corpus.documents

    @app.get("/api/eval-report", response_model=RAGBenchmarkReport, tags=["evaluation"])
    def get_benchmark_report() -> RAGBenchmarkReport:
        return rt.get_eval_report()

    @app.post("/api/query", response_model=QueryResponse, tags=["retrieval"])
    def execute_hybrid_query(req: QueryRequest) -> QueryResponse:
        retrieval_res = rt.retriever.retrieve(
            query=req.query,
            user_role=req.user_role,
            mode=req.retrieval_mode,
            top_k=req.top_k,
            as_of_date=req.as_of_date,
        )
        chunks = retrieval_res.ranked_chunks
        answer_text = synthesize_grounded_answer(
            query=req.query,
            retrieved_chunks=chunks,
            should_abstain=retrieval_res.should_abstain,
            abstention_reason=retrieval_res.abstention_reason,
        )
        claims, support_rate = verify_answer_claims(
            answer_text=answer_text,
            retrieved_chunks=chunks,
            models=rt.models,
        )

        chunk_digests = {c.chunk_id: c.sha256_digest for c in chunks}
        evidence_digest = compute_evidence_digest(req.query, chunk_digests, answer_text)

        has_stale = any(c.is_stale for c in chunks)
        low_support = (not retrieval_res.should_abstain) and (support_rate < 0.72)
        rbac_abstained = (
            retrieval_res.should_abstain
            and retrieval_res.abstention_reason is not None
            and "Pre-retrieval RBAC blocked" in retrieval_res.abstention_reason
        )

        requires_hitl = has_stale or low_support or rbac_abstained
        hitl_reasons: list[str] = []
        if has_stale:
            stale_ids = sorted({c.doc_id for c in chunks if c.is_stale})
            hitl_reasons.append(f"Stale source policy detected: {', '.join(stale_ids)}")
        if low_support:
            hitl_reasons.append(f"Passage claim support rate {support_rate:.3f} < 0.72 floor")
        if rbac_abstained and retrieval_res.abstention_reason:
            hitl_reasons.append(retrieval_res.abstention_reason)
        hitl_reason_str = "; ".join(hitl_reasons) if hitl_reasons else None

        hitl_queue_id: str | None = None
        if requires_hitl and req.enqueue_hitl_if_flagged and chunks:
            submitted = rt.hitl_queue.submit(
                HITLSubmitRequest(
                    query=req.query,
                    user_role=req.user_role,
                    retrieval_mode=req.retrieval_mode,
                    answer_text=answer_text,
                    cited_chunk_ids=[c.chunk_id for c in chunks],
                    chunk_digests=chunk_digests,
                    flagged_reason=hitl_reason_str or "Policy governance check",
                )
            )
            hitl_queue_id = submitted.review_id

        return QueryResponse(
            query_id=f"QRY-{uuid.uuid4().hex[:10].upper()}",
            query=req.query,
            user_role=req.user_role.value,
            retrieval_mode=req.retrieval_mode.value,
            retrieval_mode_label=RETRIEVAL_MODE_LABELS[req.retrieval_mode],
            as_of_date=req.as_of_date,
            abstained=retrieval_res.should_abstain,
            abstention_reason=retrieval_res.abstention_reason,
            requires_hitl_review=requires_hitl,
            hitl_reason=hitl_reason_str,
            answer_text=answer_text,
            evidence_digest=evidence_digest,
            retrieved_chunks=chunks,
            rbac_audit=retrieval_res.rbac_audit,
            claims=claims,
            passage_claim_support_rate=support_rate,
            hitl_queue_id=hitl_queue_id,
        )

    @app.get("/api/hitl/queue", response_model=list[HITLReviewRecord], tags=["hitl"])
    def list_hitl_queue(
        status: HITLReviewStatus | None = Query(default=None),
    ) -> list[HITLReviewRecord]:
        return rt.hitl_queue.list_records(status=status)

    @app.post("/api/hitl/submit", response_model=HITLReviewRecord, tags=["hitl"])
    def submit_hitl_review(req: HITLSubmitRequest) -> HITLReviewRecord:
        return rt.hitl_queue.submit(req)

    @app.post("/api/hitl/decision", response_model=HITLReviewRecord, tags=["hitl"])
    def record_hitl_decision(req: HITLDecisionRequest) -> HITLReviewRecord:
        live_digests = {cid: c.sha256_digest for cid, c in rt.corpus.chunks_by_id.items()}
        try:
            return rt.hitl_queue.record_decision(req, current_corpus_chunk_digests=live_digests)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/hitl/verify-digest", response_model=HITLVerifyDigestResponse, tags=["hitl"])
    def verify_hitl_digest(req: HITLVerifyDigestRequest) -> HITLVerifyDigestResponse:
        digests = req.current_chunk_digests or {
            cid: c.sha256_digest for cid, c in rt.corpus.chunks_by_id.items()
        }
        try:
            return rt.hitl_queue.verify_digest(req.review_id, current_chunk_digests=digests)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    if SITE_DIR.is_dir():
        data_sub = SITE_DIR / "data"
        if data_sub.is_dir():
            app.mount("/data", StaticFiles(directory=data_sub), name="site-data")

        @app.get("/", include_in_schema=False)
        def serve_workbench_index() -> FileResponse:
            return FileResponse(SITE_DIR / "index.html")
    else:

        @app.get("/", include_in_schema=False)
        def serve_fallback_root() -> JSONResponse:
            return JSONResponse({"service": "enterprise-hybrid-rag-workbench", "version": __version__})

    return app


app = create_app()
