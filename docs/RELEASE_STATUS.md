# Release Status & Verification Evidence

## Release Identification
- **Project**: `enterprise-hybrid-rag-workbench`
- **Version**: `0.1.0` (`v0.1.0-reference`)
- **Date**: `2026-10-27`
- **License**: MIT

---

## 1. Verified Capabilities (Local CPU & CI Executed)
The following capabilities are implemented, tested in `tests/`, and verified in GitHub Actions CI across Linux (Python 3.12, 3.13), Windows (Python 3.13), and non-root Docker:

| Capability | Implementation Owner | Verification Evidence |
| :--- | :--- | :--- |
| **Page-Aware Ingestion & SHA-256 Provenance** | `src/hybrid_rag/ingest.py` | 12 documents $\to$ 30 page chunks; deterministic `corpus_sha256` (`b225f443...`) and `dataset_sha256` (`857f52e5...`) verified in `tests/test_governance_and_ingest.py`. |
| **Pre-Retrieval RBAC & Freshness Gate** | `src/hybrid_rag/governance.py` | Role predicates (`public`, `engineering`, `procurement`, `legal_compliance`, `hr_restricted`, `executive`) filter chunks before FTS5/Dense indexing (`RBAC_Leakage_Rate = 0.0000`); stale policies (`DOC-STALE-010`, `DOC-STALE-011`) detected at `1.0000` rate. |
| **4-Mode Hybrid Retrieval & RRF ($k=60$)** | `src/hybrid_rag/retriever.py` | SQLite FTS5 BM25, 384-d ONNX Bi-Encoder (`96ac9a24...`), Reciprocal Rank Fusion ($k=60$), and ONNX Cross-Encoder Reranker (`1f1c4ec7...`) verified across 50 benchmark cases. |
| **Passage-Level NLI Claim Verification** | `src/hybrid_rag/claim_verifier.py` | Atomic sentence splitter, exact `[DOC-ID#P#]` citation verification, and ONNX NLI Cross-Encoder (`b6bd0693...`) verified in `tests/test_claims_hitl_and_api.py`. |
| **Digest-Bound SQLite HITL Review Queue** | `src/hybrid_rag/hitl.py` | Transactional SQLite state machine binding approvals to `SHA256(normalized_query \|\| sorted_chunk_sha256s \|\| answer_sha256)` and transitioning to `invalidated_by_upstream_edit` on 1-byte chunk hash changes. |
| **FastAPI Service & Interactive Static Workbench** | `src/hybrid_rag/api.py`, `src/hybrid_rag/site/index.html` | Typed Pydantic v2 endpoints (`/api/health`, `/api/corpus`, `/api/eval-report`, `/api/query`, `/api/hitl/*`) and static GitHub Pages workbench verified. |

---

## 2. Unconfigured / Optional External Connectors (Not Claimed as Active Services)
In accordance with our strict evidence-led engineering policy:
- **External Paid LLM APIs (OpenAI / Azure OpenAI / Anthropic / Groq)**: **Not activated**. Answer synthesis runs deterministically over retrieved authorized passages so the benchmark and public demo require zero API keys, incur zero token cost, and never transmit data externally.
- **External Vector Databases (Pinecone / Qdrant / Milvus / pgvector)**: **Not required**. Exact L2-normalized cosine similarity is computed in-memory via NumPy over ONNX Runtime embeddings alongside SQLite FTS5.
- **Production Multi-Tenant Identity (OIDC / Entra ID / OAuth2)**: **Not claimed**. The API demonstrates deterministic pre-retrieval RBAC role enforcement via request schema (`RoleEnum`); deploying as a multi-user enterprise service requires binding `user_role` to verified JWT/OIDC claims behind TLS.
