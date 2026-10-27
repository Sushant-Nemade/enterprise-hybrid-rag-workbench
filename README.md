# Enterprise Hybrid RAG, Pre-Retrieval RBAC & Faithfulness Workbench

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)

An evidence-led **Python / FastAPI / ONNX Runtime** reference architecture that upgrades lexical-only PDF QA into a full **4-Mode Hybrid Retrieval (SQLite FTS5 BM25 + 384-d Dense Vectors + Reciprocal Rank Fusion $k=60$ + Cross-Encoder Reranking)**, **Pre-Retrieval Role-Based Access Control (RBAC) & Freshness Governance**, **Passage-Level NLI Claim Support Evaluator**, and **Cryptographic SHA-256 Digest-Bound Human-in-the-Loop (HITL)** workbench.

## Live Public Workbench & Reference Links
- **Interactive Public Workbench (GitHub Pages)**: [https://sushant-nemade.github.io/enterprise-hybrid-rag-workbench/](https://sushant-nemade.github.io/enterprise-hybrid-rag-workbench/)
  - *Explore the 50-case 4-mode ablation table, switch RBAC roles (`engineering`, `procurement`, `legal_compliance`, `executive`, `public`), inspect multi-stage rank shifts, view atomic NLI claim scores, and simulate upstream policy edits invalidating HITL approvals directly in the browser—no local PC or API keys required.*
- **Architecture Deep Dive**: [docs/ARCHITECTURE_DEEP_DIVE.md](docs/ARCHITECTURE_DEEP_DIVE.md)
- **Model Card & SHA-256 Provenance**: [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- **Release Status & Operational Boundaries**: [docs/RELEASE_STATUS.md](docs/RELEASE_STATUS.md)
- **Machine-Readable Benchmark Artifact**: [artifacts/rag_eval_report.json](artifacts/rag_eval_report.json)

---

## 1. Verified 50-Case Ablation Benchmark (`artifacts/rag_eval_report.json`)

All 50 labelled benchmark test cases in [data/benchmark/eval_cases.json](data/benchmark/eval_cases.json) (`dataset_sha256: 857f52e52946474338ba898e94f8dd7c1385f5beacbf3b6d8f24fe3ab993096d`) are evaluated deterministically against the 12 synthetic enterprise governance documents (`30` page-aware chunks, `corpus_sha256: b225f443cd5f10e9a5c54cde07d83feee20a5a1b7d13b80d5897070f449f73fa`) across 4 retrieval modes:

| Retrieval Mode | Recall@5 | MRR@10 | NDCG@10 | Passage Claim Support | RBAC Leakage | Abstention Acc | Lexical Slice R@5 | Semantic Slice R@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. FTS5 Lexical Only** | `0.6471` | `0.6324` | `0.6362` | `0.6176` | `0.0000` | `1.0000` | `1.0000` | `0.1429` |
| **2. Dense Only (384-d ONNX)** | `0.9706` | `0.9363` | `0.9450` | `0.9074` | `0.0000` | `1.0000` | `1.0000` | `1.0000` |
| **3. Hybrid RRF ($k=60$)** | `1.0000` | `0.8873` | `0.9154` | `0.8397` | `0.0000` | `1.0000` | `1.0000` | `1.0000` |
| **4. Hybrid RRF + Reranker** | **`1.0000`** | **`0.9853`** | **`0.9891`** | **`0.9868`** | **`0.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** |

- **Measurable Delta (`FTS5 Lexical Only` $\to$ `Hybrid RRF + Reranker`)**:
  - **Recall@5**: `+0.3529` (`0.6471` $\to$ `1.0000`)
  - **MRR@10**: `+0.3529` (`0.6324` $\to$ `0.9853`, and `+0.0980` over un-reranked `Hybrid RRF`)
  - **NDCG@10**: `+0.3529` (`0.6362` $\to$ `0.9891`)
  - **Passage Claim Support Rate**: `+0.3692` (`0.6176` $\to$ `0.9868`)
  - **Pre-Retrieval RBAC Leakage Rate**: `0.0000` across all 4 modes (zero unauthorized passages retrieved).

---

## 2. Core Engineering Modules

1. **Hybrid Retrieval + Cross-Encoder Reranking ([src/hybrid_rag/retriever.py](src/hybrid_rag/retriever.py))**:
   - **Lexical Search**: `SQLite FTS5` virtual table with BM25 scoring over page-aware document chunks (`#P1`, `#P2`, `#P3`), capturing exact regulatory/control identifiers (`GATE-AI-104`, `SPEC-RET-208`, `CL-ZDR-09`, `ISO-13485-CL-7.5.8`).
   - **Semantic Vector Search**: 384-dimensional L2-normalized dense embeddings via local ONNX Runtime (`bi_encoder_minilm_l6_v2_384d.onnx`, `CPUExecutionProvider`) or optional `sentence-transformers` (`all-MiniLM-L6-v2`) with exact cosine similarity.
   - **Reciprocal Rank Fusion & Reranking**: Combines lexical and dense ranks using **RRF ($k=60$)**, then rescores the top-12 fused candidates with a local Cross-Encoder reranker (`cross_encoder_msmarco_minilm_l6_v2.onnx`).
2. **Pre-Retrieval RBAC & Freshness Governance ([src/hybrid_rag/governance.py](src/hybrid_rag/governance.py))**:
   - Every document and chunk carries validated Pydantic v2 metadata: `doc_id`, `sha256_digest`, `allowed_roles: list[str]`, `effective_date: str`, and `max_staleness_days: int`.
   - Enforces **pre-retrieval RBAC filtering** prior to FTS5 and vector scoring so unauthorized roles never retrieve or suffer top-$k$ displacement from restricted policies (`DOC-HR-008`, `DOC-LEGAL-009`, `DOC-PROC-004`).
   - Evaluates document age against `effective_date + max_staleness_days` and flags stale guidelines (`DOC-STALE-010`, `DOC-STALE-011`) prior to answer synthesis.
3. **Passage-Level Claim Support & NLI Faithfulness Evaluator ([src/hybrid_rag/claim_verifier.py](src/hybrid_rag/claim_verifier.py), [src/hybrid_rag/evaluation.py](src/hybrid_rag/evaluation.py))**:
   - Decomposes synthesized answers into atomic sentences/claims, verifies exact `[DOC-ID#P#]` citations against authorized retrieved chunks, and scores premise-hypothesis entailment using a local ONNX NLI Cross-Encoder (`nli_entailment_cross_encoder.onnx`).
4. **Digest-Bound Human-in-the-Loop (HITL) SQLite Queue ([src/hybrid_rag/hitl.py](src/hybrid_rag/hitl.py))**:
   - Persists low-confidence, stale-source, or policy-sensitive responses in a transactional SQLite review queue.
   - Binds human approvals to $\text{SHA-256}(\text{normalized\_query} \parallel \text{sorted\_chunk\_sha256s} \parallel \text{answer\_sha256})$ so any upstream document edit automatically transitions prior approvals to `invalidated_by_upstream_edit`.

---

## 3. Repository Structure

```text
├── .github/workflows/
│   ├── ci.yml                      # Linux (3.12, 3.13), Windows (3.13), ruff, mypy, pytest, evaluate --check, pip-audit, Docker
│   └── pages.yml                   # Automated GitHub Pages deployment gated on successful CI
├── data/
│   ├── corpus/                     # 12 synthetic enterprise SOP, procurement, clinical, security, and GDPR documents
│   └── benchmark/eval_cases.json   # 50 labelled test cases (lexical, semantic, RBAC-restricted, stale-source, abstention)
├── src/hybrid_rag/
│   ├── schemas.py                  # Strict Pydantic v2 data contracts
│   ├── corpus_defaults.py          # Canonical 12-doc corpus and 50-case benchmark generator
│   ├── ingest.py                   # Page-aware chunker and deterministic SHA-256 provenance hashing
│   ├── governance.py               # Pre-retrieval RBAC filter and document freshness gate
│   ├── onnx_models.py              # SHA-256 verified ONNX Runtime local inference (Bi-Encoder, Reranker, NLI)
│   ├── retriever.py                # 4-mode retriever: FTS5 BM25, Dense 384-d, Hybrid RRF (k=60), Cross-Encoder Reranker
│   ├── claim_verifier.py           # Atomic claim decomposition, citation validator, and NLI entailment scorer
│   ├── evaluation.py               # Recall@5, MRR@10, NDCG@10, Claim Support, RBAC Leakage, and Abstention metrics
│   ├── hitl.py                     # Transactional SQLite review queue with SHA-256 digest invalidation
│   ├── evaluate.py                 # Deterministic 4-mode ablation CLI runner
│   ├── api.py                      # FastAPI service with OpenAPI docs and mounted static workbench
│   ├── models/                     # Bundled ONNX Runtime models and SHA-256 manifest
│   └── site/                       # Interactive Web Workbench (served at / and deployed to GitHub Pages)
├── artifacts/
│   └── rag_eval_report.json        # Machine-readable benchmark output with dataset/model SHA-256 hashes and ablation deltas
├── docs/
│   ├── ARCHITECTURE_DEEP_DIVE.md
│   ├── MODEL_CARD.md
│   └── RELEASE_STATUS.md
├── scripts/
│   └── package_site.py             # Bundles verified evaluation artifacts into src/hybrid_rag/site/data/
├── tests/                          # Pytest suite covering governance, retrieval ablation, NLI claims, HITL, and FastAPI
├── Dockerfile                      # Non-root Python 3.13 container image
└── pyproject.toml
```

---

## 4. Local Quickstart & Reproducible Verification

```bash
# 1. Install locked dependencies with uv
uv sync --frozen

# 2. Run the 50-case 4-mode ablation benchmark (or --check to verify against committed artifact)
uv run --frozen python -m hybrid_rag.evaluate --check

# 3. Run ruff, strict mypy, and the pytest suite
uv run --frozen ruff check src tests scripts
uv run --frozen mypy src/hybrid_rag
uv run --frozen pytest -v

# 4. Launch the FastAPI service + Interactive Web Workbench at http://127.0.0.1:8000/
uv run --frozen uvicorn hybrid_rag.api:app --host 127.0.0.1 --port 8000
```

Once running, open:
- **Interactive Workbench UI**: `http://127.0.0.1:8000/`
- **Interactive OpenAPI / Swagger Docs**: `http://127.0.0.1:8000/docs`
