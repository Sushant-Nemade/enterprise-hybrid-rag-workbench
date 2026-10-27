# Model Card & Benchmark Provenance — Enterprise Hybrid RAG Workbench

## 1. Overview & Intended Use
`enterprise-hybrid-rag-workbench` is a deterministic, CPU-executable reference architecture for evaluating multi-stage Hybrid Retrieval, Pre-Retrieval Role-Based Access Control (RBAC), Document Freshness Governance, and Passage-Level NLI Claim Support over structured enterprise standard operating procedures (SOPs).

- **Intended Use**: Reference implementation, architectural benchmarking, retrieval ablation analysis, pre-retrieval access-control verification, and human-in-the-loop (HITL) cryptographic auditability demonstration.
- **Out-of-Scope Use**: This repository uses a synthetic 12-document enterprise governance corpus and compact local ONNX models. It is **not** a clinical medical device, legal compliance certification authority, or multi-tenant production SaaS service.

---

## 2. Cryptographic Provenance (Dataset & ONNX Model Hashes)
Every benchmark run (`python -m hybrid_rag.evaluate`) verifies the SHA-256 checksums of the dataset, corpus, and ONNX Runtime model files before execution and records them in `artifacts/rag_eval_report.json`:

| Artifact / Model | Role | Architecture & Runtime | Verified SHA-256 Digest |
| :--- | :--- | :--- | :--- |
| **`data/benchmark/eval_cases.json`** | 50-Case Benchmark Dataset | 5 slices (12 lexical, 14 semantic paraphrase, 8 RBAC-restricted, 8 stale-source, 8 unanswerable) | `857f52e52946474338ba898e94f8dd7c1385f5beacbf3b6d8f24fe3ab993096d` |
| **`data/corpus/*.json`** | 12 Enterprise Governance Docs | 30 page-aware chunks with RBAC roles & `max_staleness_days` | `b225f443cd5f10e9a5c54cde07d83feee20a5a1b7d13b80d5897070f449f73fa` |
| **`bi_encoder_minilm_l6_v2_384d.onnx`** | Dense Semantic Bi-Encoder | 122 $\to$ 384 Isometric Semi-Orthogonal Projection (`W @ W.T = I`) with L2 Unit Normalization (`onnxruntime` `CPUExecutionProvider`) | `96ac9a24934324e148b40baa27ca6a85544319bc8ed685c9e3b500687211d397` |
| **`cross_encoder_msmarco_minilm_l6_v2.onnx`** | Cross-Encoder Reranker | 12 $\to$ 32 $\to$ 16 $\to$ 2 Joint Query-Passage MLP Classifier (`onnxruntime` `CPUExecutionProvider`) | `1f1c4ec72672cf557a17d101f3981efd670fb6051d5d286d7238463a519efe04` |
| **`nli_entailment_cross_encoder.onnx`** | Passage-Level NLI Claim Verifier | 10 $\to$ 24 $\to$ 12 $\to$ 2 Premise-Hypothesis Entailment Classifier (`onnxruntime` `CPUExecutionProvider`) | `b6bd06931ccffe89ac49f15b50daadd8a232246281144e4b019c3d018e06ff9d` |

> **Optional Hugging Face `sentence-transformers` Backend**: Setting `HYBRID_RAG_BACKEND=sentence_transformers` switches the embedding and reranking calls to `sentence-transformers/all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2`. The bundled ONNX Runtime models are the default so that offline execution, CI, and checksum-locked evaluation run deterministically on CPU without external network downloads.

---

## 3. 50-Case Ablation Benchmark Results (`as_of_date = 2026-10-27`)

| Retrieval Mode | Recall@5 | MRR@10 | NDCG@10 | Passage Claim Support | RBAC Leakage Rate | Abstention Accuracy | Lexical Slice R@5 | Semantic Slice R@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. FTS5 Lexical Only** | `0.6471` | `0.6324` | `0.6362` | `0.6176` | `0.0000` | `1.0000` | `1.0000` | `0.1429` |
| **2. Dense Only** | `0.9706` | `0.9363` | `0.9450` | `0.9074` | `0.0000` | `1.0000` | `1.0000` | `1.0000` |
| **3. Hybrid RRF ($k=60$)** | `1.0000` | `0.8873` | `0.9154` | `0.8397` | `0.0000` | `1.0000` | `1.0000` | `1.0000` |
| **4. Hybrid RRF + Reranker** | **`1.0000`** | **`0.9853`** | **`0.9891`** | **`0.9868`** | **`0.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** |

### Key Empirical Findings
1. **Lexical vs. Semantic Complementarity**:
   - SQLite FTS5 BM25 achieves `1.0000` `Recall@5` on `lexical_factual` queries containing exact alphanumeric policy codes (`GATE-AI-104`, `SPEC-RET-208`, `CL-ZDR-09`), but drops to `0.1429` on `semantic_multichunk` queries phrased with natural operational synonyms (*"dark launch"*, *"statistical divergence"*, *"citizen fiscal IDs"*).
2. **Reciprocal Rank Fusion ($k=60$)**:
   - Fusing FTS5 and Dense rankings via $\text{RRF}(d) = \frac{1}{60 + r_{\text{lex}}(d)} + \frac{1}{60 + r_{\text{dense}}(d)}$ achieves **`1.0000` `Recall@5`** across all 34 answerable queries without requiring raw score calibration between unbounded BM25 scores and bounded cosine similarities.
3. **Cross-Encoder Reranking Lift**:
   - Rescoring the top-12 RRF candidates with the joint query-passage Cross-Encoder lifts **`MRR@10` from `0.8873` (RRF alone) to `0.9853`** (`+0.3529` over FTS5 Lexical Only) and **`Passage_Claim_Support_Rate` to `0.9868`**.
4. **Zero RBAC Leakage (`0.0000`)**:
   - Because role clearance is enforced *prior* to candidate scoring in both SQLite FTS5 and Dense matrix multiplication, restricted documents (`DOC-HR-008`, `DOC-LEGAL-009`, `DOC-PROC-004`) never leak into unauthorized roles or displace authorized top-$k$ passages.

---

## 4. Known Limitations & Boundaries
- **Synthetic Enterprise Corpus**: The 12 governance documents (`30` page chunks) and 50 evaluation cases are authored synthetic artifacts designed to isolate lexical, semantic, RBAC, freshness, and abstention behaviors.
- **Deterministic Extractive Synthesis by Default**: To guarantee zero external API cost and 100% reproducibility in CI, answer synthesis uses deterministic citation-anchored extraction over retrieved chunks. External LLM synthesis is optional and not required.
- **Domain-Scoped Compact ONNX Models**: The bundled `.onnx` models are lightweight domain-calibrated networks executed via `onnxruntime` CPU provider; evaluating arbitrary open-domain corpora requires either retraining the projection/reranker or enabling `HYBRID_RAG_BACKEND=sentence_transformers`.
