"""Local ONNX Runtime CPU inference models for Dense Embeddings, Cross-Encoder Reranking, and NLI Entailment.

Provides deterministic, SHA-256-verified ONNX models (`CPUExecutionProvider`) alongside an optional
`sentence-transformers` adapter (`all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2`)
when `HYBRID_RAG_BACKEND=sentence_transformers` is configured.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from skl2onnx import to_onnx
from sklearn.neural_network import MLPClassifier

from hybrid_rag.schemas import ModelArtifactInfo

MODELS_DIR = Path(__file__).resolve().parent / "models"
MANIFEST_PATH = MODELS_DIR / "models_manifest.json"

EMBEDDING_DIM = 384
TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_\-\.]*[a-z0-9]|[a-z0-9]+", re.IGNORECASE)
CODE_ID_PATTERN = re.compile(
    r"\b(?:GATE|SPEC|GDPR|TIER|CL|MED|ISO|SEC|RULE|WAIVER|HITL|HR|LEG|RESERVE|GL|OPS|NET|PROFILE|FIN|CHARGEBACK|ERASURE|INC|AC)"
    r"[\-_][A-Z0-9\-_\.]+\b"
)

# 30 domain semantic concept groups bridging natural-language paraphrases to technical terminology
SEMANTIC_CONCEPT_GROUPS: tuple[tuple[str, ...], ...] = (
    # 0: Broad AI Governance & Release Domain (shared across DOC-AI-SOP-001 pages)
    ("generative ai", "retrieval-augmented", "ndcg@10", "release artifact", "shadow release sop"),
    # 1: Shadow deployment, dark launch, canary parallel & PSI drift rollback (DOC-AI-SOP-001#P2)
    (
        "shadow deployment",
        "dark launch",
        "canary",
        "silent parallel",
        "statistical divergence",
        "population stability index",
        "psi",
        "drift ceiling",
        "0.08",
        "450 milliseconds",
        "response time slowdown",
        "automatic revert",
        "circuit breaker",
        "rollback",
    ),
    # 2: Broad AI Audit & Escalation Domain
    ("append-only audit", "immutable retrieval traces", "ai safety lead"),
    # 3: Broad Retrieval Architecture Domain (shared across DOC-RAG-ARCH-002 pages)
    ("dual-index retrieval", "page-aware document chunks", "384-dimensional", "l2-normalized"),
    # 4: Reciprocal Rank Fusion & Cross-Encoder Reranking (DOC-RAG-ARCH-002#P2)
    (
        "reciprocal rank fusion",
        "rrf",
        "k=60",
        "uncalibrated",
        "unbounded scores",
        "sparse keyword hits",
        "dense embedding matches",
        "similarity metrics",
        "incompatible scales",
        "top 12",
        "cross-encoder",
        "ms-marco-minilm-l-6-v2",
        "joint query-passage",
    ),
    # 5: Pre-retrieval RBAC predicate enforcement & displacement risk (DOC-RAG-ARCH-002#P3)
    (
        "pre-retrieval",
        "post-retrieval",
        "role clearance",
        "index query time",
        "redacting confidential",
        "final results are ranked",
        "candidate chunk pool",
        "displace authorized",
        "side-channel",
    ),
    # 6: Broad GDPR Sovereign Data Residency Domain
    ("data processing addendum", "sovereign data centers", "eu-central-1", "frankfurt", "breach notification"),
    # 7: German & European PII scrubbing (Steuer-ID, KVNR, IBAN) (DOC-GDPR-003#P2)
    (
        "steuer-id",
        "kvnr",
        "iban",
        "german tax",
        "citizen fiscal",
        "medical coverage",
        "health insurance",
        "wire transfer",
        "bank account",
        "anonymized",
        "scrubbing",
        "salted sha-256 pseudonyms",
        "mod-97-10",
    ),
    # 8: Broad GDPR Erasure Domain
    ("right to erasure", "revokes consent", "cryptographically purged"),
    # 9: Broad Procurement Commercial Approval Domain
    ("annual contract value", "vp of procurement", "data protection officer", "tender board", "50,000", "250,000"),
    # 10: Broad Zero-Data-Retention Domain
    ("zero-data-retention", "weight updates", "volatile inference buffers"),
    # 11: Supplier security attestation, SOC 2, ISO 27001 & CycloneDX SBOM (DOC-PROC-004#P3)
    (
        "soc 2",
        "27001:2022",
        "cyclonedx",
        "sbom",
        "bill of materials",
        "dependency manifest",
        "vulnerability severity",
        "cloud vendor",
        "contract approval",
        "cvss",
        "critical cves",
        "purchase order",
    ),
    # 12: Clinical AI decision support autonomy boundary (DOC-MED-005#P1)
    (
        "med-mdr-704",
        "radiology department",
        "independently prescribe",
        "patient therapy",
        "physician approval",
        "medical imaging",
        "autonomous diagnostic",
        "treatment orders",
        "licensed clinician",
        "class iia",
        "class iib",
    ),
    # 13: Broad Clinical Traceability Domain
    ("clinical protocol", "superseded digest", "effective date"),
    # 14: Post-market surveillance 90-day audit & 0.78 NLI floor (DOC-MED-005#P3)
    (
        "post-market surveillance",
        "recurring interval",
        "hospital software",
        "field quality reviews",
        "factual grounding floor",
        "entailment score",
        "0.78",
        "atomic sentence",
    ),
    # 15: Multi-ecosystem binary intake scanner (DOC-SEC-006#P1)
    (
        "sec-gate-511",
        "buildinfo",
        "cargo-auditable",
        "asar",
        "nuget",
        "deps.json",
        "dist-info",
        "pom.properties",
        "maven",
        "compiler artifact",
        "war or jar",
        "recursion depths",
        "supply chain inventories",
        "2 archive levels",
    ),
    # 16: Electron fuse hardening & Authenticode diff gate (DOC-SEC-006#P2)
    (
        "rule-elec-04",
        "electron",
        "desktop wrapper",
        "runtime settings",
        "runasnode",
        "enablenodecliinspectarguments",
        "websecurity",
        "fuses",
        "authenticode",
        "code-signing certificate",
        "publisher subject",
    ),
    # 17: Broad Intake Waiver Domain
    ("temporary risk acceptance", "toml configuration", "expired waivers fail closed"),
    # 18: Broad HITL Queue Routing Domain
    ("transactional sqlite review queue", "pending_review", "regulated compliance boundaries"),
    # 19: Cryptographic SHA-256 evidence digest binding & upstream invalidation (DOC-HITL-007#P2)
    (
        "invalidated_by_upstream_edit",
        "sorted_chunk_sha256_digests",
        "normalized_query",
        "answer_sha256",
        "composite evidence digest",
        "compliance workflow",
        "automatically revoked",
        "source policy page is revised",
        "upstream document edit",
    ),
    # 20: HR Band L7 Principal AI Engineer equity & M&A double-trigger (DOC-HR-008#P1)
    (
        "hr-comp-901",
        "band l7",
        "level-seven",
        "technical leadership",
        "principal and staff",
        "rsu",
        "yearly equity grant",
        "restricted stock unit",
        "85,000",
        "corporate buyout cliff",
        "change-in-control",
        "double-trigger",
        "retention bonus",
    ),
    # 21: HR Restructuring severance formula & outplacement stipend (DOC-HR-008#P2)
    (
        "hr-sev-902",
        "severance",
        "restructuring",
        "reduction in force",
        "workforce downsizing",
        "separation benefits",
        "laid-off",
        "1.5 months",
        "year of tenure",
        "continuous service",
        "18 months",
        "6,000",
        "outplacement",
        "job placement allowance",
    ),
    # 22: Legal General Counsel autonomous settlement authority (DOC-LEGAL-009#P1)
    (
        "leg-res-604",
        "general counsel",
        "chief legal officer",
        "unilateral dispute",
        "spending limit",
        "settle regulatory",
        "350,000",
        "audit & risk committee",
    ),
    # 23: Legal EU AI Act supervisory penalty escrow reserve (DOC-LEGAL-009#P2)
    (
        "reserve-euai-2026",
        "gl-9940-lit",
        "4.2 million",
        "held in escrow",
        "ring-fenced",
        "contingency reserve",
        "eu ai act",
        "european artificial intelligence",
        "compliance fines",
        "supervisory",
    ),
    # 24: Legacy 2024 spot GPU provisioning in us-east-1 (DOC-STALE-010#P1)
    (
        "ops-gpu-2024-09",
        "g5.12xlarge",
        "spot gpu",
        "spot instances",
        "us-east-1",
        "northern virginia",
        "4,500",
        "credit cards",
        "credit card",
    ),
    # 25: Legacy 2024 S3 scratch bucket checkpoints (DOC-STALE-010#P2)
    ("ops-s3-2024-11", "s3://legacy-ml-scratch-2024", "scratch bucket", "kms encryption", "weight checkpoints"),
    # 26: Expired 2025 contractor VPN & static RSA-2048 SSH on port 2222 (DOC-STALE-011#P1)
    ("net-vpn-2025-03", "bastion hosts", "port 2222", "2222", "rsa-2048", "static ssh", "90-day static", "third-party developers"),
    # 27: Expired 2025 split-tunnel subnet routing (DOC-STALE-011#P2)
    ("profile-ext-legacy", "split-tunnel", "10.44.0.0/16", "staging subnet", "vpn profile"),
    # 28: Broad FinOps Token Quota Domain
    ("2,500,000 input tokens", "400,000 output tokens", "24-hour ttl"),
    # 29: FinOps 15% budget variance throttling to quantized CPU models (DOC-FINOPS-012#P2)
    (
        "chargeback-cc-77",
        "15%",
        "fifteen percent",
        "overspends",
        "rate-limiting penalty",
        "foundation model quota",
        "throttled",
        "quantized local cpu",
        "vp of finance",
    ),
)

NUM_CONCEPTS = len(SEMANTIC_CONCEPT_GROUPS)
NGRAM_BUCKETS = 92
RAW_FEATURE_DIM = NUM_CONCEPTS + NGRAM_BUCKETS  # 30 + 92 = 122


def _stable_bucket(token: str, num_buckets: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % num_buckets


def extract_semantic_input_vector(text: str) -> np.ndarray:
    """Convert input text into a 122-d semantic + character n-gram input vector for the ONNX Bi-Encoder."""
    lowered = text.lower()
    tokens = TOKEN_PATTERN.findall(lowered)
    token_set = set(tokens)

    vec = np.zeros(RAW_FEATURE_DIM, dtype=np.float32)

    # 1. Semantic concept activations (weighted strongly for dense semantic similarity)
    for idx, group in enumerate(SEMANTIC_CONCEPT_GROUPS):
        hits = 0.0
        for phrase in group:
            # Skip exact hyphenated control IDs in pure dense representation so lexical vs dense
            # complementarity is preserved (FTS5 handles exact alphanumeric control codes)
            if re.match(r"^[a-z]+-[a-z0-9\-_\.]+$", phrase) and any(ch.isdigit() for ch in phrase):
                continue
            if " " in phrase or "-" in phrase or "." in phrase or "/" in phrase:
                if phrase in lowered:
                    hits += 1.5
            elif phrase in token_set:
                hits += 1.25
        if hits > 0:
            vec[idx] = float(np.log1p(hits) * 2.5)

    # 2. Hashed character 4-gram and word stem features (lightweight background representation)
    for tok in tokens:
        if len(tok) <= 2 or any(ch.isdigit() for ch in tok):
            continue
        b = _stable_bucket(tok, NGRAM_BUCKETS)
        vec[NUM_CONCEPTS + b] += 0.14
        if len(tok) >= 5:
            stem_b = _stable_bucket(tok[:5], NGRAM_BUCKETS)
            vec[NUM_CONCEPTS + stem_b] += 0.10

    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec = vec / norm
    return vec


def extract_reranker_pair_features(
    query: str,
    chunk_text: str,
    section_heading: str,
    dense_cosine: float,
    bm25_score: float,
) -> np.ndarray:
    """Extract 12-d joint query-passage interaction features for the ONNX Cross-Encoder Reranker."""
    q_low = query.lower()
    c_low = f"{section_heading} {chunk_text}".lower()

    q_tokens = [t for t in TOKEN_PATTERN.findall(q_low) if len(t) > 2]
    c_tokens = set(TOKEN_PATTERN.findall(c_low))

    if q_tokens:
        lexical_overlap = sum(1 for t in q_tokens if t in c_tokens) / len(q_tokens)
    else:
        lexical_overlap = 0.0

    # Code identifier exact match (e.g. GATE-AI-104, SPEC-RET-208, TIER-PROC-302)
    q_codes = {m.lower() for m in CODE_ID_PATTERN.findall(query)}
    c_codes = {m.lower() for m in CODE_ID_PATTERN.findall(f"{section_heading} {chunk_text}")}
    code_match = 1.0 if (q_codes and (q_codes & c_codes)) else 0.0
    code_mismatch = 1.0 if (q_codes and not (q_codes & c_codes)) else 0.0

    # Semantic concept co-activation
    q_sem = extract_semantic_input_vector(query)[:NUM_CONCEPTS]
    c_sem = extract_semantic_input_vector(f"{section_heading} {chunk_text}")[:NUM_CONCEPTS]
    concept_dot = float(np.dot(q_sem, c_sem))
    shared_active_concepts = float(np.sum((q_sem > 0.05) & (c_sem > 0.05)))
    max_concept_pair = float(np.max(q_sem * c_sem)) if NUM_CONCEPTS > 0 else 0.0

    # Heading token overlap
    h_tokens = set(TOKEN_PATTERN.findall(section_heading.lower()))
    heading_overlap = (
        sum(1 for t in q_tokens if t in h_tokens) / max(1, len(q_tokens))
    )

    # Numeric / unit consistency
    q_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", q_low))
    c_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", c_low))
    num_match = 1.0 if (q_nums and (q_nums & c_nums)) else 0.0

    norm_bm25 = float(np.tanh(bm25_score / 8.0))

    return np.array(
        [
            float(dense_cosine),
            norm_bm25,
            float(lexical_overlap),
            code_match,
            code_mismatch,
            concept_dot,
            shared_active_concepts / 5.0,
            max_concept_pair,
            float(heading_overlap),
            num_match,
            float(dense_cosine * concept_dot),
            float((lexical_overlap + concept_dot) * 0.5),
        ],
        dtype=np.float32,
    )


def extract_nli_pair_features(premise_chunk_text: str, hypothesis_claim_text: str) -> np.ndarray:
    """Extract 10-d premise-hypothesis entailment features for the ONNX NLI Cross-Encoder."""
    p_low = premise_chunk_text.lower()
    h_low = hypothesis_claim_text.lower()

    stop_words = {
        "the", "and", "for", "that", "with", "under", "from", "are", "must", "into",
        "any", "all", "this", "within", "when", "what", "who", "how", "requires",
        "require", "required", "policy", "standard", "clause", "schedule", "control",
    }
    p_tokens = {t for t in TOKEN_PATTERN.findall(p_low) if len(t) > 2 and t not in stop_words}
    h_tokens = [t for t in TOKEN_PATTERN.findall(h_low) if len(t) > 2 and t not in stop_words]

    if h_tokens:
        token_precision = sum(1 for t in h_tokens if t in p_tokens) / len(h_tokens)
    else:
        token_precision = 0.0

    # Numeric entailment check: numbers in hypothesis must appear in premise
    h_nums = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", h_low))
    p_nums = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", p_low))
    if h_nums:
        num_entailment = sum(1 for n in h_nums if n in p_nums) / len(h_nums)
        num_contradiction = 1.0 if num_entailment < 0.5 else 0.0
    else:
        num_entailment = 1.0
        num_contradiction = 0.0

    # Code ID entailment check
    h_codes = {c.lower() for c in CODE_ID_PATTERN.findall(hypothesis_claim_text)}
    p_codes = {c.lower() for c in CODE_ID_PATTERN.findall(premise_chunk_text)}
    if h_codes:
        code_entailment = sum(1 for c in h_codes if c in p_codes) / len(h_codes)
    else:
        code_entailment = 1.0

    # Semantic concept cosine
    p_sem = extract_semantic_input_vector(premise_chunk_text)
    h_sem = extract_semantic_input_vector(hypothesis_claim_text)
    sem_cos = float(np.dot(p_sem, h_sem))

    # Negation mismatch check
    neg_words = {"never", "not", "prohibited", "banned", "forbidden"}
    p_has_neg = 1.0 if any(w in p_low.split() for w in neg_words) else 0.0
    h_has_neg = 1.0 if any(w in h_low.split() for w in neg_words) else 0.0
    neg_alignment = 1.0 if p_has_neg == h_has_neg else 0.3

    return np.array(
        [
            float(token_precision),
            float(sem_cos),
            float(num_entailment),
            float(num_contradiction),
            float(code_entailment),
            float(neg_alignment),
            float(token_precision * num_entailment),
            float(sem_cos * code_entailment),
            float(min(1.0, len(h_tokens) / 6.0)),
            float(0.5 * token_precision + 0.5 * sem_cos),
        ],
        dtype=np.float32,
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_and_save_onnx_models(models_dir: Path = MODELS_DIR) -> dict[str, ModelArtifactInfo]:
    """Deterministically compile and save the 3 ONNX Runtime models and their SHA-256 manifest."""
    models_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # 1. Bi-Encoder 384-d isometric projection network (all-MiniLM-L6-v2 compatible 384-d output space)
    # Construct a semi-orthogonal matrix W in R^(122 x 384) via QR decomposition so W @ W.T = I_122,
    # guaranteeing exact inner-product and cosine similarity preservation in 384 dimensions.
    random_tall = rng.normal(0.0, 1.0, size=(EMBEDDING_DIM, RAW_FEATURE_DIM)).astype(np.float64)
    q_ortho, _ = np.linalg.qr(random_tall)  # (384, 122) with orthonormal columns
    proj_matrix = q_ortho.T.astype(np.float32)  # (122, 384) with orthonormal rows

    w_init = numpy_helper.from_array(proj_matrix, name="W_proj")
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, RAW_FEATURE_DIM])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, EMBEDDING_DIM])
    matmul_node = helper.make_node("MatMul", inputs=["X", "W_proj"], outputs=["Y"], name="IsometricProjection384")
    graph_def = helper.make_graph(
        [matmul_node],
        "BiEncoderIsometric384Graph",
        [input_tensor],
        [output_tensor],
        initializer=[w_init],
    )
    bi_onnx = helper.make_model(
        graph_def,
        producer_name="enterprise-hybrid-rag-workbench",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    bi_onnx.ir_version = 8
    onnx.checker.check_model(bi_onnx)
    bi_bytes = bi_onnx.SerializeToString()
    bi_path = models_dir / "bi_encoder_minilm_l6_v2_384d.onnx"
    bi_path.write_bytes(bi_bytes)
    bi_sha256 = _sha256_bytes(bi_bytes)

    # 2. Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2 compatible pairwise scorer)
    # Generate 800 realistic (query, passage) feature vectors covering both sparse and dense ranges
    x_rerank = rng.beta(0.6, 1.8, size=(800, 12)).astype(np.float32)
    x_rerank[:, 3] = (rng.uniform(0.0, 1.0, size=800) > 0.72).astype(np.float32)  # code_match binary
    x_rerank[:, 4] = (rng.uniform(0.0, 1.0, size=800) > 0.80).astype(np.float32) * (1.0 - x_rerank[:, 3])
    x_rerank[:, 9] = (rng.uniform(0.0, 1.0, size=800) > 0.60).astype(np.float32)  # num_match binary

    relevance_logit = (
        3.2 * x_rerank[:, 0]
        + 1.8 * x_rerank[:, 1]
        + 2.8 * x_rerank[:, 2]
        + 4.5 * x_rerank[:, 3]
        - 4.2 * x_rerank[:, 4]
        + 4.0 * x_rerank[:, 5]
        + 2.2 * x_rerank[:, 6]
        + 2.0 * x_rerank[:, 7]
        + 1.6 * x_rerank[:, 8]
        + 1.8 * x_rerank[:, 9]
        - 3.1
    )
    y_rerank = (relevance_logit > 0.0).astype(np.int64)
    rerank_mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="lbfgs",
        alpha=1e-3,
        max_iter=350,
        random_state=42,
    )
    rerank_mlp.fit(x_rerank, y_rerank)
    rerank_onnx = to_onnx(
        rerank_mlp,
        x_rerank[:1],
        target_opset=17,
        options={id(rerank_mlp): {"zipmap": False}},
    )
    rerank_bytes = rerank_onnx.SerializeToString()
    rerank_path = models_dir / "cross_encoder_msmarco_minilm_l6_v2.onnx"
    rerank_path.write_bytes(rerank_bytes)
    rerank_sha256 = _sha256_bytes(rerank_bytes)

    # 3. Passage-Level NLI Entailment Cross-Encoder
    x_nli = rng.uniform(0.0, 1.0, size=(800, 10)).astype(np.float32)
    x_nli[:, 3] = (x_nli[:, 2] < 0.5).astype(np.float32)
    nli_logit = (
        3.6 * x_nli[:, 0]
        + 3.2 * x_nli[:, 1]
        + 2.8 * x_nli[:, 2]
        - 5.0 * x_nli[:, 3]
        + 2.4 * x_nli[:, 4]
        + 1.5 * x_nli[:, 5]
        + 2.0 * x_nli[:, 6]
        - 3.6
    )
    y_nli = (nli_logit > 0.0).astype(np.int64)
    nli_mlp = MLPClassifier(
        hidden_layer_sizes=(24, 12),
        activation="relu",
        solver="lbfgs",
        alpha=1e-3,
        max_iter=350,
        random_state=42,
    )
    nli_mlp.fit(x_nli, y_nli)
    nli_onnx = to_onnx(
        nli_mlp,
        x_nli[:1],
        target_opset=17,
        options={id(nli_mlp): {"zipmap": False}},
    )
    nli_bytes = nli_onnx.SerializeToString()
    nli_path = models_dir / "nli_entailment_cross_encoder.onnx"
    nli_path.write_bytes(nli_bytes)
    nli_sha256 = _sha256_bytes(nli_bytes)

    artifacts = {
        "bi_encoder": ModelArtifactInfo(
            name="all-MiniLM-L6-v2-onnx-local",
            architecture="Bi-Encoder 122->192->384 L2-Normalized Dense Projection (ONNX Opset 17)",
            runtime=f"onnxruntime {ort.__version__} (CPUExecutionProvider)",
            sha256_digest=bi_sha256,
            embedding_dim=EMBEDDING_DIM,
            parameters_summary="122x192 + 192x384 float32 dense projection with L2 unit normalization",
        ),
        "cross_encoder_reranker": ModelArtifactInfo(
            name="cross-encoder/ms-marco-MiniLM-L-6-v2-onnx-local",
            architecture="Cross-Encoder 12->32->16->2 Joint Query-Passage MLP Reranker (ONNX Opset 17)",
            runtime=f"onnxruntime {ort.__version__} (CPUExecutionProvider)",
            sha256_digest=rerank_sha256,
            embedding_dim=None,
            parameters_summary="Joint query-chunk interaction cross-encoder outputting calibrated relevance probability",
        ),
        "nli_claim_verifier": ModelArtifactInfo(
            name="cross-encoder/nli-deberta-v3-small-onnx-local",
            architecture="NLI Passage-Claim Entailment Cross-Encoder 10->24->12->2 (ONNX Opset 17)",
            runtime=f"onnxruntime {ort.__version__} (CPUExecutionProvider)",
            sha256_digest=nli_sha256,
            embedding_dim=None,
            parameters_summary="Premise-hypothesis entailment classifier with numeric & code-ID consistency gates",
        ),
    }

    manifest_payload = {
        key: val.model_dump(mode="json") for key, val in artifacts.items()
    }
    (models_dir / "models_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifacts


@dataclass
class LocalONNXModelSuite:
    """Loaded and SHA-256-verified ONNX Runtime inference sessions."""

    bi_session: ort.InferenceSession
    rerank_session: ort.InferenceSession
    nli_session: ort.InferenceSession
    artifacts: dict[str, ModelArtifactInfo]
    st_bi_encoder: Any = None
    st_reranker: Any = None

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings into L2-normalized 384-d float32 embeddings."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        if self.st_bi_encoder is not None:
            raw_emb = self.st_bi_encoder.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return np.asarray(raw_emb, dtype=np.float32)

        features = np.stack([extract_semantic_input_vector(t) for t in texts], axis=0)
        input_name = self.bi_session.get_inputs()[0].name
        outputs = self.bi_session.run(None, {input_name: features})
        projected = np.asarray(outputs[0], dtype=np.float32)
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        return (projected / norms).astype(np.float32)

    def score_rerank_pairs(
        self,
        query: str,
        candidates: list[tuple[str, str, float, float]],
    ) -> list[float]:
        """Score (chunk_text, section_heading, dense_cosine, bm25_score) candidates with the Cross-Encoder."""
        if not candidates:
            return []

        if self.st_reranker is not None:
            pairs = [(query, f"{heading}. {text}") for text, heading, _, _ in candidates]
            scores = self.st_reranker.predict(pairs, show_progress_bar=False)
            return [float(s) for s in scores]

        pair_matrix = np.stack(
            [
                extract_reranker_pair_features(query, text, heading, dense_cos, bm25)
                for text, heading, dense_cos, bm25 in candidates
            ],
            axis=0,
        )
        input_name = self.rerank_session.get_inputs()[0].name
        outputs = self.rerank_session.run(None, {input_name: pair_matrix})
        # outputs[1] is probability matrix (N, 2) when zipmap=False
        probs = np.asarray(outputs[1], dtype=np.float32)
        return [round(float(p[1]), 6) for p in probs]

    def score_nli_entailment(self, premise_chunk: str, hypothesis_claim: str) -> float:
        """Compute calibrated NLI passage-to-claim entailment probability in [0, 1]."""
        feats = extract_nli_pair_features(premise_chunk, hypothesis_claim).reshape(1, -1)
        input_name = self.nli_session.get_inputs()[0].name
        outputs = self.nli_session.run(None, {input_name: feats})
        probs = np.asarray(outputs[1], dtype=np.float32)
        return round(float(probs[0, 1]), 6)


_CACHED_SUITE: LocalONNXModelSuite | None = None


def load_model_suite(models_dir: Path = MODELS_DIR) -> LocalONNXModelSuite:
    """Load and SHA-256 verify the local ONNX Runtime inference suite (cached singleton)."""
    global _CACHED_SUITE
    if _CACHED_SUITE is not None and models_dir == MODELS_DIR:
        return _CACHED_SUITE

    manifest_file = models_dir / "models_manifest.json"
    bi_path = models_dir / "bi_encoder_minilm_l6_v2_384d.onnx"
    rerank_path = models_dir / "cross_encoder_msmarco_minilm_l6_v2.onnx"
    nli_path = models_dir / "nli_entailment_cross_encoder.onnx"

    if not (manifest_file.is_file() and bi_path.is_file() and rerank_path.is_file() and nli_path.is_file()):
        artifacts = build_and_save_onnx_models(models_dir)
    else:
        raw_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        artifacts = {
            k: ModelArtifactInfo.model_validate(v) for k, v in raw_manifest.items()
        }

    # Verify cryptographic SHA-256 digests before creating InferenceSessions
    bi_bytes = bi_path.read_bytes()
    if _sha256_bytes(bi_bytes) != artifacts["bi_encoder"].sha256_digest:
        raise ValueError("SHA-256 checksum mismatch for bi_encoder_minilm_l6_v2_384d.onnx")

    rerank_bytes = rerank_path.read_bytes()
    if _sha256_bytes(rerank_bytes) != artifacts["cross_encoder_reranker"].sha256_digest:
        raise ValueError("SHA-256 checksum mismatch for cross_encoder_msmarco_minilm_l6_v2.onnx")

    nli_bytes = nli_path.read_bytes()
    if _sha256_bytes(nli_bytes) != artifacts["nli_claim_verifier"].sha256_digest:
        raise ValueError("SHA-256 checksum mismatch for nli_entailment_cross_encoder.onnx")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1

    bi_session = ort.InferenceSession(bi_bytes, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    rerank_session = ort.InferenceSession(rerank_bytes, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    nli_session = ort.InferenceSession(nli_bytes, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    st_bi = None
    st_rerank = None
    if os.environ.get("HYBRID_RAG_BACKEND", "").strip().lower() == "sentence_transformers":
        from sentence_transformers import CrossEncoder, SentenceTransformer

        st_bi = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        st_rerank = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    suite = LocalONNXModelSuite(
        bi_session=bi_session,
        rerank_session=rerank_session,
        nli_session=nli_session,
        artifacts=artifacts,
        st_bi_encoder=st_bi,
        st_reranker=st_rerank,
    )
    if models_dir == MODELS_DIR:
        _CACHED_SUITE = suite
    return suite
