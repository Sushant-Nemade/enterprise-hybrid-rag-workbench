"""Canonical 12 enterprise governance documents and 50 labelled evaluation benchmark cases."""

from __future__ import annotations

import json
from pathlib import Path

from hybrid_rag.schemas import (
    EvalCase,
    EvalCaseCategory,
    EvalDatasetFile,
    PageSectionInput,
    RawCorpusDocument,
    RoleEnum,
)


def get_canonical_corpus_documents() -> list[RawCorpusDocument]:
    """Return the 12 synthetic enterprise SOP, procurement, clinical, security, and GDPR documents."""
    return [
        RawCorpusDocument(
            doc_id="DOC-AI-SOP-001",
            title="Enterprise GenAI Model Deployment & Shadow Release SOP",
            allowed_roles=["public", "engineering", "legal_compliance", "executive"],
            effective_date="2026-08-15",
            max_staleness_days=180,
            domain="ai_governance",
            version="2.4",
            pages=[
                PageSectionInput(
                    page_number=1,
                    section_heading="1. Pre-Production Release Gate (GATE-AI-104)",
                    content=(
                        "Under Enterprise Release Gate GATE-AI-104, every generative AI and retrieval-augmented "
                        "service must pass an offline deterministic benchmark before promotion to staging or "
                        "production. The release artifact must record SHA-256 digests for both the evaluation "
                        "dataset and model weights, achieve a minimum NDCG@10 score of 0.82, maintain a "
                        "Passage Claim Support Rate of at least 0.85, and demonstrate a pre-retrieval RBAC "
                        "Leakage Rate of exactly 0.0 across all role tiers."
                    ),
                ),
                PageSectionInput(
                    page_number=2,
                    section_heading="2. Shadow Deployment Telemetry & PSI Drift Ceiling",
                    content=(
                        "New retrieval or reranking models must execute in shadow deployment mode (canary silent "
                        "parallel evaluation alongside the active production baseline) for a minimum of 72 hours. "
                        "Automated telemetry monitors query-embedding Population Stability Index (PSI) and latency. "
                        "If the Population Stability Index (PSI) distribution shift exceeds the drift ceiling of "
                        "0.08 or CPU P99 latency exceeds 450 milliseconds, the circuit breaker triggers an "
                        "immediate rollback to the deterministic baseline."
                    ),
                ),
                PageSectionInput(
                    page_number=3,
                    section_heading="3. Audit Log Retention & Hallucination Incident Escalation",
                    content=(
                        "Immutable retrieval traces containing query hashes, role clearances, and cited chunk "
                        "SHA-256 digests must be retained for 90 calendar days in append-only audit storage. "
                        "If post-synthesis NLI monitoring detects an unsupported claim rate above 3.0% over any "
                        "rolling 24-hour window, an incident ticket under code INC-SEV2-RAG must be escalated "
                        "to the AI Safety Lead within 4 hours."
                    ),
                ),
            ],
        ),
        RawCorpusDocument(
            doc_id="DOC-RAG-ARCH-002",
            title="Hybrid Retrieval & Cross-Encoder Reranking Architecture Standard",
            allowed_roles=["public", "engineering", "executive"],
            effective_date="2026-09-01",
            max_staleness_days=365,
            domain="retrieval_architecture",
            version="3.1",
            pages=[
                PageSectionInput(
                    page_number=1,
                    section_heading="1. Dual-Index Lexical and Dense Specification (SPEC-RET-208)",
                    content=(
                        "Architecture Specification SPEC-RET-208 mandates a dual-index retrieval layer over "
                        "page-aware document chunks. The lexical index uses SQLite FTS5 with BM25 scoring "
                        "(parameters k1=1.2 and b=0.75) to capture exact alphanumeric identifiers, regulatory "
                        "codes, and statutory clauses. The dense semantic index projects chunks into 384-dimensional "
                        "L2-normalized embeddings (all-MiniLM-L6-v2 / ONNX CPU execution) scored via exact "
                        "cosine similarity."
                    ),
                ),
                PageSectionInput(
                    page_number=2,
                    section_heading="2. Reciprocal Rank Fusion (RRF k=60) & Cross-Encoder Reranking",
                    content=(
                        "To reconcile uncalibrated BM25 unbounded scores with bounded cosine similarities [-1, 1], "
                        "rankings are combined via Reciprocal Rank Fusion using smoothing constant k=60: "
                        "RRF_score(d) = 1/(60 + rank_lexical(d)) + 1/(60 + rank_dense(d)). The top 12 fused "
                        "candidates are then rescored by a local Cross-Encoder reranker "
                        "(cross-encoder/ms-marco-MiniLM-L-6-v2) that computes joint query-passage interaction "
                        "logits before selecting the final top-k context window."
                    ),
                ),
                PageSectionInput(
                    page_number=3,
                    section_heading="3. Pre-Retrieval Security Predicate Enforcement (AC-PRE-01)",
                    content=(
                        "Security Control AC-PRE-01 strictly prohibits post-retrieval permission filtering. "
                        "Role-based access control predicates must filter the candidate chunk pool before "
                        "executing SQLite FTS5 match queries or dense matrix products. Post-retrieval filtering "
                        "is banned because restricted chunks can displace authorized evidence from the top-k "
                        "window and create side-channel inference risks."
                    ),
                ),
            ],
        ),
        RawCorpusDocument(
            doc_id="DOC-GDPR-003",
            title="GDPR Article 28 Data Processing, PII Redaction & Cross-Border Transfer Standard",
            allowed_roles=["engineering", "legal_compliance", "executive"],
            effective_date="2026-07-10",
            max_staleness_days=180,
            domain="privacy_gdpr",
            version="2.0",
            pages=[
                PageSectionInput(
                    page_number=1,
                    section_heading="1. Subprocessor Onboarding & Sovereign Residency (GDPR-ART28-ANNEX-B)",
                    content=(
                        "Under Clause GDPR-ART28-ANNEX-B, any external cloud or AI subprocessor handling "
                        "enterprise data must execute a countersigned Data Processing Addendum (DPA) prior to "
                        "integration. All vector indices, relational databases, and telemetry logs must reside "
                        "exclusively within EU/EEA sovereign data centers in eu-central-1 (Frankfurt), with a "
                        "binding 72-hour personal data breach notification SLA."
                    ),
                ),
                PageSectionInput(
                    page_number=2,
                    section_heading="2. Pre-Ingestion PII Scrubbing & Checksum Pseudonymization",
                    content=(
                        "Before any document is chunked or embedded into the retrieval store, deterministic PII "
                        "detectors must screen for German tax identifiers (Steuer-ID, 11 digits), statutory "
                        "health insurance numbers (KVNR, 10-character alphanumeric), and international bank "
                        "account numbers (IBAN, MOD-97-10 checksum). Confirmed personal identifiers must be "
                        "replaced with salted SHA-256 pseudonyms prior to indexing."
                    ),
                ),
                PageSectionInput(
                    page_number=3,
                    section_heading="3. Article 17 Right to Erasure Purge SLA (ERASURE-SLA-14D)",
                    content=(
                        "Workflow ERASURE-SLA-14D governs GDPR Article 17 erasure requests. When a data subject "
                        "or enterprise client revokes consent, all associated document chunks, SQLite FTS5 "
                        "inverted index tokens, dense vector embeddings, and cached HITL review records must "
                        "be cryptographically purged and verified absent within 14 calendar days."
                    ),
                ),
            ],
        ),
        RawCorpusDocument(
            doc_id="DOC-PROC-004",
            title="Enterprise Cloud & AI Vendor Procurement Policy",
            allowed_roles=["procurement", "legal_compliance", "executive"],
            effective_date="2026-06-20",
            max_staleness_days=180,
            domain="procurement",
            version="4.2",
            pages=[
                PageSectionInput(
                    page_number=1,
                    section_heading="1. Annual Contract Value Approval Tiers (TIER-PROC-302)",
                    content=(
                        "Schedule TIER-PROC-302 defines commercial approval authority by Annual Contract Value "
                        "(ACV). Contracts up to EUR 50,000 require Procurement Lead approval. Contracts between "
                        "EUR 50,001 and EUR 250,000 require joint sign-off from the VP of Procurement and the "
                        "Data Protection Officer (DPO). Any AI or cloud agreement exceeding EUR 250,000 ACV "
                        "requires formal authorization from the Executive Tender Board."
                    ),
                ),
                PageSectionInput(
                    page_number=2,
                    section_heading="2. Mandatory Zero-Data-Retention Clause (CL-ZDR-09)",
                    content=(
                        "Every commercial agreement with a foundation model or external inference provider must "
                        "incorporate Clause CL-ZDR-09 (Mandatory Zero-Data-Retention). Under CL-ZDR-09, the "
                        "vendor warrants that enterprise prompts, retrieved passages, and outputs are never "
                        "used for model training or weight updates, and volatile inference buffers are wiped "
                        "within 24 hours."
                    ),
                ),
                PageSectionInput(
                    page_number=3,
                    section_heading="3. Supplier Security Attestation & CycloneDX SBOM Requirement",
                    content=(
                        "Prior to purchase order issuance, software vendors must submit a current SOC 2 Type II "
                        "report or ISO/IEC 27001:2022 certificate alongside a machine-readable CycloneDX 1.6 "
                        "Software Bill of Materials (SBOM). Artifacts with unresolved critical CVEs (CVSS >= 9.0) "
                        "are blocked from procurement onboarding."
                    ),
                ),
            ],
        ),
        RawCorpusDocument(
            doc_id="DOC-MED-005",
            title="Regulated Medical Device & Clinical AI Software Governance (ISO 13485 / EU MDR)",
            allowed_roles=["engineering", "legal_compliance", "executive"],
            effective_date="2026-08-01",
            max_staleness_days=150,
            domain="clinical_regulatory",
            version="1.8",
            pages=[
                PageSectionInput(
                    page_number=1,
                    section_heading="1. Clinical Decision Support Autonomy Boundary (MED-MDR-704)",
                    content=(
                        "Directive MED-MDR-704 governs AI-assisted workflows in healthcare and medical imaging "
                        "environments under EU MDR and ISO 13485. Generative and retrieval systems are strictly "
                        "classified as clinical documentation and evidence-lookup aids; they must never issue "
                        "autonomous diagnostic conclusions or treatment orders. Licensed clinician sign-off is "
                        "mandatory for all Class IIa and Class IIb workflows."
                    ),
                ),
                PageSectionInput(
                    page_number=2,
                    section_heading="2. Guideline Provenance & Digest Traceability (ISO-13485-CL-7.5.8)",
                    content=(
                        "In accordance with traceability control ISO-13485-CL-7.5.8, every synthesized clinical "
                        "protocol answer must display the exact source document ID, page number, effective date, "
                        "and SHA-256 content digest. If an underlying clinical SOP is updated, all previously "
                        "approved summaries referencing the superseded digest are automatically invalidated."
                    ),
                ),
                PageSectionInput(
                    page_number=3,
                    section_heading="3. Post-Market Surveillance & NLI Entailment Floor",
                    content=(
                        "Post-Market Surveillance (PMS) audits must be executed every 90 days on clinical "
                        "retrieval pipelines. Synthesized responses must achieve a minimum NLI passage entailment "
                        "score of 0.78 on every atomic sentence, and any response citing a guideline within "
                        "15 days of staleness expiration must display an explicit clinical governance warning."
                    ),
                ),
            ],
        ),
        RawCorpusDocument(
            doc_id="DOC-SEC-006",
            title="Third-Party Binary & Open-Source Software Intake Gatekeeper SOP",
            allowed_roles=["public", "engineering", "procurement", "legal_compliance", "executive"],
            effective_date="2026-09-12",
            max_staleness_days=180,
            domain="devsecops_intake",
            version="3.0",
            pages=[
                PageSectionInput(
                    page_number=1,
                    section_heading="1. Multi-Ecosystem Binary & Archive Inventory (SEC-GATE-511)",
                    content=(
                        "Standard SEC-GATE-511 requires automated read-only intake inspection of third-party "
                        "packages and compiled binaries before enterprise use. The scanner extracts embedded "
                        "dependency metadata across Go buildinfo, Rust cargo-auditable sections, Electron ASAR "
                        "bundles, NuGet deps.json, Python dist-info, and nested Maven pom.properties up to "
                        "2 archive levels deep."
                    ),
                ),
                PageSectionInput(
                    page_number=2,
                    section_heading="2. Electron Fuse Hardening & Code-Signing Diff Gate (RULE-ELEC-04)",
                    content=(
                        "Under Enforcement Rule RULE-ELEC-04, desktop Electron applications must disable "
                        "RunAsNode and EnableNodeCliInspectArguments fuses and keep webSecurity set to true. "
                        "Any binary update that weakens Electron fuses, enables insecure browser flags, strips "
                        "Windows Authenticode signatures, or alters the Authenticode publisher subject triggers "
                        "an automatic REJECT or mandatory security review."
                    ),
                ),
                PageSectionInput(
                    page_number=3,
                    section_heading="3. Expiring Policy Waivers (WAIVER-MAX-30D)",
                    content=(
                        "Temporary risk acceptance for non-critical intake findings is governed by control "
                        "WAIVER-MAX-30D. Every policy waiver must be recorded in TOML configuration with a "
                        "named security approver, issue tracker reference, and an expiration timestamp no more "
                        "than 30 calendar days from issuance. Expired waivers fail closed in CI."
                    ),
                ),
            ],
        ),
        RawCorpusDocument(
            doc_id="DOC-HITL-007",
            title="Cryptographic Human-in-the-Loop (HITL) Review & Digest Binding Standard",
            allowed_roles=["public", "engineering", "legal_compliance", "executive"],
            effective_date="2026-09-20",
            max_staleness_days=180,
            domain="hitl_governance",
            version="1.5",
            pages=[
                PageSectionInput(
                    page_number=1,
                    section_heading="1. Mandatory Review Queue Routing Criteria (HITL-DIGEST-801)",
                    content=(
                        "Control HITL-DIGEST-801 specifies when a RAG response cannot be released directly to "
                        "end users and must enter the transactional SQLite review queue with status "
                        "pending_review. Automatic queueing occurs whenever: (a) any retrieved passage is "
                        "flagged as stale beyond max_staleness_days, (b) the passage claim support rate falls "
                        "below 0.72, or (c) the query touches regulated compliance boundaries."
                    ),
                ),
                PageSectionInput(
                    page_number=2,
                    section_heading="2. SHA-256 Evidence Digest Binding & Upstream Invalidation",
                    content=(
                        "Human reviewer sign-off (approve or reject) is cryptographically bound to the composite "
                        "evidence digest: SHA256(normalized_query || sorted_chunk_sha256_digests || answer_sha256). "
                        "Before serving any previously approved response, the runtime recomputes the digest "
                        "against current corpus chunk hashes. Any upstream document edit changes the chunk "
                        "SHA-256 and automatically transitions the record status to invalidated_by_upstream_edit."
                    ),
                ),
            ],
        ),
        RawCorpusDocument(
            doc_id="DOC-HR-008",
            title="Confidential Executive Compensation, M&A Retention & Workforce Severance Policy",
            allowed_roles=["hr_restricted", "executive"],
            effective_date="2026-08-10",
            max_staleness_days=180,
            domain="hr_confidential",
            version="2.1",
            pages=[
                PageSectionInput(
                    page_number=1,
                    section_heading="1. Band L7 Principal AI Engineer Equity & M&A Acceleration (HR-COMP-901)",
                    content=(
                        "CONFIDENTIAL SCHEDULE HR-COMP-901: Principal and Staff AI Engineers classified in "
                        "Band L7 are eligible for an annual Restricted Stock Unit (RSU) refresh ceiling of "
                        "EUR 85,000, vesting quarterly over 4 years. In the event of a change-in-control M&A "
                        "transaction, Band L7 agreements include a 12-month double-trigger vesting acceleration "
                        "clause and a retention bonus equal to 35% of base salary."
                    ),
                ),
                PageSectionInput(
                    page_number=2,
                    section_heading="2. Involuntary Restructuring Severance Formula (HR-SEV-902)",
                    content=(
                        "CONFIDENTIAL SCHEDULE HR-SEV-902: In an organizational restructuring or reduction in "
                        "force, eligible employees receive a severance payout of 1.5 months of base salary per "
                        "completed year of continuous service, up to a maximum cap of 18 months, plus a fixed "
                        "EUR 6,000 career outplacement transition stipend."
                    ),
                ),
            ],
        ),
        RawCorpusDocument(
            doc_id="DOC-LEGAL-009",
            title="Board Litigation Reserve, Regulatory Investigation & Settlement Authority Protocol",
            allowed_roles=["legal_compliance", "executive"],
            effective_date="2026-07-25",
            max_staleness_days=180,
            domain="legal_privileged",
            version="1.9",
            pages=[
                PageSectionInput(
                    page_number=1,
                    section_heading="1. Autonomous Regulatory Settlement Authority (LEG-RES-604)",
                    content=(
                        "PRIVILEGED PROTOCOL LEG-RES-604: The General Counsel holds autonomous authority to "
                        "settle regulatory investigations or commercial IP disputes up to EUR 350,000 per matter. "
                        "Any proposed settlement or consent decree exceeding EUR 350,000 requires formal "
                        "resolution from the Board Audit & Risk Committee."
                    ),
                ),
                PageSectionInput(
                    page_number=2,
                    section_heading="2. EU AI Act Supervisory Penalty Reserve (RESERVE-EUAI-2026)",
                    content=(
                        "PRIVILEGED SCHEDULE RESERVE-EUAI-2026: The corporation maintains a ring-fenced "
                        "regulatory contingency reserve of EUR 4.2 million under general ledger account "
                        "GL-9940-LIT to cover potential supervisory conformity audits or administrative "
                        "proceedings under high-risk AI system annexes."
                    ),
                ),
            ],
        ),
        RawCorpusDocument(
            doc_id="DOC-STALE-010",
            title="Legacy 2024 Cloud GPU Spot Instance & Unencrypted Model Bucket Guideline",
            allowed_roles=["public", "engineering", "procurement", "executive"],
            effective_date="2025-01-15",
            max_staleness_days=180,
            domain="legacy_cloud_ops",
            version="0.9",
            pages=[
                PageSectionInput(
                    page_number=1,
                    section_heading="1. Ad-Hoc Spot GPU Provisioning (OPS-GPU-2024-09)",
                    content=(
                        "LEGACY GUIDELINE OPS-GPU-2024-09: Research engineers may provision shared g5.12xlarge "
                        "spot GPU instances in AWS region us-east-1 up to USD 4,500 per month using departmental "
                        "corporate credit cards without prior FinOps architecture review."
                    ),
                ),
                PageSectionInput(
                    page_number=2,
                    section_heading="2. Temporary Scratch Bucket Checkpoints (OPS-S3-2024-11)",
                    content=(
                        "LEGACY GUIDELINE OPS-S3-2024-11: Experimental model weight checkpoints may be staged "
                        "in bucket s3://legacy-ml-scratch-2024 for up to 14 days with server-side KMS encryption "
                        "marked optional during benchmark debugging."
                    ),
                ),
            ],
        ),
        RawCorpusDocument(
            doc_id="DOC-STALE-011",
            title="Expired 2025 Third-Party Contractor VPN & Static SSH Key SOP",
            allowed_roles=["public", "engineering", "legal_compliance", "executive"],
            effective_date="2025-05-10",
            max_staleness_days=120,
            domain="legacy_network_sec",
            version="1.1",
            pages=[
                PageSectionInput(
                    page_number=1,
                    section_heading="1. Static RSA-2048 Bastion SSH Access (NET-VPN-2025-03)",
                    content=(
                        "EXPIRED SOP NET-VPN-2025-03: External engineering contractors are permitted to connect "
                        "to staging bastion hosts on TCP port 2222 using static RSA-2048 SSH keys rotated "
                        "manually every 90 days."
                    ),
                ),
                PageSectionInput(
                    page_number=2,
                    section_heading="2. Split-Tunnel Subnet Routing (PROFILE-EXT-LEGACY)",
                    content=(
                        "EXPIRED SOP PROFILE-EXT-LEGACY: Contractor VPN profile PROFILE-EXT-LEGACY grants "
                        "split-tunnel routing to internal staging subnet 10.44.0.0/16 upon email approval from "
                        "a project manager."
                    ),
                ),
            ],
        ),
        RawCorpusDocument(
            doc_id="DOC-FINOPS-012",
            title="Enterprise Token Quota, Caching & FinOps Cost Attribution Standard",
            allowed_roles=["public", "engineering", "procurement", "executive"],
            effective_date="2026-09-05",
            max_staleness_days=180,
            domain="finops_governance",
            version="2.2",
            pages=[
                PageSectionInput(
                    page_number=1,
                    section_heading="1. Daily Tenant Token Quotas & Semantic Cache TTL (FIN-QUOTA-312)",
                    content=(
                        "Standard FIN-QUOTA-312 enforces deterministic token governance across all internal "
                        "workspaces. Each business unit tenant is capped at 2,500,000 input tokens per day and "
                        "400,000 output tokens per day. Identical queries against an unchanged corpus SHA-256 "
                        "digest are served from a deterministic response cache with a 24-hour TTL."
                    ),
                ),
                PageSectionInput(
                    page_number=2,
                    section_heading="2. Budget Variance Throttling & Local CPU Fallback (CHARGEBACK-CC-77)",
                    content=(
                        "Under Cost Control CHARGEBACK-CC-77, cost centers that exceed their monthly LLM API "
                        "budget allocation by more than 15% are automatically throttled to quantized local CPU "
                        "models until a variance sign-off is approved by the VP of Finance."
                    ),
                ),
            ],
        ),
    ]


def get_canonical_eval_cases() -> list[EvalCase]:
    """Return the 50 labelled evaluation benchmark cases across 5 rigorous slices."""
    return [
        # =====================================================================
        # SLICE 1: 12 In-Scope Lexical Factual & Exact Identifier Queries
        # =====================================================================
        EvalCase(
            case_id="CASE-001",
            category=EvalCaseCategory.LEXICAL_FACTUAL,
            query="Under release gate GATE-AI-104, what minimum NDCG@10 score and RBAC Leakage Rate are required before production promotion?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-AI-SOP-001"],
            expected_chunk_ids=["DOC-AI-SOP-001#P1"],
            reference_answer="Under release gate GATE-AI-104, promotion requires a minimum NDCG@10 score of 0.82, a Passage Claim Support Rate of at least 0.85, and an RBAC Leakage Rate of 0.0 [DOC-AI-SOP-001#P1].",
            expected_key_terms=["GATE-AI-104", "0.82", "0.0"],
        ),
        EvalCase(
            case_id="CASE-002",
            category=EvalCaseCategory.LEXICAL_FACTUAL,
            query="What is the incident escalation code and response SLA when unsupported claims exceed 3.0% in 24 hours?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-AI-SOP-001"],
            expected_chunk_ids=["DOC-AI-SOP-001#P3"],
            reference_answer="If the unsupported claim rate exceeds 3.0% over a rolling 24-hour window, incident code INC-SEV2-RAG must be escalated to the AI Safety Lead within 4 hours [DOC-AI-SOP-001#P3].",
            expected_key_terms=["INC-SEV2-RAG", "4 hours", "90"],
        ),
        EvalCase(
            case_id="CASE-003",
            category=EvalCaseCategory.LEXICAL_FACTUAL,
            query="In specification SPEC-RET-208, what BM25 parameters and dense embedding dimension are mandated?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-RAG-ARCH-002"],
            expected_chunk_ids=["DOC-RAG-ARCH-002#P1"],
            reference_answer="Specification SPEC-RET-208 mandates SQLite FTS5 BM25 parameters k1=1.2 and b=0.75 paired with 384-dimensional L2-normalized dense embeddings [DOC-RAG-ARCH-002#P1].",
            expected_key_terms=["SPEC-RET-208", "1.2", "384"],
        ),
        EvalCase(
            case_id="CASE-004",
            category=EvalCaseCategory.LEXICAL_FACTUAL,
            query="What smoothing constant k and cross-encoder model are specified for Reciprocal Rank Fusion in DOC-RAG-ARCH-002?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-RAG-ARCH-002"],
            expected_chunk_ids=["DOC-RAG-ARCH-002#P2"],
            reference_answer="Reciprocal Rank Fusion uses smoothing constant k=60, and the top 12 fused candidates are rescored by cross-encoder/ms-marco-MiniLM-L-6-v2 [DOC-RAG-ARCH-002#P2].",
            expected_key_terms=["k=60", "ms-marco-MiniLM-L-6-v2", "12"],
        ),
        EvalCase(
            case_id="CASE-005",
            category=EvalCaseCategory.LEXICAL_FACTUAL,
            query="Under clause GDPR-ART28-ANNEX-B, which AWS region and breach notification SLA are required for subprocessors?",
            user_role=RoleEnum.LEGAL_COMPLIANCE,
            expected_doc_ids=["DOC-GDPR-003"],
            expected_chunk_ids=["DOC-GDPR-003#P1"],
            reference_answer="Clause GDPR-ART28-ANNEX-B requires EU/EEA data residency in eu-central-1 (Frankfurt), a signed DPA, and a 72-hour breach notification SLA [DOC-GDPR-003#P1].",
            expected_key_terms=["GDPR-ART28-ANNEX-B", "eu-central-1", "72-hour"],
        ),
        EvalCase(
            case_id="CASE-006",
            category=EvalCaseCategory.LEXICAL_FACTUAL,
            query="What is the purge deadline under workflow ERASURE-SLA-14D for GDPR Article 17 erasure requests?",
            user_role=RoleEnum.LEGAL_COMPLIANCE,
            expected_doc_ids=["DOC-GDPR-003"],
            expected_chunk_ids=["DOC-GDPR-003#P3"],
            reference_answer="Under workflow ERASURE-SLA-14D, FTS5 tokens, dense embeddings, and cached HITL records must be purged within 14 calendar days [DOC-GDPR-003#P3].",
            expected_key_terms=["ERASURE-SLA-14D", "14 calendar days", "Article 17"],
        ),
        EvalCase(
            case_id="CASE-007",
            category=EvalCaseCategory.LEXICAL_FACTUAL,
            query="According to schedule TIER-PROC-302, who must approve an AI contract with an ACV between EUR 50,001 and EUR 250,000?",
            user_role=RoleEnum.PROCUREMENT,
            expected_doc_ids=["DOC-PROC-004"],
            expected_chunk_ids=["DOC-PROC-004#P1"],
            reference_answer="Under schedule TIER-PROC-302, contracts between EUR 50,001 and EUR 250,000 ACV require joint sign-off from the VP of Procurement and the Data Protection Officer (DPO) [DOC-PROC-004#P1].",
            expected_key_terms=["TIER-PROC-302", "250,000", "DPO"],
        ),
        EvalCase(
            case_id="CASE-008",
            category=EvalCaseCategory.LEXICAL_FACTUAL,
            query="What does contractual clause CL-ZDR-09 require regarding foundation model training and buffer wiping?",
            user_role=RoleEnum.PROCUREMENT,
            expected_doc_ids=["DOC-PROC-004"],
            expected_chunk_ids=["DOC-PROC-004#P2"],
            reference_answer="Clause CL-ZDR-09 mandates Zero-Data-Retention: prompts and outputs are never used for model training, and inference buffers must be wiped within 24 hours [DOC-PROC-004#P2].",
            expected_key_terms=["CL-ZDR-09", "24 hours", "Zero-Data-Retention"],
        ),
        EvalCase(
            case_id="CASE-009",
            category=EvalCaseCategory.LEXICAL_FACTUAL,
            query="What does traceability control ISO-13485-CL-7.5.8 require when an underlying clinical SOP is updated?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-MED-005"],
            expected_chunk_ids=["DOC-MED-005#P2"],
            reference_answer="Control ISO-13485-CL-7.5.8 requires binding every clinical summary to source document ID, page number, effective date, and SHA-256 digest, automatically invalidating prior approvals when the SOP changes [DOC-MED-005#P2].",
            expected_key_terms=["ISO-13485-CL-7.5.8", "SHA-256", "invalidated"],
        ),
        EvalCase(
            case_id="CASE-010",
            category=EvalCaseCategory.LEXICAL_FACTUAL,
            query="Under control WAIVER-MAX-30D, what is the maximum duration of a temporary intake policy waiver?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-SEC-006"],
            expected_chunk_ids=["DOC-SEC-006#P3"],
            reference_answer="Under control WAIVER-MAX-30D, temporary intake waivers require a named approver and tracker reference and expire within 30 calendar days [DOC-SEC-006#P3].",
            expected_key_terms=["WAIVER-MAX-30D", "30 calendar days", "TOML"],
        ),
        EvalCase(
            case_id="CASE-011",
            category=EvalCaseCategory.LEXICAL_FACTUAL,
            query="What are the three conditions under control HITL-DIGEST-801 that route a RAG response to pending_review?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-HITL-007"],
            expected_chunk_ids=["DOC-HITL-007#P1"],
            reference_answer="Under HITL-DIGEST-801, responses route to pending_review when any passage is stale beyond max_staleness_days, passage claim support falls below 0.72, or the query touches regulated compliance boundaries [DOC-HITL-007#P1].",
            expected_key_terms=["HITL-DIGEST-801", "0.72", "pending_review"],
        ),
        EvalCase(
            case_id="CASE-012",
            category=EvalCaseCategory.LEXICAL_FACTUAL,
            query="Under standard FIN-QUOTA-312, what are the daily input and output token caps per business unit tenant?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-FINOPS-012"],
            expected_chunk_ids=["DOC-FINOPS-012#P1"],
            reference_answer="Standard FIN-QUOTA-312 caps each business unit tenant at 2,500,000 input tokens per day and 400,000 output tokens per day with a 24-hour semantic cache TTL [DOC-FINOPS-012#P1].",
            expected_key_terms=["FIN-QUOTA-312", "2,500,000", "400,000"],
        ),
        # =====================================================================
        # SLICE 2: 14 Semantic Paraphrase & Multi-Chunk Synthesis Queries
        # Phrased with natural operational synonyms (zero rare code leakage)
        # so pure lexical BM25 misses while Dense + RRF + Reranker excels!
        # =====================================================================
        EvalCase(
            case_id="CASE-013",
            category=EvalCaseCategory.SEMANTIC_MULTICHUNK,
            query="During a dark launch of a candidate language model in production, what statistical divergence threshold or response time slowdown triggers an automatic revert?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-AI-SOP-001"],
            expected_chunk_ids=["DOC-AI-SOP-001#P2"],
            reference_answer="During shadow deployment (canary silent parallel evaluation), a Population Stability Index (PSI) drift above 0.08 or CPU P99 latency above 450 milliseconds triggers an immediate circuit-breaker rollback [DOC-AI-SOP-001#P2].",
            expected_key_terms=["0.08", "450", "PSI"],
        ),
        EvalCase(
            case_id="CASE-014",
            category=EvalCaseCategory.SEMANTIC_MULTICHUNK,
            query="Why does our enterprise architecture require evaluating user role clearance at index query time rather than redacting confidential documents after final results are ranked?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-RAG-ARCH-002"],
            expected_chunk_ids=["DOC-RAG-ARCH-002#P3"],
            reference_answer="Pre-retrieval filtering (AC-PRE-01) is required because post-retrieval filtering lets unauthorized chunks displace authorized evidence from the top-k window and creates side-channel inference risks [DOC-RAG-ARCH-002#P3].",
            expected_key_terms=["AC-PRE-01", "displace", "pre-retrieval"],
        ),
        EvalCase(
            case_id="CASE-015",
            category=EvalCaseCategory.SEMANTIC_MULTICHUNK,
            query="How must European citizen fiscal IDs, medical coverage records, and wire transfer routing strings be anonymized prior to generating production vector representations?",
            user_role=RoleEnum.LEGAL_COMPLIANCE,
            expected_doc_ids=["DOC-GDPR-003"],
            expected_chunk_ids=["DOC-GDPR-003#P2"],
            reference_answer="German tax identifiers (Steuer-ID), statutory health insurance numbers (KVNR), and IBAN bank numbers must be checksum-verified and replaced with salted SHA-256 pseudonyms before indexing [DOC-GDPR-003#P2].",
            expected_key_terms=["Steuer-ID", "KVNR", "IBAN"],
        ),
        EvalCase(
            case_id="CASE-016",
            category=EvalCaseCategory.SEMANTIC_MULTICHUNK,
            query="What open-source dependency manifest specification and maximum vulnerability severity score must a cloud vendor satisfy before contract approval?",
            user_role=RoleEnum.PROCUREMENT,
            expected_doc_ids=["DOC-PROC-004"],
            expected_chunk_ids=["DOC-PROC-004#P3"],
            reference_answer="Vendors must provide a SOC 2 Type II or ISO/IEC 27001:2022 attestation and a CycloneDX 1.6 Software Bill of Materials (SBOM), with zero unresolved critical CVEs (CVSS >= 9.0) [DOC-PROC-004#P3].",
            expected_key_terms=["CycloneDX 1.6", "SBOM", "9.0"],
        ),
        EvalCase(
            case_id="CASE-017",
            category=EvalCaseCategory.SEMANTIC_MULTICHUNK,
            query="Can an enterprise neural assistant deployed in a radiology department independently prescribe patient therapy, and what physician approval rule applies?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-MED-005"],
            expected_chunk_ids=["DOC-MED-005#P1"],
            reference_answer="Under directive MED-MDR-704, clinical AI tools are strictly documentation and evidence-lookup aids and must never issue autonomous diagnostic or treatment orders; licensed clinician sign-off is mandatory for Class IIa and IIb workflows [DOC-MED-005#P1].",
            expected_key_terms=["MED-MDR-704", "clinician", "Class IIa"],
        ),
        EvalCase(
            case_id="CASE-018",
            category=EvalCaseCategory.SEMANTIC_MULTICHUNK,
            query="At what recurring interval must deployed hospital software undergo regulatory field quality reviews, and what factual grounding floor is required?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-MED-005"],
            expected_chunk_ids=["DOC-MED-005#P3"],
            reference_answer="Post-Market Surveillance (PMS) audits run every 90 days and require a minimum NLI passage entailment score of 0.78 on every atomic sentence [DOC-MED-005#P3].",
            expected_key_terms=["90 days", "0.78", "Post-Market Surveillance"],
        ),
        EvalCase(
            case_id="CASE-019",
            category=EvalCaseCategory.SEMANTIC_MULTICHUNK,
            query="Which cross-platform desktop wrapper runtime settings or executable code-signing certificate changes cause a package release to be blocked?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-SEC-006"],
            expected_chunk_ids=["DOC-SEC-006#P2"],
            reference_answer="Under RULE-ELEC-04, enabling RunAsNode or EnableNodeCliInspectArguments, setting webSecurity to false, stripping Windows Authenticode signatures, or changing the publisher subject triggers rejection or security review [DOC-SEC-006#P2].",
            expected_key_terms=["RULE-ELEC-04", "RunAsNode", "Authenticode"],
        ),
        EvalCase(
            case_id="CASE-020",
            category=EvalCaseCategory.SEMANTIC_MULTICHUNK,
            query="Which language compiler artifact formats and Java WAR or JAR recursion depths are supported when extracting software supply chain inventories?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-SEC-006"],
            expected_chunk_ids=["DOC-SEC-006#P1"],
            reference_answer="Standard SEC-GATE-511 inspects Go buildinfo, Rust cargo-auditable sections, Electron ASAR bundles, NuGet deps.json, Python dist-info, and nested Maven pom.properties up to 2 archive levels deep [DOC-SEC-006#P1].",
            expected_key_terms=["SEC-GATE-511", "cargo-auditable", "pom.properties"],
        ),
        EvalCase(
            case_id="CASE-021",
            category=EvalCaseCategory.SEMANTIC_MULTICHUNK,
            query="How does the compliance workflow guarantee that a previously authorized answer is automatically revoked if a source policy page is revised?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-HITL-007"],
            expected_chunk_ids=["DOC-HITL-007#P2"],
            reference_answer="Approvals are bound to SHA256(normalized_query || sorted_chunk_sha256_digests || answer_sha256); any upstream edit alters the chunk SHA-256 and transitions the status to invalidated_by_upstream_edit [DOC-HITL-007#P2].",
            expected_key_terms=["SHA256", "invalidated_by_upstream_edit", "sorted_chunk_sha256_digests"],
        ),
        EvalCase(
            case_id="CASE-022",
            category=EvalCaseCategory.SEMANTIC_MULTICHUNK,
            query="What automated rate-limiting penalty is triggered when an engineering division overspends its cloud foundation model quota by fifteen percent?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-FINOPS-012"],
            expected_chunk_ids=["DOC-FINOPS-012#P2"],
            reference_answer="Under CHARGEBACK-CC-77, cost centers exceeding monthly LLM budgets by over 15% are automatically throttled to quantized local CPU models until VP of Finance variance sign-off [DOC-FINOPS-012#P2].",
            expected_key_terms=["CHARGEBACK-CC-77", "15%", "VP of Finance"],
        ),
        EvalCase(
            case_id="CASE-023",
            category=EvalCaseCategory.SEMANTIC_MULTICHUNK,
            query="How does the retrieval pipeline merge sparse keyword hits and dense embedding matches when their raw similarity metrics have incompatible scales?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-RAG-ARCH-002"],
            expected_chunk_ids=["DOC-RAG-ARCH-002#P2"],
            reference_answer="Unbounded BM25 scores and bounded cosine similarities are combined via Reciprocal Rank Fusion (k=60) over ranks before cross-encoder reranking of the top 12 candidates [DOC-RAG-ARCH-002#P2].",
            expected_key_terms=["Reciprocal Rank Fusion", "k=60", "top 12"],
        ),
        EvalCase(
            case_id="CASE-024",
            category=EvalCaseCategory.SEMANTIC_MULTICHUNK,
            query="For an executive reviewing senior level-seven technical leadership packages, what is the yearly equity grant limit and corporate buyout cliff policy?",
            user_role=RoleEnum.EXECUTIVE,
            expected_doc_ids=["DOC-HR-008"],
            expected_chunk_ids=["DOC-HR-008#P1"],
            reference_answer="Under HR-COMP-901, Band L7 Principal and Staff AI Engineers have an annual RSU refresh ceiling of EUR 85,000 vesting over 4 years, with 12-month double-trigger M&A acceleration and a 35% retention bonus [DOC-HR-008#P1].",
            expected_key_terms=["HR-COMP-901", "85,000", "double-trigger"],
        ),
        EvalCase(
            case_id="CASE-025",
            category=EvalCaseCategory.SEMANTIC_MULTICHUNK,
            query="When an executive evaluates workforce downsizing separation benefits, how many weeks or months of pay per year of tenure and what job placement allowance apply?",
            user_role=RoleEnum.EXECUTIVE,
            expected_doc_ids=["DOC-HR-008"],
            expected_chunk_ids=["DOC-HR-008#P2"],
            reference_answer="Schedule HR-SEV-902 provides 1.5 months of base salary per completed year of service (capped at 18 months) plus a EUR 6,000 outplacement transition stipend [DOC-HR-008#P2].",
            expected_key_terms=["HR-SEV-902", "1.5 months", "18 months"],
        ),
        EvalCase(
            case_id="CASE-026",
            category=EvalCaseCategory.SEMANTIC_MULTICHUNK,
            query="What is the chief legal officer's unilateral dispute resolution spending limit and how much capital is held in escrow for European artificial intelligence compliance fines?",
            user_role=RoleEnum.LEGAL_COMPLIANCE,
            expected_doc_ids=["DOC-LEGAL-009"],
            expected_chunk_ids=["DOC-LEGAL-009#P1", "DOC-LEGAL-009#P2"],
            reference_answer="Under LEG-RES-604, the General Counsel has autonomous settlement authority up to EUR 350,000 [DOC-LEGAL-009#P1], and RESERVE-EUAI-2026 holds EUR 4.2 million in account GL-9940-LIT for EU AI Act audits [DOC-LEGAL-009#P2].",
            expected_key_terms=["350,000", "4.2 million", "GL-9940-LIT"],
        ),
        # =====================================================================
        # SLICE 3: 8 Pre-Retrieval RBAC-Restricted Queries (Zero Leakage Gate)
        # Unauthorized roles query restricted documents -> must block pre-retrieval
        # with 0.0 leakage and abstain cleanly!
        # =====================================================================
        EvalCase(
            case_id="CASE-027",
            category=EvalCaseCategory.RBAC_RESTRICTED,
            query="What is the Band L7 Principal AI Engineer annual RSU equity ceiling and M&A retention bonus under HR-COMP-901?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            forbidden_doc_ids=["DOC-HR-008"],
            expect_abstention=True,
            expect_hitl_flag=True,
            reference_answer="ABSTAIN: Role 'engineering' lacks pre-retrieval RBAC clearance for restricted HR compensation policy DOC-HR-008.",
        ),
        EvalCase(
            case_id="CASE-028",
            category=EvalCaseCategory.RBAC_RESTRICTED,
            query="Under HR-SEV-902, how many months of severance pay and what outplacement stipend do laid-off employees receive?",
            user_role=RoleEnum.PUBLIC,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            forbidden_doc_ids=["DOC-HR-008"],
            expect_abstention=True,
            expect_hitl_flag=True,
            reference_answer="ABSTAIN: Role 'public' lacks pre-retrieval RBAC clearance for restricted HR severance policy DOC-HR-008.",
        ),
        EvalCase(
            case_id="CASE-029",
            category=EvalCaseCategory.RBAC_RESTRICTED,
            query="What is the General Counsel autonomous settlement limit under protocol LEG-RES-604?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            forbidden_doc_ids=["DOC-LEGAL-009"],
            expect_abstention=True,
            expect_hitl_flag=True,
            reference_answer="ABSTAIN: Role 'engineering' lacks pre-retrieval RBAC clearance for privileged legal protocol DOC-LEGAL-009.",
        ),
        EvalCase(
            case_id="CASE-030",
            category=EvalCaseCategory.RBAC_RESTRICTED,
            query="How much money is held in ledger account GL-9940-LIT under RESERVE-EUAI-2026 for EU AI Act supervisory penalties?",
            user_role=RoleEnum.PROCUREMENT,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            forbidden_doc_ids=["DOC-LEGAL-009"],
            expect_abstention=True,
            expect_hitl_flag=True,
            reference_answer="ABSTAIN: Role 'procurement' lacks pre-retrieval RBAC clearance for privileged legal reserve DOC-LEGAL-009.",
        ),
        EvalCase(
            case_id="CASE-031",
            category=EvalCaseCategory.RBAC_RESTRICTED,
            query="What are the ACV commercial approval tiers in schedule TIER-PROC-302 and the zero-data-retention clause CL-ZDR-09?",
            user_role=RoleEnum.PUBLIC,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            forbidden_doc_ids=["DOC-PROC-004"],
            expect_abstention=True,
            expect_hitl_flag=True,
            reference_answer="ABSTAIN: Role 'public' lacks pre-retrieval RBAC clearance for internal procurement policy DOC-PROC-004.",
        ),
        EvalCase(
            case_id="CASE-032",
            category=EvalCaseCategory.RBAC_RESTRICTED,
            query="What are the ACV commercial approval tiers in schedule TIER-PROC-302 for AI vendor contracts?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            forbidden_doc_ids=["DOC-PROC-004"],
            expect_abstention=True,
            expect_hitl_flag=True,
            reference_answer="ABSTAIN: Role 'engineering' lacks pre-retrieval RBAC clearance for procurement policy DOC-PROC-004.",
        ),
        EvalCase(
            case_id="CASE-033",
            category=EvalCaseCategory.RBAC_RESTRICTED,
            query="Under GDPR-ART28-ANNEX-B and ERASURE-SLA-14D, what are the subprocessor Frankfurt residency and 14-day erasure rules?",
            user_role=RoleEnum.PUBLIC,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            forbidden_doc_ids=["DOC-GDPR-003"],
            expect_abstention=True,
            expect_hitl_flag=True,
            reference_answer="ABSTAIN: Role 'public' lacks pre-retrieval RBAC clearance for internal GDPR compliance standard DOC-GDPR-003.",
        ),
        EvalCase(
            case_id="CASE-034",
            category=EvalCaseCategory.RBAC_RESTRICTED,
            query="What does clinical directive MED-MDR-704 and ISO-13485-CL-7.5.8 require for Class IIb medical device software?",
            user_role=RoleEnum.PUBLIC,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            forbidden_doc_ids=["DOC-MED-005"],
            expect_abstention=True,
            expect_hitl_flag=True,
            reference_answer="ABSTAIN: Role 'public' lacks pre-retrieval RBAC clearance for regulated clinical governance policy DOC-MED-005.",
        ),
        # =====================================================================
        # SLICE 4: 8 Stale-Source Governance Queries (Freshness Gate)
        # Queries matching expired/stale policies -> retrieved for provenance
        # inspection AND flagged stale + routed to HITL review!
        # =====================================================================
        EvalCase(
            case_id="CASE-035",
            category=EvalCaseCategory.STALE_SOURCE,
            query="Under guideline OPS-GPU-2024-09, can engineers rent g5.12xlarge spot GPU nodes in us-east-1 up to USD 4,500/month on a corporate card?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-STALE-010"],
            expected_chunk_ids=["DOC-STALE-010#P1"],
            expect_stale_flag=True,
            expect_hitl_flag=True,
            reference_answer="STALE POLICY WARNING: Guideline OPS-GPU-2024-09 (effective 2025-01-15, max staleness 180 days) is expired; it previously described provisioning g5.12xlarge spot instances in us-east-1 up to USD 4,500/month [DOC-STALE-010#P1].",
            expected_key_terms=["OPS-GPU-2024-09", "g5.12xlarge", "4,500"],
        ),
        EvalCase(
            case_id="CASE-036",
            category=EvalCaseCategory.STALE_SOURCE,
            query="Does guideline OPS-S3-2024-11 allow storing temporary model checkpoints in s3://legacy-ml-scratch-2024 without KMS encryption?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-STALE-010"],
            expected_chunk_ids=["DOC-STALE-010#P2"],
            expect_stale_flag=True,
            expect_hitl_flag=True,
            reference_answer="STALE POLICY WARNING: Guideline OPS-S3-2024-11 (effective 2025-01-15, max staleness 180 days) is expired; it referenced staging checkpoints in s3://legacy-ml-scratch-2024 for 14 days [DOC-STALE-010#P2].",
            expected_key_terms=["OPS-S3-2024-11", "s3://legacy-ml-scratch-2024", "14 days"],
        ),
        EvalCase(
            case_id="CASE-037",
            category=EvalCaseCategory.STALE_SOURCE,
            query="According to NET-VPN-2025-03, can external contractors connect to staging bastion hosts on port 2222 with static RSA-2048 SSH keys?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-STALE-011"],
            expected_chunk_ids=["DOC-STALE-011#P1"],
            expect_stale_flag=True,
            expect_hitl_flag=True,
            reference_answer="STALE POLICY WARNING: SOP NET-VPN-2025-03 (effective 2025-05-10, max staleness 120 days) is expired; it previously permitted bastion connections on TCP port 2222 using static RSA-2048 keys rotated every 90 days [DOC-STALE-011#P1].",
            expected_key_terms=["NET-VPN-2025-03", "2222", "RSA-2048"],
        ),
        EvalCase(
            case_id="CASE-038",
            category=EvalCaseCategory.STALE_SOURCE,
            query="What subnet access does contractor VPN profile PROFILE-EXT-LEGACY grant upon manager email approval?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=["DOC-STALE-011"],
            expected_chunk_ids=["DOC-STALE-011#P2"],
            expect_stale_flag=True,
            expect_hitl_flag=True,
            reference_answer="STALE POLICY WARNING: SOP PROFILE-EXT-LEGACY (effective 2025-05-10, max staleness 120 days) is expired; it referenced split-tunnel routing to staging subnet 10.44.0.0/16 [DOC-STALE-011#P2].",
            expected_key_terms=["PROFILE-EXT-LEGACY", "10.44.0.0/16", "split-tunnel"],
        ),
        EvalCase(
            case_id="CASE-039",
            category=EvalCaseCategory.STALE_SOURCE,
            query="Can research teams use credit cards to launch unreviewed spot GPU machines in Northern Virginia under legacy cloud rules?",
            user_role=RoleEnum.PROCUREMENT,
            expected_doc_ids=["DOC-STALE-010"],
            expected_chunk_ids=["DOC-STALE-010#P1"],
            expect_stale_flag=True,
            expect_hitl_flag=True,
            reference_answer="STALE POLICY WARNING: Legacy guideline OPS-GPU-2024-09 is expired beyond its 180-day freshness window; it previously covered g5.12xlarge spot instances in us-east-1 up to USD 4,500 per month [DOC-STALE-010#P1].",
            expected_key_terms=["OPS-GPU-2024-09", "us-east-1", "4,500"],
        ),
        EvalCase(
            case_id="CASE-040",
            category=EvalCaseCategory.STALE_SOURCE,
            query="Are third-party developers allowed to use 90-day static SSH keys on port 2222 for staging bastion access?",
            user_role=RoleEnum.LEGAL_COMPLIANCE,
            expected_doc_ids=["DOC-STALE-011"],
            expected_chunk_ids=["DOC-STALE-011#P1"],
            expect_stale_flag=True,
            expect_hitl_flag=True,
            reference_answer="STALE POLICY WARNING: Expired SOP NET-VPN-2025-03 exceeds its 120-day max staleness limit; it previously described static RSA-2048 SSH keys on port 2222 [DOC-STALE-011#P1].",
            expected_key_terms=["NET-VPN-2025-03", "2222", "90 days"],
        ),
        EvalCase(
            case_id="CASE-041",
            category=EvalCaseCategory.STALE_SOURCE,
            query="As of mid-2027, is clinical AI governance directive MED-MDR-704 still within its allowed freshness window without recertification?",
            user_role=RoleEnum.ENGINEERING,
            as_of_date="2027-06-15",
            expected_doc_ids=["DOC-MED-005"],
            expected_chunk_ids=["DOC-MED-005#P1"],
            expect_stale_flag=True,
            expect_hitl_flag=True,
            reference_answer="STALE POLICY WARNING: As of 2027-06-15, directive MED-MDR-704 (effective 2026-08-01, max staleness 150 days) has exceeded its freshness window and requires recertification [DOC-MED-005#P1].",
            expected_key_terms=["MED-MDR-704", "Class IIa", "clinician"],
        ),
        EvalCase(
            case_id="CASE-042",
            category=EvalCaseCategory.STALE_SOURCE,
            query="As of June 2027, what are the procurement approval thresholds in schedule TIER-PROC-302 if the policy has not been renewed since June 2026?",
            user_role=RoleEnum.PROCUREMENT,
            as_of_date="2027-06-15",
            expected_doc_ids=["DOC-PROC-004"],
            expected_chunk_ids=["DOC-PROC-004#P1"],
            expect_stale_flag=True,
            expect_hitl_flag=True,
            reference_answer="STALE POLICY WARNING: As of 2027-06-15, schedule TIER-PROC-302 (effective 2026-06-20, max staleness 180 days) is stale; it specifies ACV tiers of EUR 50,000 and EUR 250,000 [DOC-PROC-004#P1].",
            expected_key_terms=["TIER-PROC-302", "50,000", "250,000"],
        ),
        # =====================================================================
        # SLICE 5: 8 Unanswerable Out-of-Scope Queries (Abstention Gate)
        # =====================================================================
        EvalCase(
            case_id="CASE-043",
            category=EvalCaseCategory.UNANSWERABLE_ABSTENTION,
            query="What is the daily vegan lunch subsidy amount in the Erlangen campus cafeteria and what time does the coffee bar close?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            expect_abstention=True,
            reference_answer="ABSTAIN: No authorized enterprise governance passage provides evidence for campus cafeteria subsidies or operating hours.",
        ),
        EvalCase(
            case_id="CASE-044",
            category=EvalCaseCategory.UNANSWERABLE_ABSTENTION,
            query="What is the departure timetable for the inter-campus hydrogen shuttle bus between Nuremberg and Munich?",
            user_role=RoleEnum.PUBLIC,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            expect_abstention=True,
            reference_answer="ABSTAIN: No authorized enterprise governance passage covers inter-campus shuttle bus timetables.",
        ),
        EvalCase(
            case_id="CASE-045",
            category=EvalCaseCategory.UNANSWERABLE_ABSTENTION,
            query="How much cash bonus is awarded to employees when the European Patent Office grants a semiconductor hardware patent?",
            user_role=RoleEnum.EXECUTIVE,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            expect_abstention=True,
            reference_answer="ABSTAIN: No enterprise governance passage in the corpus specifies patent grant cash bonuses.",
        ),
        EvalCase(
            case_id="CASE-046",
            category=EvalCaseCategory.UNANSWERABLE_ABSTENTION,
            query="What is the minimum flight duration in hours required to book business class airfare for international sales conferences?",
            user_role=RoleEnum.PROCUREMENT,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            expect_abstention=True,
            reference_answer="ABSTAIN: Corporate travel and airfare cabin class policies are not present in the governance corpus.",
        ),
        EvalCase(
            case_id="CASE-047",
            category=EvalCaseCategory.UNANSWERABLE_ABSTENTION,
            query="Which D-Wave quantum annealing cryostat lease vendor is approved for superconducting qubit calibration?",
            user_role=RoleEnum.ENGINEERING,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            expect_abstention=True,
            reference_answer="ABSTAIN: No governance document in the corpus addresses quantum annealing hardware or cryostat vendors.",
        ),
        EvalCase(
            case_id="CASE-048",
            category=EvalCaseCategory.UNANSWERABLE_ABSTENTION,
            query="What is the annual wellness reimbursement limit for ergonomic standing desks and fitness center memberships?",
            user_role=RoleEnum.HR_RESTRICTED,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            expect_abstention=True,
            reference_answer="ABSTAIN: Wellness, gym membership, and home-office furniture reimbursements are outside the indexed governance corpus.",
        ),
        EvalCase(
            case_id="CASE-049",
            category=EvalCaseCategory.UNANSWERABLE_ABSTENTION,
            query="What temperature and humidity settings must be maintained in the cold-storage biological tissue biobank freezer room?",
            user_role=RoleEnum.LEGAL_COMPLIANCE,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            expect_abstention=True,
            reference_answer="ABSTAIN: Physical biobank freezer temperature and humidity controls are not covered in the software and AI governance corpus.",
        ),
        EvalCase(
            case_id="CASE-050",
            category=EvalCaseCategory.UNANSWERABLE_ABSTENTION,
            query="How many paid sabbatical weeks are granted to employees celebrating their tenth work anniversary?",
            user_role=RoleEnum.EXECUTIVE,
            expected_doc_ids=[],
            expected_chunk_ids=[],
            expect_abstention=True,
            reference_answer="ABSTAIN: Sabbatical leave policies for work anniversaries are not present in the indexed governance corpus.",
        ),
    ]


def materialize_corpus_and_benchmark(root_dir: Path) -> tuple[Path, Path]:
    """Write the 12 corpus JSON documents and 50-case eval_cases.json deterministically to disk."""
    corpus_dir = root_dir / "data" / "corpus"
    benchmark_dir = root_dir / "data" / "benchmark"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    docs = get_canonical_corpus_documents()
    for doc in docs:
        doc_path = corpus_dir / f"{doc.doc_id}.json"
        doc_path.write_text(
            json.dumps(doc.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    dataset = EvalDatasetFile(
        dataset_name="enterprise-hybrid-rag-workbench-50case-benchmark",
        schema_version="1.0.0",
        default_as_of_date="2026-10-27",
        description=(
            "50 labelled enterprise RAG evaluation cases across 5 slices: "
            "12 lexical_factual, 14 semantic_multichunk, 8 rbac_restricted, "
            "8 stale_source, and 8 unanswerable_abstention."
        ),
        cases=get_canonical_eval_cases(),
    )
    eval_path = benchmark_dir / "eval_cases.json"
    eval_path.write_text(
        json.dumps(dataset.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return corpus_dir, eval_path
