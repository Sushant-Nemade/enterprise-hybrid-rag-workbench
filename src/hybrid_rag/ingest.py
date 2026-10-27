"""Page-aware document ingestion and deterministic SHA-256 provenance hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from hybrid_rag.corpus_defaults import (
    get_canonical_corpus_documents,
    get_canonical_eval_cases,
)
from hybrid_rag.schemas import (
    DocumentChunk,
    DocumentMetadata,
    EvalCase,
    EvalDatasetFile,
    RawCorpusDocument,
)


def sha256_hex(text: str | bytes) -> str:
    """Compute a lowercase hex SHA-256 digest."""
    if isinstance(text, str):
        payload = text.encode("utf-8")
    else:
        payload = text
    return hashlib.sha256(payload).hexdigest()


def compute_document_sha256(raw_doc: RawCorpusDocument) -> str:
    """Compute deterministic SHA-256 digest over a raw governance document."""
    canonical = json.dumps(
        raw_doc.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256_hex(canonical)


def compute_chunk_sha256(
    doc_id: str,
    page_number: int,
    section_heading: str,
    content: str,
    allowed_roles: list[str],
    effective_date: str,
    max_staleness_days: int,
) -> str:
    """Compute deterministic SHA-256 digest over a page-aware chunk and its governance header."""
    roles_part = ",".join(sorted(r.strip().lower() for r in allowed_roles))
    canonical = (
        f"doc_id={doc_id}|page={page_number}|heading={section_heading.strip()}|"
        f"roles={roles_part}|effective={effective_date}|max_staleness={max_staleness_days}|"
        f"text={content.strip()}"
    )
    return sha256_hex(canonical)


@dataclass(frozen=True)
class IngestedCorpus:
    """Complete ingested enterprise corpus with document metadata, chunks, and manifest digest."""

    documents: list[DocumentMetadata]
    chunks: list[DocumentChunk]
    corpus_sha256: str
    chunks_by_id: dict[str, DocumentChunk]
    docs_by_id: dict[str, DocumentMetadata]


def ingest_raw_documents(raw_docs: list[RawCorpusDocument]) -> IngestedCorpus:
    """Convert raw governance documents into page-aware chunks with SHA-256 provenance."""
    sorted_raw = sorted(raw_docs, key=lambda d: d.doc_id)
    documents: list[DocumentMetadata] = []
    chunks: list[DocumentChunk] = []

    for raw_doc in sorted_raw:
        doc_digest = compute_document_sha256(raw_doc)
        doc_meta = DocumentMetadata(
            doc_id=raw_doc.doc_id,
            title=raw_doc.title,
            sha256_digest=doc_digest,
            allowed_roles=raw_doc.allowed_roles,
            effective_date=raw_doc.effective_date,
            max_staleness_days=raw_doc.max_staleness_days,
            domain=raw_doc.domain,
            version=raw_doc.version,
        )
        documents.append(doc_meta)

        for page in sorted(raw_doc.pages, key=lambda p: p.page_number):
            chunk_id = f"{raw_doc.doc_id}#P{page.page_number}"
            normalized_text = page.content.strip()
            chunk_digest = compute_chunk_sha256(
                doc_id=raw_doc.doc_id,
                page_number=page.page_number,
                section_heading=page.section_heading,
                content=normalized_text,
                allowed_roles=doc_meta.allowed_roles,
                effective_date=doc_meta.effective_date,
                max_staleness_days=doc_meta.max_staleness_days,
            )
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=raw_doc.doc_id,
                title=raw_doc.title,
                page_number=page.page_number,
                section_heading=page.section_heading.strip(),
                text=normalized_text,
                sha256_digest=chunk_digest,
                doc_sha256_digest=doc_digest,
                allowed_roles=doc_meta.allowed_roles,
                effective_date=doc_meta.effective_date,
                max_staleness_days=doc_meta.max_staleness_days,
                domain=doc_meta.domain,
            )
            chunks.append(chunk)

    manifest_lines = [f"{c.chunk_id}:{c.sha256_digest}" for c in chunks]
    corpus_sha256 = sha256_hex("\n".join(manifest_lines))
    return IngestedCorpus(
        documents=documents,
        chunks=chunks,
        corpus_sha256=corpus_sha256,
        chunks_by_id={c.chunk_id: c for c in chunks},
        docs_by_id={d.doc_id: d for d in documents},
    )


def load_corpus_from_directory(corpus_dir: Path | None = None) -> IngestedCorpus:
    """Load enterprise governance documents from data/corpus/*.json or canonical defaults."""
    if corpus_dir is not None and corpus_dir.is_dir():
        json_files = sorted(corpus_dir.glob("*.json"))
        if json_files:
            raw_docs = [
                RawCorpusDocument.model_validate_json(path.read_text(encoding="utf-8"))
                for path in json_files
            ]
            return ingest_raw_documents(raw_docs)
    return ingest_raw_documents(get_canonical_corpus_documents())


def load_eval_cases(eval_path: Path | None = None) -> tuple[list[EvalCase], str]:
    """Load the 50 labelled evaluation cases and compute the dataset SHA-256 digest."""
    if eval_path is not None and eval_path.is_file():
        raw_bytes = eval_path.read_bytes()
        parsed = EvalDatasetFile.model_validate_json(raw_bytes)
        canonical_json = json.dumps(
            parsed.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return parsed.cases, sha256_hex(canonical_json)

    cases = get_canonical_eval_cases()
    dataset = EvalDatasetFile(
        dataset_name="enterprise-hybrid-rag-workbench-50case-benchmark",
        schema_version="1.0.0",
        default_as_of_date="2026-10-27",
        description=(
            "50 labelled enterprise RAG evaluation cases across 5 slices: "
            "12 lexical_factual, 14 semantic_multichunk, 8 rbac_restricted, "
            "8 stale_source, and 8 unanswerable_abstention."
        ),
        cases=cases,
    )
    canonical_json = json.dumps(
        dataset.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return cases, sha256_hex(canonical_json)
