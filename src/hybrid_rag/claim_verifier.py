"""Atomic Claim Decomposition, Exact Citation Verification, and Local NLI Passage Entailment."""

from __future__ import annotations

import re

from hybrid_rag.onnx_models import TOKEN_PATTERN, LocalONNXModelSuite, load_model_suite
from hybrid_rag.schemas import AtomicClaimSupport, ScoredChunk

CITATION_TAG_RE = re.compile(r"\[([A-Za-z0-9_\-]+#P\d+(?:\s*,\s*[A-Za-z0-9_\-]+#P\d+)*)\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")


def decompose_into_atomic_claims(answer_text: str) -> list[str]:
    """Decompose a generated answer into individual atomic claims/sentences."""
    cleaned = answer_text.strip()
    if not cleaned:
        return []
    raw_sentences = SENTENCE_SPLIT_RE.split(cleaned)
    claims: list[str] = []
    for sent in raw_sentences:
        s = sent.strip()
        if len(s) >= 5:
            claims.append(s)
    return claims if claims else [cleaned]


def extract_citation_ids(claim_text: str) -> list[str]:
    """Extract cited chunk IDs such as [DOC-AI-SOP-001#P1] from a claim."""
    found: list[str] = []
    for match in CITATION_TAG_RE.findall(claim_text):
        for part in match.split(","):
            cid = part.strip()
            if cid and cid not in found:
                found.append(cid)
    return found


def strip_citation_tags(claim_text: str) -> str:
    """Remove bracketed citation tags and governance prefixes for clean NLI hypothesis scoring."""
    without_tags = CITATION_TAG_RE.sub("", claim_text)
    without_prefix = re.sub(r"^STALE POLICY WARNING:\s*", "", without_tags, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", without_prefix).strip()


def synthesize_grounded_answer(
    query: str,
    retrieved_chunks: list[ScoredChunk],
    should_abstain: bool,
    abstention_reason: str | None = None,
) -> str:
    """Synthesize a deterministic, citation-anchored response from retrieved authorized chunks."""
    if should_abstain or not retrieved_chunks:
        reason = abstention_reason or "No authorized passage meets the minimum evidence threshold."
        return f"ABSTAIN: {reason}"

    top_chunk = retrieved_chunks[0]
    stale_prefix = ""
    if top_chunk.is_stale:
        stale_prefix = (
            f"STALE POLICY WARNING: Document {top_chunk.doc_id} (effective {top_chunk.effective_date}, "
            f"max_staleness_days={top_chunk.max_staleness_days}, {top_chunk.days_overdue} days overdue) "
            f"requires governance recertification [{top_chunk.chunk_id}]. "
        )

    # Select the most query-relevant sentences from the top chunk (and second chunk if multi-chunk)
    q_tokens = {
        t.lower()
        for t in TOKEN_PATTERN.findall(query.lower())
        if len(t) > 2
    }
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(top_chunk.text.strip()) if s.strip()]
    scored_sents: list[tuple[float, int, str]] = []
    for idx, sent in enumerate(sentences):
        s_tokens = {t.lower() for t in TOKEN_PATTERN.findall(sent.lower())}
        overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
        has_digits = 0.25 if re.search(r"\d", sent) else 0.0
        scored_sents.append((overlap + has_digits, idx, sent))

    scored_sents.sort(key=lambda item: (-item[0], item[1]))
    # Include all sentences from top chunk (each explicitly cited with [chunk_id])
    cited_sentences = [
        f"{item[2].rstrip('.')} [{top_chunk.chunk_id}]."
        for item in sorted(scored_sents[:3], key=lambda x: x[1])
    ]
    primary_claim = " ".join(cited_sentences)

    # Check if second chunk is from the same document and contributes complementary high-confidence facts
    secondary_claim = ""
    if (
        len(retrieved_chunks) >= 2
        and retrieved_chunks[1].doc_id == top_chunk.doc_id
        and retrieved_chunks[1].chunk_id != top_chunk.chunk_id
        and (
            retrieved_chunks[1].dense_cosine_score >= 0.35
            or (retrieved_chunks[1].reranker_score or 0.0) >= 0.45
        )
    ):
        second_chunk = retrieved_chunks[1]
        sec_sents = [s.strip() for s in SENTENCE_SPLIT_RE.split(second_chunk.text.strip()) if s.strip()]
        if sec_sents:
            best_sec = max(
                sec_sents,
                key=lambda s: len(q_tokens & {t.lower() for t in TOKEN_PATTERN.findall(s.lower())})
                + (0.5 if re.search(r"\d", s) else 0.0),
            ).rstrip(".")
            secondary_claim = f" {best_sec} [{second_chunk.chunk_id}]."

    return f"{stale_prefix}{primary_claim}{secondary_claim}".strip()


def verify_answer_claims(
    answer_text: str,
    retrieved_chunks: list[ScoredChunk],
    models: LocalONNXModelSuite | None = None,
    expected_key_terms: list[str] | None = None,
) -> tuple[list[AtomicClaimSupport], float]:
    """Decompose answer into atomic claims and verify passage-level NLI support + exact citations."""
    suite = models or load_model_suite()
    if answer_text.strip().upper().startswith("ABSTAIN:"):
        # If the case had expected ground-truth factual key terms, abstaining fails to support the answer
        if expected_key_terms:
            return [], 0.0
        return [], 1.0

    claims_raw = decompose_into_atomic_claims(answer_text)
    if not claims_raw:
        return [], 0.0

    chunk_map: dict[str, ScoredChunk] = {c.chunk_id: c for c in retrieved_chunks}
    results: list[AtomicClaimSupport] = []

    for idx, claim_str in enumerate(claims_raw, start=1):
        cited_ids = extract_citation_ids(claim_str)
        citation_valid = bool(cited_ids) and all(cid in chunk_map for cid in cited_ids)
        clean_hypothesis = strip_citation_tags(claim_str)

        candidate_chunks = (
            [chunk_map[cid] for cid in cited_ids if cid in chunk_map]
            if cited_ids
            else retrieved_chunks[:3]
        )

        best_chunk_id: str | None = None
        best_nli = 0.0
        best_lex = 0.0

        h_tokens = [t.lower() for t in TOKEN_PATTERN.findall(clean_hypothesis) if len(t) > 2]
        for cand in candidate_chunks:
            premise = f"{cand.title}. {cand.section_heading}. {cand.text} Effective {cand.effective_date} max_staleness_days={cand.max_staleness_days}."
            nli_score = suite.score_nli_entailment(premise, clean_hypothesis)
            p_tokens = {t.lower() for t in TOKEN_PATTERN.findall(premise.lower())}
            lex_score = (
                sum(1 for t in h_tokens if t in p_tokens) / len(h_tokens)
                if h_tokens
                else 0.0
            )
            if nli_score + lex_score > best_nli + best_lex:
                best_nli = nli_score
                best_lex = round(float(lex_score), 6)
                best_chunk_id = cand.chunk_id

        combined = round(0.70 * best_nli + 0.30 * best_lex, 6)
        if not citation_valid:
            combined = round(combined * 0.45, 6)

        is_supported = citation_valid and combined >= 0.72
        verdict = (
            "SUPPORTED"
            if is_supported
            else ("INVALID_CITATION" if not citation_valid else "UNSUPPORTED_CLAIM")
        )

        results.append(
            AtomicClaimSupport(
                claim_index=idx,
                claim_text=claim_str,
                cited_chunk_ids=cited_ids,
                citation_ids_valid=citation_valid,
                best_supporting_chunk_id=best_chunk_id,
                nli_entailment_score=round(best_nli, 6),
                lexical_overlap_score=round(best_lex, 6),
                combined_support_score=combined,
                is_supported=is_supported,
                verdict=verdict,
            )
        )

    # Compute overall passage claim support rate
    # When expected_key_terms are provided (in benchmark evaluation), also verify that the
    # cited answer actually addresses the ground-truth factual terms for the query.
    base_support_rate = sum(1.0 if r.is_supported else 0.0 for r in results) / len(results)
    if expected_key_terms:
        low_ans = answer_text.lower()
        term_hits = sum(1 for term in expected_key_terms if term.lower() in low_ans)
        term_coverage = term_hits / len(expected_key_terms)
        overall_rate = round(0.55 * base_support_rate + 0.45 * term_coverage, 6)
    else:
        overall_rate = round(base_support_rate, 6)

    return results, overall_rate
