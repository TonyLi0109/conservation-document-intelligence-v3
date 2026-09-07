"""Deterministic evaluation measurements; these are not semantic entailment judges.

No provider, network, production database, or environment configuration is loaded.
Unknown denominators are reported as ``None``, never as a successful empty score.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from data_models import Claim, KnowledgeArtifact, is_knowledge_artifact
from validator import (
    INSUFFICIENT_MESSAGE,
    SYSTEM_FALLBACK_MESSAGE,
    VALIDATION_FAILED_MESSAGE,
    format_artifact_location,
)


def retrieval_metrics(
    ranked_document_ids: list[str], relevance: dict[str, int], k: int,
) -> dict[str, Any]:
    """Measure document retrieval against explicit nonnegative graded judgments.

    Duplicate document IDs are removed before applying the cutoff. Precision uses
    denominator ``k`` even for short result sets; recall uses all positively
    judged documents. MRR uses the complete supplied ranking. NDCG uses gain
    ``2**grade - 1`` and log2 discount. Unjudged results have gain zero, but their
    count is exposed: these measurements do not establish annotation completeness.
    With no positive relevance judgments, all five quality scores are undefined.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not isinstance(relevance, dict) or any(
        not isinstance(key, str) or not key.strip()
        or isinstance(grade, bool) or not isinstance(grade, int) or grade < 0
        for key, grade in relevance.items()
    ):
        raise ValueError("relevance must map document IDs to nonnegative integer grades")
    if not isinstance(ranked_document_ids, list) or any(
        not isinstance(item, str) or not item.strip() for item in ranked_document_ids
    ):
        raise ValueError("ranked_document_ids must be a list of nonempty document IDs")
    ranked = list(dict.fromkeys(ranked_document_ids))
    relevant = {doc for doc, grade in relevance.items() if grade > 0}
    result: dict[str, Any] = {
        "k": k,
        "ranked_count": len(ranked),
        "duplicate_count": len(ranked_document_ids) - len(ranked),
        "judged_count": len(relevance),
        "relevant_count": len(relevant),
        "unjudged_at_k": sum(doc not in relevance for doc in ranked[:k]),
        "precision_at_k": None,
        "recall_at_k": None,
        "mrr": None,
        "ndcg_at_k": None,
        "hit_at_k": None,
    }
    if not relevant:
        return result
    hits = sum(doc in relevant for doc in ranked[:k])
    dcg = sum(
        (2 ** relevance.get(doc, 0) - 1) / math.log2(position + 2)
        for position, doc in enumerate(ranked[:k])
    )
    ideal = sum(
        (2 ** grade - 1) / math.log2(position + 2)
        for position, grade in enumerate(sorted(relevance.values(), reverse=True)[:k])
    )
    result.update(
        precision_at_k=hits / k,
        recall_at_k=hits / len(relevant),
        mrr=next((1 / rank for rank, doc in enumerate(ranked, 1) if doc in relevant), 0.0),
        ndcg_at_k=dcg / ideal,
        hit_at_k=float(hits > 0),
    )
    return result


def _identity(artifact: object) -> tuple[str | None, ...] | None:
    if not is_knowledge_artifact(artifact):
        return None
    return tuple(getattr(artifact, field, None) for field in (
        "document_id", "title", "page_number", "original_text_chunk",
        "source_url", "printed_page_label",
    ))


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


# Deliberately lexical: a matching number does not prove its unit, polarity,
# population, time period, or causal interpretation. Words such as "three" and
# derived arithmetic are not assessed. Percentage markers remain significant.
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%)?(?!\w)")


def _numbers(text: str) -> set[str]:
    return {re.sub(r"[\s,]", "", item) for item in _NUMBER.findall(text)}


def evaluate_claims(
    payload: dict[str, Any],
    artifacts: dict[str, KnowledgeArtifact],
    canonical_artifacts: Sequence[KnowledgeArtifact] | None = None,
) -> dict[str, Any]:
    """Check handles, span ownership, canonical membership and numeric overlap.

    ``payload`` is a synthesis envelope containing ``claims`` (the optional
    preamble and other envelope fields are not scored). Each claim must exactly
    match the production Claim schema. Every cited handle must contribute an
    exact span, and every span must belong to a cited artifact. If supplied,
    ``canonical_artifacts`` is an independently trusted fixture/store snapshot;
    without it canonical metadata authenticity is explicitly not assessed.

    Numeric support requires every digit token in the claim to occur in its
    verified supporting spans. This and exact-span validity are proxies, not an
    entailment, correctness, or answer-completeness judgment.
    """
    canonical = None if canonical_artifacts is None else {
        identity for item in canonical_artifacts if (identity := _identity(item)) is not None
    }
    raw_claims = payload.get("claims") if isinstance(payload, dict) else None
    schema_valid = isinstance(raw_claims, list)
    failures: list[dict[str, Any]] = []
    if not schema_valid:
        failures.append({"claim_index": None, "reasons": ["claims_must_be_an_array"]})
        raw_claims = []
    valid_count = numeric_count = numeric_supported = grounded_count = 0
    for index, raw in enumerate(raw_claims):
        reasons: list[str] = []
        raw_text = raw.get("text", "") if isinstance(raw, dict) else ""
        numbers = _numbers(raw_text) if isinstance(raw_text, str) else set()
        numeric_count += bool(numbers)
        try:
            if not isinstance(raw, dict) or set(raw) != {"text", "evidence_ids", "supporting_spans"}:
                raise ValueError("claim schema mismatch")
            claim = Claim(**raw)
        except (TypeError, ValueError):
            schema_valid = False
            failures.append({"claim_index": index, "reasons": ["invalid_claim_schema"]})
            continue
        referenced: list[KnowledgeArtifact] = []
        for handle in claim.evidence_ids:
            artifact = artifacts.get(handle) if isinstance(artifacts, Mapping) else None
            identity = _identity(artifact)
            if identity is None:
                reasons.append("unknown_or_invalid_evidence_handle")
                continue
            if canonical is not None and identity not in canonical:
                reasons.append("artifact_not_in_canonical_evidence")
            referenced.append(artifact)
        contributed: set[int] = set()
        verified_spans: list[str] = []
        for span in claim.supporting_spans:
            owners = {position for position, item in enumerate(referenced) if span in item.original_text_chunk}
            if owners:
                contributed.update(owners)
                verified_spans.append(span)
            else:
                reasons.append("span_not_verbatim_in_cited_evidence")
        if contributed != set(range(len(referenced))):
            reasons.append("cited_artifact_has_no_supporting_span")
        provenance_valid = not reasons
        valid_count += provenance_valid
        unsupported_numbers = sorted(numbers - _numbers("\n".join(verified_spans)))
        if unsupported_numbers:
            reasons.append("numeric_token_not_in_verified_spans")
        numeric_supported += bool(numbers) and provenance_valid and not unsupported_numbers
        grounded_count += provenance_valid and not unsupported_numbers
        if reasons:
            failure: dict[str, Any] = {"claim_index": index, "reasons": list(dict.fromkeys(reasons))}
            if unsupported_numbers:
                failure["unsupported_numeric_tokens"] = unsupported_numbers
            failures.append(failure)
    count = len(raw_claims)
    return {
        "claim_count": count,
        "valid_claim_count": valid_count,
        "provenance_validity_rate": _rate(valid_count, count),
        "numeric_claim_count": numeric_count,
        "numeric_supported_count": numeric_supported,
        "numeric_support_rate": _rate(numeric_supported, numeric_count),
        "grounded_proxy_rate": _rate(grounded_count, count),
        "canonical_membership_assessed": canonical is not None,
        "semantic_support": "not_assessed",
        "schema_valid": schema_valid,
        "passed": bool(count) and schema_valid and not failures,
        "failures": failures,
    }


def _citation(artifact: KnowledgeArtifact) -> str:
    title = " ".join(artifact.title.split()).replace("[", "\\[").replace("]", "\\]")
    return f"[{artifact.document_id} \u2014 {title}, {format_artifact_location(artifact)}]"


_BRACKET = re.compile(r"\[(?:\\.|[^\]\\\n])*\]")
_CITATION_START = re.compile(r"\[(?:DOC|K\d)", re.IGNORECASE)
_CITATION_ID = re.compile(r"^\[(DOC\d{3,})\s+\u2014\s+.+,\s+(?:PDF pp?\.|printed p\.|Web)")
_BULLET = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_ABSTENTION_PREFIXES = (
    INSUFFICIENT_MESSAGE, VALIDATION_FAILED_MESSAGE, SYSTEM_FALLBACK_MESSAGE,
    "Please name the subject, method, option, or report",
    "I couldn't safely resolve this follow-up",
)


def evaluate_answer(
    markdown: str,
    retrieved_artifacts: Sequence[KnowledgeArtifact],
    cited_sources: Sequence[KnowledgeArtifact],
    expect_claims: bool = True,
) -> dict[str, Any]:
    """Audit rendered citation metadata and conservative claim-line coverage.

    A canonical retrieved artifact is the independent reference for the rendered
    citation AND returned source object, including original text and URL. This
    does not independently validate the retrieval snapshot against the database.
    The production unsupported-facets section and known abstention text are
    excluded from claim counting. Other prose/bullets are candidate claims;
    classifying them as factual or entailed still requires human/judge review.
    Sources-only output is identified separately and fails when claims are
    expected, even if every source citation is authentic.
    """
    if not isinstance(markdown, str):
        raise TypeError("markdown must be a string")
    canonical = {_identity(item) for item in retrieved_artifacts if _identity(item) is not None}
    expected: dict[str, set[tuple[str | None, ...]]] = {}
    for item in retrieved_artifacts:
        if (identity := _identity(item)) is not None:
            expected.setdefault(_citation(item), set()).add(identity)
    documents = {item.document_id for item in retrieved_artifacts if _identity(item) is not None}
    source_identities = [_identity(item) for item in cited_sources]
    source_set = set(source_identities) - {None}
    invalid_sources = sum(identity is None or identity not in canonical for identity in source_identities)
    source_duplicates = len(source_identities) - len(set(source_identities))
    failures: list[dict[str, Any]] = []
    if invalid_sources:
        failures.append({"reason": "returned_source_not_in_retrieved_evidence", "count": invalid_sources})

    matches = [match for match in _BRACKET.finditer(markdown) if _CITATION_START.match(match.group())]
    citations = [match.group() for match in matches]
    # An unmatched opening citation is malformed too; do not silently ignore it.
    starts = list(_CITATION_START.finditer(markdown))
    unmatched = sum(not any(match.start() <= start.start() < match.end() for match in matches) for start in starts)
    valid_count = malformed_count = 0
    if unmatched:
        failures.append({"reason": "malformed_citation", "count": unmatched})
        malformed_count += unmatched
    valid_citations: set[str] = set()
    for index, citation in enumerate(citations):
        if citation in expected:
            if not expected[citation].intersection(source_set):
                failures.append({"citation_index": index, "reason": "citation_missing_from_returned_sources"})
            else:
                valid_count += 1
                valid_citations.add(citation)
            continue
        parsed = _CITATION_ID.match(citation)
        if parsed is None:
            reason = "malformed_citation"
            malformed_count += 1
        elif parsed.group(1) not in documents:
            reason = "nonexistent_document_citation"
        else:
            reason = "forged_or_mismatched_citation_metadata"
        failures.append({"citation_index": index, "reason": reason})

    rendered_source_ids = {
        identity for citation in citations if citation in expected for identity in expected[citation]
    }
    unreferenced = len(source_set - rendered_source_ids)
    if unreferenced:
        failures.append({"reason": "returned_source_has_no_rendered_citation", "count": unreferenced})
    claim_count = covered_count = factual_bullets = sources_only_lines = 0
    in_unsupported = False
    for line in markdown.splitlines():
        text = line.strip()
        if not text:
            continue
        if re.fullmatch(r"(?:\*\*)?Unsupported facets(?:\*\*)?", text, re.IGNORECASE):
            in_unsupported = True
            continue
        if text.startswith("#") or (text.startswith("**") and text.endswith("**")):
            in_unsupported = False
            continue
        if in_unsupported or text.startswith(_ABSTENTION_PREFIXES):
            continue
        line_citations = [match.group() for match in _BRACKET.finditer(text) if _CITATION_START.match(match.group())]
        content = _BULLET.sub("", text)
        for citation in line_citations:
            content = content.replace(citation, "")
        if not content.strip():
            sources_only_lines += bool(line_citations)
            continue
        claim_count += 1
        factual_bullets += bool(_BULLET.match(text))
        covered_count += any(citation in valid_citations for citation in line_citations)

    recognized_abstention = any(line.strip().startswith(_ABSTENTION_PREFIXES) for line in markdown.splitlines())
    kind = "answer" if claim_count else (
        "sources_only" if sources_only_lines else "abstention" if recognized_abstention
        else "unclassified" if markdown.strip() else "empty"
    )
    if kind == "unclassified":
        failures.append({"reason": "output_has_no_recognized_answer_or_abstention"})
    if claim_count > covered_count:
        failures.append({"reason": "candidate_claim_without_valid_citation", "count": claim_count - covered_count})
    if expect_claims and not claim_count:
        failures.append({"reason": "expected_claims_but_no_answer"})
    total_citations = len(citations) + unmatched
    return {
        "answer_kind": kind,
        "claim_count": claim_count,
        "factual_bullet_count": factual_bullets,
        "covered_claim_count": covered_count,
        "uncited_claim_count": claim_count - covered_count,
        "citation_coverage": _rate(covered_count, claim_count),
        "citation_count": total_citations,
        "valid_citation_count": valid_count,
        "citation_validity_rate": _rate(valid_count, total_citations),
        "malformed_citation_count": malformed_count,
        "duplicate_citation_count": sum(count - 1 for count in Counter(citations).values()),
        "source_count": len(cited_sources),
        "invalid_source_count": invalid_sources,
        "source_membership_rate": _rate(len(cited_sources) - invalid_sources, len(cited_sources)),
        "duplicate_source_count": source_duplicates,
        "unreferenced_source_count": unreferenced,
        "semantic_support": "not_assessed",
        "passed": bool(markdown.strip()) and not failures,
        "failures": failures,
    }
