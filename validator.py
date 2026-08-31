"""Fail-closed validation and trusted citation rendering for V3 synthesis."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from data_models import Claim, KnowledgeArtifact, SynthesisResponse, SynthesisStatus


LOGGER = logging.getLogger(__name__)


INSUFFICIENT_MESSAGE = (
    "The corpus does not provide enough evidence to answer that question reliably."
)
VALIDATION_FAILED_MESSAGE = (
    "The generated answer could not be verified against the retrieved source text."
)
SYSTEM_FALLBACK_MESSAGE = (
    "The synthesis service is unavailable. Please try again or review the retrieved sources."
)


def _parse_response(llm_response_json: str) -> SynthesisResponse:
    """Parse exact model JSON and instantiate the validated domain contracts."""

    if not isinstance(llm_response_json, str) or not llm_response_json.strip():
        raise ValueError("llm_response_json must be a non-empty string")
    payload: Any = json.loads(llm_response_json)
    if not isinstance(payload, dict):
        raise TypeError("the LLM response must be a JSON object")
    expected_keys = {"status", "claims", "unsupported_facets"}
    if set(payload) != expected_keys:
        raise ValueError("the LLM response contains missing or additional fields")

    raw_claims = payload["claims"]
    if not isinstance(raw_claims, list):
        raise TypeError("claims must be a JSON array")
    claims: list[Claim] = []
    claim_keys = {"text", "evidence_ids", "supporting_spans"}
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, dict) or set(raw_claim) != claim_keys:
            raise ValueError("each claim must exactly match the Claim schema")
        claims.append(Claim(**raw_claim))

    try:
        status = SynthesisStatus(payload["status"])
    except (TypeError, ValueError) as error:
        raise ValueError("unknown synthesis status") from error
    return SynthesisResponse(
        status=status,
        claims=claims,
        unsupported_facets=payload["unsupported_facets"],
    )


def _validated_claim(
    claim: Claim,
    artifacts: dict[str, KnowledgeArtifact],
) -> tuple[Claim, list[KnowledgeArtifact]] | None:
    """Return a claim and its trusted sources only if provenance is complete.

    Every span must occur verbatim in at least one referenced artifact, and every
    referenced artifact must contribute at least one span. This prevents a model
    from attaching an unrelated but valid source to an otherwise supported claim.
    """

    referenced: list[KnowledgeArtifact] = []
    for evidence_id in claim.evidence_ids:
        artifact = artifacts.get(evidence_id)
        if not isinstance(artifact, KnowledgeArtifact):
            LOGGER.debug(
                "Rejected claim %r: unknown evidence handle %r",
                claim.text,
                evidence_id,
            )
            return None
        referenced.append(artifact)

    supported_artifact_indexes: set[int] = set()
    for span in claim.supporting_spans:
        matching_indexes = {
            index
            for index, artifact in enumerate(referenced)
            if artifact.original_text_chunk.find(span) >= 0
        }
        if not matching_indexes:
            LOGGER.debug(
                "Rejected claim %r: supporting span is not verbatim in referenced artifacts: %r",
                claim.text,
                span,
            )
            return None
        supported_artifact_indexes.update(matching_indexes)

    if supported_artifact_indexes != set(range(len(referenced))):
        unsupported_ids = [
            evidence_id
            for index, evidence_id in enumerate(claim.evidence_ids)
            if index not in supported_artifact_indexes
        ]
        LOGGER.debug(
            "Rejected claim %r: referenced artifacts have no matching span: %r; spans=%r",
            claim.text,
            unsupported_ids,
            claim.supporting_spans,
        )
        return None
    return claim, referenced


def _page_label(page_number: str) -> str:
    """Render a trusted physical page, page range, or web location."""

    page = page_number.strip()
    if page.casefold() == "web":
        return "Web"
    if re.fullmatch(r"\d+", page):
        return f"p. {page}"
    if re.fullmatch(r"\d+\s*[-–—]\s*\d+", page):
        start, end = re.split(r"\s*[-–—]\s*", page, maxsplit=1)
        return f"pp. {start}–{end}"
    return f"p. {page}"


def format_artifact_location(artifact: KnowledgeArtifact) -> str:
    """Render logical and physical location without conflating their meaning."""

    physical = artifact.page_number.strip()
    if physical.casefold() == "web":
        return "Web"
    printed = (
        artifact.printed_page_label.strip()
        if artifact.printed_page_label is not None
        else ""
    )
    physical_prefix = (
        "PDF pp." if re.fullmatch(r"\d+\s*[-–—]\s*\d+", physical)
        else "PDF p."
    )
    physical_label = f"{physical_prefix} {physical}"
    if printed and printed.casefold() != physical.casefold():
        return f"printed p. {printed}; {physical_label}"
    return physical_label


def _citation(artifact: KnowledgeArtifact) -> str:
    """Render citation metadata owned by the application, never the model."""

    safe_title = " ".join(artifact.title.split()).replace("[", "\\[").replace(
        "]", "\\]"
    )
    return (
        f"[{artifact.document_id} — {safe_title}, "
        f"{format_artifact_location(artifact)}]"
    )


def _render(response: SynthesisResponse, sources: list[list[KnowledgeArtifact]]) -> str:
    """Render a validated response without exposing opaque evidence handles."""

    if response.status is SynthesisStatus.INSUFFICIENT_EVIDENCE:
        facets = "\n".join(f"- {facet}" for facet in response.unsupported_facets)
        return f"{INSUFFICIENT_MESSAGE}\n\n**Unsupported facets**\n\n{facets}"
    if response.status is SynthesisStatus.VALIDATION_FAILED:
        return VALIDATION_FAILED_MESSAGE
    if response.status is SynthesisStatus.SYSTEM_FALLBACK and not response.claims:
        return SYSTEM_FALLBACK_MESSAGE

    lines: list[str] = []
    for claim, claim_sources in zip(response.claims, sources, strict=True):
        citations = " ".join(dict.fromkeys(_citation(item) for item in claim_sources))
        lines.append(f"- {claim.text} {citations}")
    if response.unsupported_facets:
        lines.extend(
            ["", "**Unsupported facets**", ""]
            + [f"- {facet}" for facet in response.unsupported_facets]
        )
    return "\n".join(lines)


def validate_and_render_response(
    llm_response_json: str,
    retrieved_artifacts: dict[str, KnowledgeArtifact],
) -> str:
    """Validate LLM JSON and return citation-safe Markdown.

    Invalid JSON, invalid contracts, unknown handles, and non-verbatim spans fail
    closed. If only some claims validate, invalid claims are removed and the
    result becomes ``partially_answered``. If no generated claim validates, the
    result becomes ``validation_failed`` and no model prose is rendered.
    """

    rendered, _ = validate_render_and_collect_sources(
        llm_response_json,
        retrieved_artifacts,
    )
    return rendered


def validate_render_and_collect_sources(
    llm_response_json: str,
    retrieved_artifacts: dict[str, KnowledgeArtifact],
) -> tuple[str, list[KnowledgeArtifact]]:
    """Validate, render, and return only sources attached to valid claims.

    The source list is deduplicated in first-citation order. Callers must never
    display the raw retrieval set as cited evidence because retrieval alone does
    not prove that a source supports a rendered claim.
    """

    if not isinstance(retrieved_artifacts, dict) or any(
        not isinstance(key, str) or not isinstance(value, KnowledgeArtifact)
        for key, value in retrieved_artifacts.items()
    ):
        raise TypeError(
            "retrieved_artifacts must be a dict[str, KnowledgeArtifact]"
        )

    try:
        response = _parse_response(llm_response_json)
    except (json.JSONDecodeError, TypeError, ValueError):
        return VALIDATION_FAILED_MESSAGE, []

    validated_claims: list[Claim] = []
    validated_sources: list[list[KnowledgeArtifact]] = []
    for claim in response.claims:
        result = _validated_claim(claim, retrieved_artifacts)
        if result is not None:
            valid_claim, sources = result
            validated_claims.append(valid_claim)
            validated_sources.append(sources)

    if response.claims and not validated_claims:
        failed = SynthesisResponse(
            status=SynthesisStatus.VALIDATION_FAILED,
            claims=[],
            unsupported_facets=response.unsupported_facets,
        )
        return _render(failed, []), []

    if len(validated_claims) != len(response.claims):
        unsupported = list(response.unsupported_facets)
        unsupported.append("One or more generated claims failed provenance validation.")
        response = SynthesisResponse(
            status=SynthesisStatus.PARTIALLY_ANSWERED,
            claims=validated_claims,
            unsupported_facets=unsupported,
        )

    unique_sources: list[KnowledgeArtifact] = []
    seen_sources: set[tuple[str, str, str, str]] = set()
    for claim_sources in validated_sources:
        for source in claim_sources:
            identity = (
                source.document_id,
                source.title,
                source.page_number,
                source.original_text_chunk,
            )
            if identity not in seen_sources:
                seen_sources.add(identity)
                unique_sources.append(source)
    return _render(response, validated_sources), unique_sources
