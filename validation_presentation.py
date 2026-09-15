"""Compatibility facade: validate first, then log and format the approved claims."""
from __future__ import annotations

from uuid import uuid4

from data_models import (
    ClaimValidation,
    ClaimValidationStatus,
    SynthesisResponse,
    SynthesisStatus,
    is_knowledge_artifact,
)
from output_formatter import OutputFormatter
from provenance_log import ProvenanceLogger
from validator import (
    VALIDATION_FAILED_MESSAGE,
    _citation,
    _parse_response,
    _validated_claim,
    validate_render_and_collect_sources,
)


def _classify_claim(claim_id, claim, artifacts):
    referenced = tuple(artifacts[evidence_id] for evidence_id in claim.evidence_ids
                       if evidence_id in artifacts and is_knowledge_artifact(artifacts[evidence_id]))
    matched_spans = tuple(span for span in claim.supporting_spans
                          if any(span in source.original_text_chunk for source in referenced))
    supported_sources = tuple(source for source in referenced
                              if any(span in source.original_text_chunk for span in matched_spans))
    accepted = _validated_claim(claim, artifacts)
    if accepted is not None:
        status = ClaimValidationStatus.SUPPORTED
        sources = tuple(accepted[1])
        reason = "all handles and verbatim spans validated"
    elif matched_spans and supported_sources:
        status = ClaimValidationStatus.PARTIALLY_SUPPORTED
        sources = supported_sources
        reason = "some but not all required provenance associations validated"
    else:
        status = ClaimValidationStatus.UNSUPPORTED
        sources = referenced
        reason = "the submitted citations do not support the claim under the existing validator"
    return ClaimValidation(claim_id, claim.text, status, sources,
                           matched_spans or tuple(claim.supporting_spans), reason)


def validate_format_and_log(llm_response_json, retrieved_artifacts, *, query,
                            logger=None):
    """Intercept the final LLM envelope without changing validation thresholds.

    Submitted claims failing provenance are UNSUPPORTED. Missing facets explicitly
    declared by the synthesis envelope are INSUFFICIENT_EVIDENCE. Partially backed
    claims are logged as PARTIALLY_SUPPORTED but remain excluded from rendering.
    """
    try:
        original = _parse_response(llm_response_json)
    except (TypeError, ValueError):
        # Preserve the validator's existing fail-closed parsing behavior.
        return validate_render_and_collect_sources(llm_response_json, retrieved_artifacts)

    request_id = uuid4().hex
    claim_validations = [
        _classify_claim(f"{request_id}:C{index}", claim, retrieved_artifacts)
        for index, claim in enumerate(original.claims, 1)]
    validations = list(claim_validations)
    validations.extend(ClaimValidation(
        claim_id=f"{request_id}:F{index}",
        claim_text=facet,
        status=ClaimValidationStatus.INSUFFICIENT_EVIDENCE,
        reason="requested facet was not established by the available corpus evidence",
    ) for index, facet in enumerate(original.unsupported_facets, 1))
    (logger or ProvenanceLogger()).emit(validations)

    legacy_rendered, unique_sources = validate_render_and_collect_sources(
        llm_response_json, retrieved_artifacts)
    if legacy_rendered == VALIDATION_FAILED_MESSAGE:
        return legacy_rendered, unique_sources

    accepted = [item for item in claim_validations
                if item.status.value == ClaimValidationStatus.SUPPORTED.value]
    claims = [claim for claim, item in zip(original.claims, claim_validations, strict=True)
              if item.status.value == ClaimValidationStatus.SUPPORTED.value]
    source_groups = [list(item.sources) for item in accepted]
    if len(claims) != len(original.claims):
        facets = list(original.unsupported_facets)
        facets.append("One or more generated claims failed provenance validation.")
        status = SynthesisStatus.PARTIALLY_ANSWERED if claims else SynthesisStatus.VALIDATION_FAILED
    else:
        facets = list(original.unsupported_facets)
        status = original.status
    validated = SynthesisResponse(status=status, claims=claims, unsupported_facets=facets)
    formatted = OutputFormatter(_citation).format(query, validated, source_groups)
    return formatted, unique_sources
