"""Compatibility facade: validate first, then log and format the approved claims."""
from __future__ import annotations

import json
import re
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

_CAUSAL_EVIDENCE_QUESTION = re.compile(
    r"\b(?:provide|show|contain)\w*\s+evidence\s+that\s+"
    r"(?P<cause>.+?)\s+caus(?:e|es|ed|ing)\s+(?P<effect>.+?)(?:\?|$)",
    re.I | re.S,
)
_DIRECT_CAUSAL_VERB = (
    r"(?:caus(?:e|es|ed|ing)|lead(?:s|ing)?\s+to|led\s+to|"
    r"result(?:s|ed|ing)?\s+in|driv(?:e|es|en|ing)|drove)"
)
_WEAK_CAUSAL_LANGUAGE = re.compile(
    r"\b(?:may|might|could|can|risk|risks|projected|projection|potential(?:ly)?|"
    r"uncertain(?:ty)?|associat(?:e|es|ed|ion)|interact(?:s|ed|ion)?)\b",
    re.I,
)
_EFFECT_STOPWORDS = {
    "a", "an", "and", "for", "in", "of", "or", "problem", "problems",
    "the", "to", "with",
}


def _phrase_tokens(text: str, *, effect: bool = False) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    if effect:
        tokens = [token for token in tokens if token not in _EFFECT_STOPWORDS]
    return tokens


def _contains_direct_causal_sentence(
    claims: list[dict], cause: str, effect: str
) -> bool:
    """Require one submitted evidence sentence to state the requested causal edge."""

    cause_phrase = r"\s+".join(map(re.escape, _phrase_tokens(cause)))
    effect_tokens = _phrase_tokens(effect, effect=True)
    if not cause_phrase or not effect_tokens:
        return False
    # Two content words avoid accepting a generic mention such as "problems".
    effect_phrase = r"\s+".join(map(re.escape, effect_tokens[:2]))
    forward = re.compile(
        rf"\b{cause_phrase}\b.{{0,180}}\b{_DIRECT_CAUSAL_VERB}\b"
        rf".{{0,180}}\b{effect_phrase}\b",
        re.I,
    )
    passive = re.compile(
        rf"\b{effect_phrase}\b.{{0,180}}\b(?:caused|driven|attributed)\s+by\b"
        rf".{{0,180}}\b{cause_phrase}\b",
        re.I,
    )
    for claim in claims:
        spans = claim.get("supporting_spans", [])
        if not isinstance(spans, list):
            continue
        for span in spans:
            if not isinstance(span, str):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", span):
                normalized = " ".join(sentence.split())
                if _WEAK_CAUSAL_LANGUAGE.search(normalized):
                    continue
                if forward.search(normalized) or passive.search(normalized):
                    return True
    return False


def enforce_causal_evidence_boundary(payload: dict, query: str) -> bool:
    """Fail closed when separate facts are chained into unsupported causation.

    Returns True when the payload was downgraded to a partial answer. Existing
    evidence spans and citations remain unchanged; only unsupported answer
    framing is removed.
    """

    if not isinstance(payload, dict) or not isinstance(query, str):
        return False
    match = _CAUSAL_EVIDENCE_QUESTION.search(query)
    claims = payload.get("claims")
    if match is None or not isinstance(claims, list) or not claims:
        return False
    cause = " ".join(match.group("cause").split())
    effect = " ".join(match.group("effect").split()).rstrip(".!?")
    if _contains_direct_causal_sentence(claims, cause, effect):
        return False

    facets = payload.get("unsupported_facets")
    if not isinstance(facets, list):
        return False
    gap = f"Direct causal evidence that {cause} causes {effect}"
    if not any(re.search(r"\b(?:direct\s+caus|causal\s+(?:evidence|attribution))", str(item), re.I)
               for item in facets):
        facets.append(gap)

    first = claims[0]
    if isinstance(first, dict) and isinstance(first.get("text"), str):
        claim_text = re.sub(r"^\s*Yes\.\s*", "", first["text"], count=1, flags=re.I)
        claim_text = re.sub(
            r"^(?:The\s+strongest\s+)?direct\s+"
            r"(?P<label>(?:Missouri(?:-specific)?\s+)?evidence)\s+"
            r"(?:is\s+that|reports\s+that|shows\s+that)",
            r"\g<label> shows that",
            claim_text,
            count=1,
            flags=re.I,
        )
        first["text"] = claim_text
    payload["status"] = "partially_answered"
    if "preamble" in payload:
        payload["preamble"] = ""
    return True


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
        payload = json.loads(llm_response_json)
        enforce_causal_evidence_boundary(payload, query)
        normalized_response_json = json.dumps(payload, ensure_ascii=False)
        original = _parse_response(normalized_response_json)
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
        normalized_response_json, retrieved_artifacts)
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
