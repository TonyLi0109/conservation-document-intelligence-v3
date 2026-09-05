"""Core data contracts for Conservation Document Intelligence V3.

These models deliberately contain no retrieval scores, embedding vectors, model
configuration, or database concerns. Ranking data belongs in separate retrieval
envelopes so the canonical provenance chain cannot be altered by search logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


_EVIDENCE_ID_PATTERN = re.compile(r"^K[1-9]\d*$")
_DOCUMENT_ID_PATTERN = re.compile(r"^DOC\d{3,}$")
_HTTP_URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


def _require_non_empty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty or whitespace-only")


def _require_string_list(value: Any, field_name: str, *, allow_empty: bool) -> None:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list of strings")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must not be empty")
    for index, item in enumerate(value):
        _require_non_empty_string(item, f"{field_name}[{index}]")


@dataclass(frozen=True, slots=True)
class KnowledgeArtifact:
    """Canonical, immutable evidence retrieved from one source location.

    This object is the source of truth for provenance. ``document_id``, title,
    page, and original text must come from trusted local ingestion—not from LLM
    output. ``page_number`` accepts a physical page/range or the literal ``Web``
    for a source without physical pagination. The original chunk must remain
    unchanged; excerpts and supporting spans are validated against it.

    Opaque handles such as ``K1`` are assigned outside this object for each
    synthesis request. Embeddings, distances, ranks, and scores must never be
    added to or used to mutate this canonical record.
    """

    document_id: str
    title: str
    page_number: str
    original_text_chunk: str
    source_url: str | None = None
    printed_page_label: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.document_id, "document_id")
        _require_non_empty_string(self.title, "title")
        _require_non_empty_string(self.page_number, "page_number")
        _require_non_empty_string(self.original_text_chunk, "original_text_chunk")
        if self.source_url is not None:
            _require_non_empty_string(self.source_url, "source_url")
            if not _HTTP_URL_PATTERN.match(self.source_url):
                raise ValueError("source_url must use http or https")
        if self.printed_page_label is not None:
            _require_non_empty_string(
                self.printed_page_label, "printed_page_label"
            )


def is_knowledge_artifact(value: object) -> bool:
    """Return whether ``value`` satisfies the canonical artifact contract.

    This deliberately uses structural validation instead of class identity.
    Streamlit can retain cached objects across a module hot reload, making a
    valid instance of the previous ``KnowledgeArtifact`` class fail
    ``isinstance`` against the newly imported class.
    """

    required = ("document_id", "title", "page_number", "original_text_chunk")
    for name in required:
        field_value = getattr(value, name, None)
        if not isinstance(field_value, str) or not field_value.strip():
            return False
    source_url = getattr(value, "source_url", None)
    if source_url is not None and (
        not isinstance(source_url, str) or not _HTTP_URL_PATTERN.match(source_url)
    ):
        return False
    printed_page_label = getattr(value, "printed_page_label", None)
    if printed_page_label is not None and (
        not isinstance(printed_page_label, str) or not printed_page_label.strip()
    ):
        return False
    return True


@dataclass(frozen=True, slots=True)
class DocumentSource:
    """Trusted document-level provenance imported into the V3 source registry.

    This contract is owned by V3 even when its values were initially curated
    from a legacy catalog. ``source_url`` identifies the public catalog/landing
    source, while ``resolved_url`` may identify the exact downloaded resource.
    """

    document_id: str
    title: str
    source_url: str | None
    resolved_url: str | None
    local_filename: str
    file_type: str
    year: str = ""
    agency: str = ""
    topic: str = ""

    def __post_init__(self) -> None:
        _require_non_empty_string(self.document_id, "document_id")
        if not _DOCUMENT_ID_PATTERN.fullmatch(self.document_id):
            raise ValueError("document_id must use the stable DOC001 form")
        _require_non_empty_string(self.title, "title")
        _require_non_empty_string(self.local_filename, "local_filename")
        _require_non_empty_string(self.file_type, "file_type")
        for field_name in ("source_url", "resolved_url"):
            value = getattr(self, field_name)
            if value is not None:
                _require_non_empty_string(value, field_name)
                if not _HTTP_URL_PATTERN.match(value):
                    raise ValueError(f"{field_name} must use http or https")


@dataclass(frozen=True, slots=True)
class EvidenceLink:
    """A verified edge from derived knowledge back to canonical evidence.

    ``exact_span`` must be checked against the identified artifact before this
    contract is constructed. Document metadata is copied only from trusted
    storage, never from model output.
    """

    artifact_id: int
    document_id: str
    title: str
    page_number: str
    exact_span: str
    source_url: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, int) or isinstance(self.artifact_id, bool) or self.artifact_id <= 0:
            raise ValueError("artifact_id must be a positive integer")
        for value, name in (
            (self.document_id, "document_id"),
            (self.title, "title"),
            (self.page_number, "page_number"),
            (self.exact_span, "exact_span"),
        ):
            _require_non_empty_string(value, name)
        if self.source_url is not None and not _HTTP_URL_PATTERN.match(self.source_url):
            raise ValueError("source_url must use http or https")


@dataclass(frozen=True, slots=True)
class CompiledRelationship:
    """A normalized relationship retained only with supporting evidence."""

    entity_name: str
    relationship_type: str
    evidence: list[EvidenceLink]

    def __post_init__(self) -> None:
        _require_non_empty_string(self.entity_name, "entity_name")
        _require_non_empty_string(self.relationship_type, "relationship_type")
        if not isinstance(self.evidence, list) or not self.evidence:
            raise ValueError("compiled relationships require evidence")
        if any(not isinstance(item, EvidenceLink) for item in self.evidence):
            raise TypeError("relationship evidence must contain EvidenceLink objects")


@dataclass(frozen=True, slots=True)
class CompiledKnowledgeArtifact:
    """Reusable knowledge compiled from, but never replacing, raw evidence.

    This is the professor-facing *knowledge item*. It is intentionally distinct
    from :class:`KnowledgeArtifact`, which remains an immutable source chunk.
    Facts and relationships may be displayed or retrieved only while their
    EvidenceLink records still resolve to canonical source text.
    """

    knowledge_id: str
    concept_title: str
    summary: str
    important_facts: list[str]
    related_entities: list[CompiledRelationship]
    supporting_evidence: list[EvidenceLink]
    method: str
    version: str
    model_name: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.knowledge_id, "knowledge_id"),
            (self.concept_title, "concept_title"),
            (self.summary, "summary"),
            (self.method, "method"),
            (self.version, "version"),
            (self.model_name, "model_name"),
        ):
            _require_non_empty_string(value, name)
        _require_string_list(self.important_facts, "important_facts", allow_empty=True)
        if not isinstance(self.related_entities, list) or any(
            not isinstance(item, CompiledRelationship) for item in self.related_entities
        ):
            raise TypeError("related_entities must contain CompiledRelationship objects")
        if not isinstance(self.supporting_evidence, list) or not self.supporting_evidence:
            raise ValueError("compiled knowledge requires supporting evidence")
        if any(not isinstance(item, EvidenceLink) for item in self.supporting_evidence):
            raise TypeError("supporting_evidence must contain EvidenceLink objects")


@dataclass(frozen=True, slots=True)
class Claim:
    """One atomic factual assertion and its explicit evidence associations.

    ``evidence_ids`` contains only request-local opaque handles (``K1``, ``K2``,
    and so on), never model-created document metadata. Each supporting span must
    be a verbatim substring of the ``original_text_chunk`` belonging to at least
    one referenced artifact. The synthesis validator—not the LLM—must resolve
    handles, verify those substrings, and render trusted citations from the
    corresponding :class:`KnowledgeArtifact` values.
    """

    text: str
    evidence_ids: list[str]
    supporting_spans: list[str]

    def __post_init__(self) -> None:
        _require_non_empty_string(self.text, "text")
        _require_string_list(self.evidence_ids, "evidence_ids", allow_empty=False)
        _require_string_list(
            self.supporting_spans, "supporting_spans", allow_empty=False
        )
        invalid_ids = [
            evidence_id
            for evidence_id in self.evidence_ids
            if not _EVIDENCE_ID_PATTERN.fullmatch(evidence_id)
        ]
        if invalid_ids:
            raise ValueError(
                "evidence_ids must use opaque handles K1, K2, ...; invalid: "
                + ", ".join(invalid_ids)
            )
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must not contain duplicates")
        if len(set(self.supporting_spans)) != len(self.supporting_spans):
            raise ValueError("supporting_spans must not contain duplicates")


class SynthesisStatus(str, Enum):
    """Permitted outcomes of evidence-grounded synthesis."""

    ANSWERED = "answered"
    PARTIALLY_ANSWERED = "partially_answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    VALIDATION_FAILED = "validation_failed"
    SYSTEM_FALLBACK = "system_fallback"


@dataclass(frozen=True, slots=True)
class SynthesisResponse:
    """Structured synthesis result before application-owned citation rendering.

    Claims may refer only to opaque evidence handles. A downstream validator
    resolves each handle to a :class:`KnowledgeArtifact`, proves every supporting
    span occurs verbatim in canonical text, and only then renders document ID,
    title, and page citations. ``unsupported_facets`` records requested portions
    that the corpus could not support; it must never be presented as established
    fact.

    Answered responses contain claims. Partial responses contain both claims and
    unsupported facets. An insufficient-evidence response contains no claims and
    identifies at least one missing facet. Validation failure contains no claims,
    because unvalidated assertions must never reach the user.
    """

    status: SynthesisStatus
    claims: list[Claim] = field(default_factory=list)
    unsupported_facets: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.status, SynthesisStatus):
            raise TypeError("status must be a SynthesisStatus")
        if not isinstance(self.claims, list) or any(
            not isinstance(claim, Claim) for claim in self.claims
        ):
            raise TypeError("claims must be a list of Claim objects")
        _require_string_list(
            self.unsupported_facets, "unsupported_facets", allow_empty=True
        )

        if self.status is SynthesisStatus.ANSWERED:
            if not self.claims:
                raise ValueError("answered responses must contain at least one claim")
            if self.unsupported_facets:
                raise ValueError("answered responses cannot contain unsupported facets")
        elif self.status is SynthesisStatus.PARTIALLY_ANSWERED:
            if not self.claims or not self.unsupported_facets:
                raise ValueError(
                    "partially_answered responses require claims and unsupported facets"
                )
        elif self.status is SynthesisStatus.INSUFFICIENT_EVIDENCE:
            if self.claims:
                raise ValueError("insufficient_evidence responses cannot contain claims")
            if not self.unsupported_facets:
                raise ValueError(
                    "insufficient_evidence responses require unsupported facets"
                )
        elif self.status is SynthesisStatus.VALIDATION_FAILED and self.claims:
            raise ValueError("validation_failed responses cannot contain claims")
