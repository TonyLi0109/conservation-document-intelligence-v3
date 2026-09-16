"""Adaptive rendering of claims that already passed provenance validation."""
from __future__ import annotations

from collections import defaultdict
from enum import Enum
import re
from typing import Callable, Sequence

from data_models import Claim, KnowledgeArtifact, SynthesisResponse, SynthesisStatus

VALIDATED_FINDINGS_HEADING = "**Validated Findings:**"
EVIDENCE_GAPS_HEADING = "**Remaining evidence gaps / Unsupported facets:**"
FAILED_VALIDATION_DIAGNOSTIC = (
    "One or more generated claims failed provenance validation."
)


def format_evidence_gaps(facets: Sequence[str]) -> str:
    """Render UI gaps while keeping the machine-only global status out of prose."""
    visible = []
    for facet in facets:
        plain = re.sub(r"[*`]", "", str(facet)).strip().strip("_")
        if re.fullmatch(r"(?:Status\s*:\s*)?INSUFFICIENT[_ ]EVIDENCE", plain, re.I):
            continue
        visible.append(str(facet))
    if not visible:
        return ""
    return EVIDENCE_GAPS_HEADING + "\n\n" + "\n".join(f"- {item}" for item in visible)


def normalize_user_facing_answer(
    markdown: str,
    *,
    has_validated_findings: bool = False,
) -> str:
    """Enforce the presentation contract at the final UI boundary."""
    text = str(markdown or "")
    text = re.sub(
        r"(?i)[*_`]*Status\s*:\s*INSUFFICIENT[_ ]EVIDENCE[*_`]*",
        "",
        text,
    )
    lines = []
    for line in text.splitlines():
        removed_diagnostic = FAILED_VALIDATION_DIAGNOSTIC in line
        line = line.replace(FAILED_VALIDATION_DIAGNOSTIC, "")
        if removed_diagnostic:
            line = re.sub(r"[ \t]{2,}", " ", line).strip()
        if re.fullmatch(
            r"\s*(?:\*\*)?(?:Remaining evidence gaps / )?"
            r"Unsupported facets:?(?:\*\*)?\s*",
            line,
            re.I,
        ):
            line = EVIDENCE_GAPS_HEADING
        if re.fullmatch(r"\s*(?:[-+*]\s*)?", line):
            line = ""
        if line or not lines or lines[-1]:
            lines.append(line.rstrip())

    while lines and not lines[-1]:
        lines.pop()
    if lines and lines[-1] == EVIDENCE_GAPS_HEADING:
        lines.pop()
        while lines and not lines[-1]:
            lines.pop()

    rendered = "\n".join(lines).strip()
    already_prefixed = bool(re.match(
        r"^\s*\*\*(?:Validated Findings:?|Answer:\s*Supported:?)\*\*",
        rendered,
        re.I,
    ))
    if has_validated_findings and rendered and not already_prefixed:
        rendered = f"{VALIDATED_FINDINGS_HEADING}\n\n{rendered}"
    return rendered


class OutputQueryType(str, Enum):
    SIMPLE_FACTUAL = "simple_factual"
    MULTI_DOCUMENT = "multi_document_synthesis"
    COMPARISON = "comparison"
    TEMPORAL = "temporal_currentness"
    MANAGEMENT = "management_actionable"


class OutputFormatter:
    """Select presentation without changing validated claim or provenance data."""

    # Explicit requests for observed evidence describe the desired answer form,
    # even when their subject contains domain verbs such as "control efforts".
    _FACTUAL_EVIDENCE = re.compile(
        r"\b(?:what evidence|quantitative (?:results?|evidence|data)|"
        r"how (?:much|many)|show(?:s|ed|ing)? that)\b",
        re.I,
    )
    _COMPARISON = re.compile(r"\b(?:compar\w*|differ\w*|contrast\w*|versus|vs\.?)\b", re.I)
    _TEMPORAL = re.compile(r"\b(?:current|latest|newest|most recent|as of|revised|updated|supersed\w*)\b", re.I)
    _MANAGEMENT = re.compile(r"\b(?:manage\w*|recommend\w*|action plan|implement\w*|mitigat\w*|control\w*|priorit\w*|next steps?)\b", re.I)

    def __init__(self, citation: Callable[[KnowledgeArtifact], str]) -> None:
        self.citation = citation

    def classify(self, query: str, sources: Sequence[Sequence[KnowledgeArtifact]]) -> OutputQueryType:
        if self._FACTUAL_EVIDENCE.search(query):
            return OutputQueryType.SIMPLE_FACTUAL
        if self._COMPARISON.search(query):
            return OutputQueryType.COMPARISON
        if self._TEMPORAL.search(query):
            return OutputQueryType.TEMPORAL
        if self._MANAGEMENT.search(query):
            return OutputQueryType.MANAGEMENT
        document_ids = {source.document_id for group in sources for source in group}
        return (OutputQueryType.MULTI_DOCUMENT if len(document_ids) > 1
                else OutputQueryType.SIMPLE_FACTUAL)

    _LIST_MARKER = re.compile(r"^\s*(?:\d{1,3}[.)]|[-*+])\s+")

    def _claim_line(self, claim: Claim, sources: Sequence[KnowledgeArtifact],
                    prefix: str = "", *, strip_list_marker: bool = False) -> str:
        citations = " ".join(dict.fromkeys(self.citation(source) for source in sources))
        # The validated claim remains unchanged in memory and provenance logs.
        # Remove only an LLM-supplied presentation marker when this formatter
        # owns the surrounding Markdown list.
        text = (self._LIST_MARKER.sub("", claim.text, count=1)
                if strip_list_marker else claim.text)
        return f"{prefix}{text} {citations}".rstrip()

    _CATEGORY_REQUEST = re.compile(
        r"(?P<items>[A-Za-z][A-Za-z-]*(?:\s*,\s*[A-Za-z][A-Za-z-]*)*"
        r"\s*,?\s+(?:and|or)\s+[A-Za-z][A-Za-z-]*)\s+"
        r"(?:solutions?|approaches?|strategies|recommendations?|actions?|options?|measures?)\b",
        re.I,
    )
    _METADATA_AUDIT = re.compile(
        r"\b(?:metadata|date|lifecycle|version)\s+audit\b|"
        r"\baudit\s+(?:the\s+)?(?:metadata|dates?|lifecycle|version)\b|"
        r"\b(?:explain|identify|show|list)\s+(?:any\s+)?"
        r"(?:metadata|date|version)\s+(?:conflicts?|discrepancies)\b",
        re.I,
    )
    _METADATA_BOILERPLATE = re.compile(
        r"^(?:conflicting (?:publication|revision|effective) date|several document families|"
        r"the document's revision/version year differs|.*catalog metadata)",
        re.I,
    )
    _CATEGORY_SIGNALS = (
        (r"technolog|technical|engineer", r"technolog|engineer|treatment|biofilter|detention|"
         r"impoundment|infrastructure|monitor|mapping|baseline|interception|practice|restore"),
        (r"regulat|legal|policy|compliance", r"regulat|permit|section\s+40[14]|compliance|"
         r"enforc|requirement|avoidance|minimization|mitigation|review|swampbuster"),
        (r"market|financ|economic|funding", r"market|financ|fund|grant|incentive|easement|"
         r"payment|credit|cost.?share|assistance|NRCS|MRBI|NWQI|section\s+319"),
    )

    @classmethod
    def _requested_categories(cls, query: str) -> tuple[str, ...]:
        match = cls._CATEGORY_REQUEST.search(query)
        if not match:
            return ()
        items = re.sub(r",?\s+(?:and|or)\s+", ",", match.group("items"), flags=re.I)
        return tuple(dict.fromkeys(item.strip() for item in items.split(",") if item.strip()))

    @classmethod
    def _category_for_claim(cls, text: str, categories: Sequence[str]) -> str | None:
        normalized = text.casefold()
        scored = []
        for position, category in enumerate(categories):
            terms = [term.casefold() for term in re.findall(r"[A-Za-z][A-Za-z-]*", category)
                     if term.casefold() not in {"based", "related"}]
            score = sum(bool(re.search(r"\b" + re.escape(term) + r"\w*\b", normalized))
                        for term in terms)
            for category_pattern, evidence_pattern in cls._CATEGORY_SIGNALS:
                if re.search(category_pattern, category, re.I):
                    score += len(re.findall(evidence_pattern, text, re.I))
            scored.append((score, -position, category))
        best = max(scored, default=(0, 0, None))
        return best[2] if best[0] else None

    @classmethod
    def _missing_block(cls, response: SynthesisResponse, query: str | None) -> str:
        facets = list(response.unsupported_facets)
        if not query or not cls._METADATA_AUDIT.search(query):
            facets = [facet for facet in facets if not cls._METADATA_BOILERPLATE.search(facet)]
        return format_evidence_gaps(facets)

    def _comparison(self, claims, sources) -> str:
        groups = defaultdict(list)
        titles = {}
        for claim, claim_sources in zip(claims, sources, strict=True):
            ids = tuple(dict.fromkeys(source.document_id for source in claim_sources))
            groups[ids].append((claim, claim_sources))
            for source in claim_sources:
                titles[source.document_id] = source.title
        sections = []
        for ids, items in groups.items():
            heading = " + ".join(f"{doc_id}: {titles[doc_id]}" for doc_id in ids)
            sections.append(f"### {heading}\n\n" + "\n".join(
                self._claim_line(claim, claim_sources, "- ")
                for claim, claim_sources in items))
        return "\n\n".join(sections)

    def _management(self, query, claims, sources) -> str:
        categories = self._requested_categories(query)
        lines = []
        previous_category = object()
        for index, (claim, claim_sources) in enumerate(
                zip(claims, sources, strict=True), 1):
            category = self._category_for_claim(claim.text, categories)
            if category is not None and category != previous_category:
                if lines:
                    lines.append("")
                lines.extend([f"### {category}", ""])
            lines.append(self._claim_line(
                claim, claim_sources, f"{index}. ", strip_list_marker=True
            ))
            previous_category = category
        return "\n".join(lines)
    def format(self, query: str | None, response: SynthesisResponse,
               sources: Sequence[Sequence[KnowledgeArtifact]]) -> str:
        if response.status is SynthesisStatus.VALIDATION_FAILED:
            return "The generated answer could not be verified against the retrieved source text."
        if response.status is SynthesisStatus.SYSTEM_FALLBACK and not response.claims:
            return "The synthesis service is unavailable. Please try again or review the retrieved sources."
        missing = self._missing_block(response, query)
        if not response.claims:
            lead = "The corpus does not provide enough evidence to answer that question reliably."
            return f"{lead}\n\n{missing}" if missing else lead
        if query is None:
            body = "\n".join(self._claim_line(claim, group, "- ")
                             for claim, group in zip(response.claims, sources, strict=True))
        else:
            kind = self.classify(query, sources)
            if kind is OutputQueryType.COMPARISON:
                body = self._comparison(response.claims, sources)
            elif kind is OutputQueryType.MANAGEMENT:
                body = self._management(query, response.claims, sources)
            elif kind is OutputQueryType.TEMPORAL:
                body = self._claim_line(response.claims[0], sources[0])
                remainder = "\n".join(self._claim_line(claim, group, "- ")
                                      for claim, group in zip(response.claims[1:], sources[1:], strict=True))
                if remainder:
                    body += "\n\n**Publication and planning evidence**\n\n" + remainder
            elif kind is OutputQueryType.MULTI_DOCUMENT:
                body = "\n".join(self._claim_line(claim, group, "- ")
                                 for claim, group in zip(response.claims, sources, strict=True))
            else:
                body = "\n\n".join(self._claim_line(claim, group, "- ")
                                   for claim, group in zip(response.claims, sources, strict=True))
        supported = f"{VALIDATED_FINDINGS_HEADING}\n\n{body}"
        rendered = f"{supported}\n\n{missing}" if missing else supported
        return normalize_user_facing_answer(
            rendered, has_validated_findings=True
        )
