"""Adaptive rendering of claims that already passed provenance validation."""
from __future__ import annotations

from collections import defaultdict
from enum import Enum
import re
from typing import Callable, Sequence

from data_models import Claim, KnowledgeArtifact, SynthesisResponse, SynthesisStatus


class OutputQueryType(str, Enum):
    SIMPLE_FACTUAL = "simple_factual"
    MULTI_DOCUMENT = "multi_document_synthesis"
    COMPARISON = "comparison"
    TEMPORAL = "temporal_currentness"
    MANAGEMENT = "management_actionable"


class OutputFormatter:
    """Select presentation only; never add, remove, or rewrite a claim."""

    _COMPARISON = re.compile(r"\b(?:compar\w*|differ\w*|contrast\w*|versus|vs\.?)\b", re.I)
    _TEMPORAL = re.compile(r"\b(?:current|latest|newest|most recent|as of|revised|updated|supersed\w*)\b", re.I)
    _MANAGEMENT = re.compile(r"\b(?:manage\w*|recommend\w*|action plan|implement\w*|mitigat\w*|control\w*|priorit\w*|next steps?)\b", re.I)

    def __init__(self, citation: Callable[[KnowledgeArtifact], str]) -> None:
        self.citation = citation

    def classify(self, query: str, sources: Sequence[Sequence[KnowledgeArtifact]]) -> OutputQueryType:
        if self._COMPARISON.search(query):
            return OutputQueryType.COMPARISON
        if self._TEMPORAL.search(query):
            return OutputQueryType.TEMPORAL
        if self._MANAGEMENT.search(query):
            return OutputQueryType.MANAGEMENT
        document_ids = {source.document_id for group in sources for source in group}
        return (OutputQueryType.MULTI_DOCUMENT if len(document_ids) > 1
                else OutputQueryType.SIMPLE_FACTUAL)

    def _claim_line(self, claim: Claim, sources: Sequence[KnowledgeArtifact], prefix: str = "") -> str:
        citations = " ".join(dict.fromkeys(self.citation(source) for source in sources))
        return f"{prefix}{claim.text} {citations}".rstrip()

    @staticmethod
    def _missing_block(response: SynthesisResponse) -> str:
        if not response.unsupported_facets:
            return ""
        return "**Unsupported facets**\n\n*Status: INSUFFICIENT_EVIDENCE*\n\n" + "\n".join(
            f"- {facet}" for facet in response.unsupported_facets)

    def _comparison(self, claims, sources) -> str:
        groups = defaultdict(list)
        titles = {}
        for claim, claim_sources in zip(claims, sources, strict=True):
            ids = tuple(dict.fromkeys(source.document_id for source in claim_sources))
            key = ids[0] if len(ids) == 1 else "Cross-document findings"
            groups[key].append((claim, claim_sources))
            if len(ids) == 1:
                titles[key] = claim_sources[0].title
        sections = []
        for key, items in groups.items():
            heading = key if key == "Cross-document findings" else f"{key}: {titles[key]}"
            sections.append(f"### {heading}\n\n" + "\n".join(
                self._claim_line(claim, claim_sources, "- ")
                for claim, claim_sources in items))
        return "\n\n".join(sections)

    @staticmethod
    def _action_category(text: str) -> str:
        categories = (
            ("Prevention", r"prevent|avoid|biosecurity|education"),
            ("Monitoring", r"monitor|survey|detect|assess|track"),
            ("Control", r"control|remove|harvest|treat|eradicate"),
            ("Restoration", r"restore|rehabilitat|revegetat|habitat"),
            ("Coordination", r"coordinat|partner|agency|stakeholder|communicat"),
        )
        return next((label for label, pattern in categories
                     if re.search(pattern, text, re.I)), "General actions")

    def _management(self, claims, sources) -> str:
        grouped = defaultdict(list)
        for claim, claim_sources in zip(claims, sources, strict=True):
            grouped[self._action_category(claim.text)].append((claim, claim_sources))
        lines = ["**Recommendations by category**"]
        for category, items in grouped.items():
            lines.extend(["", f"*{category}*", *[
                self._claim_line(claim, claim_sources, "- ")
                for claim, claim_sources in items]])
        lines.extend(["", "**Implementation sequence**", ""])
        lines.extend(self._claim_line(claim, claim_sources, f"{index}. ")
                     for index, (claim, claim_sources) in enumerate(
                         zip(claims, sources, strict=True), 1))
        return "\n".join(lines)

    def format(self, query: str | None, response: SynthesisResponse,
               sources: Sequence[Sequence[KnowledgeArtifact]]) -> str:
        if response.status is SynthesisStatus.VALIDATION_FAILED:
            return "The generated answer could not be verified against the retrieved source text."
        if response.status is SynthesisStatus.SYSTEM_FALLBACK and not response.claims:
            return "The synthesis service is unavailable. Please try again or review the retrieved sources."
        missing = self._missing_block(response)
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
                body = self._management(response.claims, sources)
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
        return f"{body}\n\n{missing}" if missing else body
