"""Local Wiki sentence and relationship rules over verbatim corpus evidence.

Aliases identify curated entities, never supply facts. Relationships require a
predicate in a single source sentence; entity co-occurrence alone is insufficient.
"""
from __future__ import annotations

from functools import lru_cache
import re

from database import WIKI_ENTITY_CANDIDATES


ALIASES = {
    "U.S. Army Corps of Engineers": (
        "USACE", "U.S. Army Corp of Engineers", "Army Corps of Engineers",
    ),
    "U.S. Fish and Wildlife Service": ("USFWS", "Fish and Wildlife Service"),
    "U.S. Department of the Interior": ("DOI", "Department of the Interior"),
    "Missouri Department of Conservation": ("MDC",),
    "Forest": ("Forests", "Forestlands"),
    "Marsh": ("Marshes",),
    "Wetland": ("Wetlands",),
    "Zebra mussel": ("Zebra mussels",),
    "Invasive carp": ("Asian carp",),
}
ENTITY_TYPES = {name: kind for kind, names in WIKI_ENTITY_CANDIDATES.items() for name in names}


@lru_cache(maxsize=128)
def entity_pattern(name: str) -> str:
    variants = []
    for alias in (name, *ALIASES.get(name, ())):
        pattern = r"\s+".join(re.escape(word) for word in alias.split())
        pattern = pattern.replace(r"U\.S\.", r"U\.?\s*S\.?")
        variants.append(pattern)
    return r"(?<!\w)(?:" + "|".join(variants) + r")(?!\w)"


def mentions(text: str, name: str) -> bool:
    return bool(re.search(entity_pattern(name), text, re.I))


def source_sentences(text: str) -> list[str]:
    """Split without severing U.S., author initials or common abbreviations.

    Offsets slice the original string; no replacement enters quoted evidence.
    """
    start = 0
    result = []
    for match in re.finditer(r"(?<=[.!?])\s+|\n\s*\n", text):
        prefix = text[start:match.start()]
        if re.search(r"(?:\b[A-Z]\.|\b(?:e\.g|i\.e|Dr|Mr|Ms|Mrs|St|Jr|Sr|vs|al|Fig|No|Vol|pp)\.)$", prefix):
            continue
        span = prefix.strip()
        if span:
            result.append(span)
        start = match.end()
    if text[start:].strip():
        result.append(text[start:].strip())
    return result


def span_score(span: str, topic: str) -> int:
    if not mentions(span, topic):
        return -50
    score = 10
    subject = entity_pattern(topic)
    if re.search(r"^(?:The\s+)?" + subject + r"(?:\s*\([A-Z]+\))?\s+(?:(?:is|are)\s+|a collective term\b)", span, re.I):
        score += 12
    if re.search(r"^(?:The\s+)?" + subject, span, re.I):
        score += 3
    if re.search(r"\b(?:include\w*|threat\w*|native|habitat|feed|compete|control|remov\w*|spread|harm\w*|risk|disrupt\w*)\b", span, re.I):
        score += 4
    if ENTITY_TYPES.get(topic) == "Agency":
        if re.search(subject + r"(?:\s*\([A-Z]+\))?\s+(?:currently\s+)?(?:manages?|operates?|maintains?|protects?|conserves?|oversees?|administers?|evaluates?|conducts?|provides?|supports?|serves?|owns?|works?)\b", span, re.I):
            score += 14
        if re.search(subject + r"(?:\s*\([A-Z]+\))?\s+(?:is\s+(?:the principal federal agency|responsible for|authorized)|has a mission to)", span, re.I):
            score += 20
        if re.search(r"research activities performed by\s+(?:the\s+)?" + subject, span, re.I):
            score += 20
        if re.search(r"\b(?:mission|responsib\w*|research|restor\w*|partner\w*|coordinat\w*|fund\w*|permit\w*)\b", span, re.I):
            score += 4
    if re.search(r"https?\s*:\s*/|\b(?:h\s+ttps|www\s*\.)|\.(?:org|gov|com)/|\.pdf\b|\.{2,}|\b(?:references cited|accessed|photo(?:graph)?(?:s)? by|photo credit)\b", span, re.I):
        score -= 35
    if re.search(r"\b(?:19|20)\d{2}[a-z]?,\s|\b(?:doi|ISBN)\b|\b[A-Z]\.(?:[A-Z]\.)?,|\b(?:v\.|pp\.)\s*\d", span):
        score -= 25
    if (re.search(r"(?:\b[A-Z]{2,}\b\s+){3}|\bX(?:\s+X){2,}", span)
            or re.search(r"\b(?:table of contents|acronyms|abbreviations|PREFACE|Page \d+ \|)", span, re.I)):
        score -= 12
    if re.search(r"\b(?:Image Details|Front cover|Open-File Report|Contact\s+U\.?S|Source:|Prepared by)", span, re.I):
        score -= 20
    if len(re.findall("•", span)) > 1 or not re.search(r"[.!?][\"’')]*$", span):
        score -= 10
    if len(span) > 450:
        score -= 3
    return score


def _entity_sentence(span: str):
    # Longest matches prevent 'Missouri' inside MDC and 'Forest' inside a name
    # from turning the agency's own name into a second related entity.
    found = []
    for name in ENTITY_TYPES:
        for match in re.finditer(entity_pattern(name), span, re.I):
            found.append((match.start(), match.end(), name))
    selected = []
    for item in sorted(found, key=lambda v: (-(v[1] - v[0]), v[0])):
        if not any(item[0] < other[1] and other[0] < item[1] for other in selected):
            selected.append(item)
    names, parts, pos = {}, [], 0
    for start, end, name in sorted(selected):
        token = f"E{len(names)}Z"
        names[token] = name
        parts.extend((span[pos:start], token))
        pos = end
    parts.append(span[pos:])
    text = "".join(parts)
    for left, name in names.items():
        for right, other in names.items():
            if left != right and name == other:
                text = re.sub(rf"{left}\s*\({right}\)", left, text)
    return text, names


def explicit_relationships(span: str, topic: str) -> list[tuple[str, str]]:
    """Return directional edges proved by bounded predicates, not a keyword bag."""
    if re.search(r"\b(?:no|not|never|neither|except|without|unrelated|independent(?:ly)?|if|unless|separate(?:ly)?|hypothetical)\b", span, re.I):
        return []
    # Never bridge two independently asserted sentences.
    edges = []
    for sentence in source_sentences(span):
        text, names = _entity_sentence(sentence)
        gap = r"(?:(?!E\d+Z|\b(?:while|whereas|but|and|which|who|although)\b)[^.;!?]){0,140}?"
        listing = r"(?:(?!E\d+Z|\b(?:while|whereas|but|which|who|although)\b)[^.;!?]){0,140}?"
        for left, source in names.items():
            for right, target in names.items():
                if left == right or source == target or topic not in (source, target):
                    continue
                source_type, target_type = ENTITY_TYPES[source], ENTITY_TYPES[target]
                rules = []
                if source_type == "Species":
                    if target_type == "Species":
                        rules.append((rf"{left},?\s+(?:include\w*|such as|(?:is\s+)?a collective term for)\s+{listing}{right}", "includes", "included in"))
                    if target_type in ("Location", "Habitat"):
                        rules.append((rf"{left}\s+(?:(?:are|is)\s+)?(?:found|present|occur\w*|live\w*)\s+in\s+(?:the\s+)?{right}", "occurs in", "habitat or range of"))
                    if target_type == "Location":
                        rules.append((rf"contained\s+{left}\s+within established ranges, preventing their spread into the\s+{right}", "spread prevention protects", "protected from spread of"))
                if source_type == "Agency":
                    if target_type == "Agency":
                        rules.append((rf"{left},?\s+in\s+(?:cooperation|partnership|collaboration|conjunction)\s+with\s+(?:the\s+)?{right}", "works with", "works with"))
                        rules.append((rf"{left}\s+co-led,\s+with\s+{listing}{right},\s+the formation\b", "works with", "works with"))
                        rules.append((rf"{left}(?:\s*\([A-Z]+\))?\s+(?:has\s+|is\s+)?(?:partner\w*|work\w*|collaborat\w*|cooperat\w*|coordinat\w*)\s+(?:closely\s+)?with\s+(?:the\s+)?{right}", "works with", "works with"))
                        rules.append((rf"{left}(?:\s*\([A-Z]+\))?,?\s+and\s+(?:the\s+)?{right}(?:\s*\([A-Z]+\))?\s+(?:have\s+|jointly\s+)?(?:partner\w*|collaborat\w*|cooperat\w*|conduct\w*|work\w*)\b", "works with", "works with"))
                    if target_type in ("Species", "Habitat", "Threat"):
                        rules.append((rf"{left}(?:\s*\([A-Z]+\))?\s+(?:currently\s+)?(?:manages?|controls?|monitors?|protects?|restores?|conserves?|supports?|conducts?|prevents?|evaluates?|serves?)\s+{gap}{right}", "documents work on", "management or research by"))
                        rules.append((rf"{left}\s+is the principal federal agency tasked with providing information\s+{listing}{right}", "provides information on", "information provided by"))
                        rules.append((rf"{left}\s+(?:serves as a national leader in combating|has unique responsibilities and capabilities to manage)\s+{right}", "documents work on", "management or research by"))
                    if target_type == "Location":
                        # A named habitat can be the managed object before 'in'.
                        nonobjects = "|".join(token for token, name in names.items() if ENTITY_TYPES[name] != "Habitat")
                        location_gap = rf"(?:(?!{nonobjects}|\b(?:while|whereas|but|and|which|who)\b)[^.;!?]){{0,140}}?"
                        rules.append((rf"{left}(?:\s*\([A-Z]+\))?\s+(?:owns and\s+)?(?:manages?|operates?|maintains?)\s+{location_gap}\bin\s+(?:the\s+)?{right}", "manages resources in", "resource management by"))
                if source_type in ("Threat", "Species") and target_type in ("Habitat", "Species"):
                    rules.append((rf"{left}\s+(?:can\s+|may\s+)?(?:threaten\w*|harm\w*|affect\w*|disrupt\w*)\s+(?:native\s+)?{right}", "can affect", "can be affected by"))
                if source_type == "Habitat" and target_type == "Location":
                    rules.append((rf"{left}\s+(?:are\s+)?(?:found|occur\w*|present)\s+in\s+{right}", "occurs in", "contains habitat"))
                if source_type == "Habitat" and target_type == "Species":
                    rules.append((rf"{left}\s+(?:provide\w*|offer\w*)\s+habitat\s+for\s+{right}", "provides habitat for", "uses habitat"))
                if source_type == "Habitat" and target_type == "Habitat":
                    rules.append((rf"{left}\s+(?:types\s+)?include\w*\s+{listing}{right}", "includes habitat type", "type of habitat"))
                if source_type == "Threat" and target_type == "Threat":
                    rules.append((rf"{left}\s+(?:is a particularly challenging threat because of the ways in which it\s+)?may interact with\s+(?:other threats such as\s+)?{right}", "may interact with", "may interact with"))
                for pattern, forward, reverse in rules:
                    if re.search(pattern, text, re.I):
                        edges.append((target, forward) if source == topic else (source, reverse))
                        break
        # Explicit coordination/partnership lists connect their named agencies.
        coordination = re.search(r"\b(?:coordination with|partnership between|partners with)\s+(.{1,260}?)(?=\s+(?:through|that|on|to)\b|[.;]|$)", text, re.I)
        if coordination:
            participants = list(dict.fromkeys(names[token] for token in re.findall(r"E\d+Z", coordination[1]) if ENTITY_TYPES[names[token]] == "Agency"))
            if topic in participants and not re.search(r"\b(?:proposed|potential|could|would|may)\b", sentence, re.I):
                edges.extend((name, "named coordination partner") for name in participants if name != topic)
    return list(dict.fromkeys(edges))
