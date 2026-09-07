"""Inspectable, corpus-bounded document applicability and temporal answers.

Lifecycle decisions use persisted evidence, never model-generated relationships.
Non-temporal retrieval keeps its existing ranking. No global recency multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import calendar
import re

from data_models import KnowledgeArtifact
from validator import _citation


@dataclass(frozen=True)
class TemporalIntent:
    mode: str = "none"
    years: tuple[str, ...] = ()
    as_of: str | None = None


def detect_temporal_intent(question: str) -> TemporalIntent:
    text = question.casefold()
    text = re.sub(r"\b(?:electric(?:al)?|water|ocean|river) current\b|\bcurrent (?:velocity|speed|flow|strength|density|amperage)\b", "physical flow", text)
    years = tuple(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", text)))
    as_of = re.search(r"\bas of\s+((?:19|20)\d{2}(?:-\d{2}(?:-\d{2})?)?)", text)
    if re.search(r"changed? over time|compare (?:the )?old and new|previous (?:vs\.?|versus|and) current|how (?:has|have|was|were|did).{0,100}(?:chang|revis)|has (?:this|that|the) .{0,50}changed|变化|修订前后|新旧对比", text):
        mode = "comparison"
    elif re.search(r"\b(?:current|currently|still|recommended now|recommendations? now)\b|当前|现行|现在.*建议", text):
        mode = "current"
    elif re.search(r"\b(?:latest|newest|most recent|up.to.date)\b|最新", text):
        mode = "latest"
    elif re.search(r"\b(?:earlier|older|historical|originally|original|previous guidance|previous recommendation|as of)\b|旧版|以前|历史", text):
        mode = "historical"
    elif years and re.search(r"\b(?:guidance|strateg(?:y|ies)|recommendations?|reports?|plans?|amendments?|supplements?|versions?|editions?)\b", text) and not re.search(r"\bfy\s*\d{4}", text):
        mode = "comparison" if len(years) > 1 and re.search(r"compare|between|from", text) else "historical"
    elif re.search(r"\b(?:updated|revised)\s+(?:guidance|recommendations?|strategy|report)\b", text):
        mode = "current"
    else:
        mode = "none"
    return TemporalIntent(mode, years, as_of[1] if as_of else None)


_STOP = set("what which how has have was were did does do is are the a an of for to in on and or from with about at by as still this that these those it they them their tell me please provide find say says recommend recommends recommended recommendation recommendations guidance strategy strategies plan plans report reports document documents management current currently latest newest recent most earlier older previous original historical changed change changes over time revised revision updated update compare between now effective applicable version versions final draft control".split())
_STOP.update("topic applied apply applies said discuss discussed discusses used use available known present strongest methods method evidence quantitative data whether regarding corpus effectiveness".split())
_STOP.update("briefly concise short detailed detail summary summarize explain give answer old new".split())


def _terms(question):
    return list(dict.fromkeys(word for word in re.findall(r"[a-z][a-z'-]+", question.casefold())
                              if word not in _STOP and len(word) > 2))


def _date_start(value):
    if not value:
        return None
    try:
        return date.fromisoformat(str(value) + {4: "-01-01", 7: "-01"}.get(len(str(value)), ""))
    except ValueError:
        return None


def _document_date(document):
    for field in ("effective_date", "revision_date", "publication_date"):
        value = _date_start(document.get(field))
        if value:
            return value
    return None


def _active(document, when):
    effective = _date_start(document.get("effective_date"))
    published = _date_start(document.get("publication_date"))
    revised = _date_start(document.get("revision_date"))
    value = str(document.get("effective_date") or "")
    effective_end = effective
    if effective and len(value) == 4:
        effective_end = date(effective.year, 12, 31)
    elif effective and len(value) == 7:
        effective_end = date(effective.year, effective.month, calendar.monthrange(effective.year, effective.month)[1])
    return document.get("status") not in {"withdrawn", "draft"} and not (
        effective_end and effective_end > when or
        effective and effective > when or published and published > when or revised and revised > when)


def verified_relationships(store, index):
    """Recheck every edge against canonical text before it affects an answer."""
    result = []
    for doc_id, document in index.items():
        for relation in document.get("relations", []):
            evidence = relation.get("evidence") or {}
            if relation.get("source") != "explicit" or relation.get("target_id") not in index:
                continue
            if evidence.get("document_id") != doc_id or not evidence.get("exact_span"):
                continue
            rows = store.connection.execute(
                "SELECT original_text_chunk FROM knowledge_artifacts WHERE document_id=? AND page_number=?",
                (doc_id, evidence.get("page_number")),
            ).fetchall()
            if any(evidence["exact_span"] in row[0] for row in rows):
                result.append({**relation, "document_id": doc_id, "source_id": doc_id})
    return result


def _doc_artifacts(store, document_ids):
    if not document_ids:
        return []
    placeholders = ",".join("?" for _ in document_ids)
    rows = store.connection.execute(
        f"SELECT artifact_id FROM knowledge_artifacts WHERE document_id IN ({placeholders}) ORDER BY artifact_id",
        list(document_ids),
    ).fetchall()
    return store._artifacts_by_ranked_ids([row[0] for row in rows])


def select_temporal_evidence(question, store, *, top_k=5, intent=None,
                             anchor_document_ids=(), as_of=None):
    from document_lifecycle import get_lifecycles
    intent = intent or detect_temporal_intent(question)
    policy_mode = "current" if intent.as_of else intent.mode
    if top_k < 1:
        raise ValueError("top_k must be positive")
    when = _date_start(as_of or intent.as_of) or date.today()
    index = get_lifecycles(store)
    terms = _terms(question)
    clean_query = " ".join(terms) or question
    candidates = store.retrieve(None, max(20, top_k * 8), method="keyword", query_text=clean_query)
    rank = {}
    best = {}
    for artifact in candidates:
        text = (artifact.title + " " + artifact.original_text_chunk).casefold()
        covered = sum(term in text for term in terms)
        if terms and covered < min(2, len(terms)):
            continue
        if artifact.document_id not in index:
            continue
        rank.setdefault(artifact.document_id, len(rank))
        best.setdefault(artifact.document_id, artifact)
    anchors = [doc_id for doc_id in anchor_document_ids if doc_id in index]
    # A named report/ID is an anchor as well; it does not force its old version.
    anchors += [doc_id for doc_id in index if re.search(r"\b" + re.escape(doc_id) + r"\b", question, re.I)]
    families = {index[doc_id]["family_id"] for doc_id in anchors}
    if families:
        seeds = {doc_id for doc_id, doc in index.items() if doc["family_id"] in families}
    else:
        seeds = set(rank)
        # Metadata lookup recovers a revised report omitted by chunk top K.
        seeds.update(doc_id for doc_id, doc in index.items() if terms and
                     all(term in doc.get("title", "").casefold() for term in terms))
        if not terms:
            seeds = set(index)
        families = {index[doc_id]["family_id"] for doc_id in seeds}
        seeds.update(doc_id for doc_id, doc in index.items() if doc["family_id"] in families)
    edges = verified_relationships(store, index)
    edges = [edge for edge in edges if edge["document_id"] in seeds and edge["target_id"] in seeds]
    uncertainties = []
    if not terms and not anchors:
        uncertainties.append("The question does not identify a unique document family; sources are shown by their available status, without claiming a corpus-wide authority.")
    if len({index[doc_id]["family_id"] for doc_id in seeds}) > 1:
        uncertainties.append("Several document families are relevant. A newer document from another family or issuer does not establish replacement or greater authority.")
    for doc_id in sorted(seeds):
        effective = str(index[doc_id].get("effective_date") or "")
        if len(effective) in {4, 7} and _date_start(effective) <= when and not _active(index[doc_id], when):
            uncertainties.append(f"{doc_id}'s effective date is known only as {effective}; applicability within that period is uncertain, so its replacement relationships are not yet activated.")

    active_edges = [edge for edge in edges if _active(index[edge["document_id"]], when)]
    if policy_mode == "comparison" and intent.years:
        active_edges = [edge for edge in active_edges if not _document_date(index[edge["document_id"]]) or
                        str(_document_date(index[edge["document_id"]]).year) <= max(intent.years)]
    for family in families:
        members = {doc_id for doc_id in seeds if index[doc_id]["family_id"] == family}
        if len(members) > 1 and not any(edge["document_id"] in members for edge in active_edges):
            uncertainties.append("Similar titles or different dates suggest candidate versions, but no explicit replacement relationship establishes which is current.")
    full_replacements = {edge["target_id"]: edge["document_id"] for edge in active_edges
                         if edge["type"] in {"supersedes", "replaces", "withdraws"}}
    revisions = {edge["target_id"]: edge["document_id"] for edge in active_edges if edge["type"] == "revision_of"}
    # Cycles are ambiguous, never grounds for selecting a supposedly final node.
    conflicted = set()
    for origin in list(full_replacements):
        seen, current = set(), origin
        while current in full_replacements and current not in seen:
            seen.add(current)
            current = full_replacements[current]
        if current in seen:
            conflicted.update(seen)
            uncertainties.append("Conflicting replacement relationships form a cycle; currentness cannot be established for that chain.")
            for doc_id in seen:
                full_replacements.pop(doc_id, None)
    eligible = list(seeds)
    if policy_mode in {"current", "latest"}:
        withdrawn = {edge["target_id"] for edge in active_edges if edge["type"] == "withdraws"}
        applicable = [doc_id for doc_id in eligible if _active(index[doc_id], when) and doc_id not in withdrawn]
        if applicable:
            eligible = applicable
        elif eligible:
            uncertainties.append("No non-draft, non-withdrawn source with an applicable known date was found; the remaining sources are not established current guidance.")
    elif policy_mode == "historical":
        if intent.as_of:
            eligible = [doc_id for doc_id in eligible if not _document_date(index[doc_id]) or _document_date(index[doc_id]) <= when]
        elif intent.years:
            eligible = [doc_id for doc_id in eligible if any(
                str(index[doc_id].get(field) or "").startswith(year)
                for field in ("publication_date", "revision_date", "effective_date") for year in intent.years)]
    elif policy_mode == "comparison" and intent.years:
        # Include endpoints and intervening versions; do not reduce to newest.
        low, high = min(intent.years), max(intent.years)
        eligible = [doc_id for doc_id in eligible if not _document_date(index[doc_id]) or
                    low <= str(_document_date(index[doc_id]).year) <= high]

    guidance_query = bool(re.search(r"guidance|recommendation|strategy|recommended|指导|建议|策略", question, re.I))
    def ordering(doc_id):
        doc = index[doc_id]
        known_date = _document_date(doc)
        stamp = known_date.toordinal() if known_date else 0
        relevance = rank.get(doc_id, len(rank))
        if policy_mode in {"historical", "comparison"}:
            return (stamp or 9999999, relevance, doc_id)
        if policy_mode == "latest":
            return (-stamp, relevance, doc_id)
        role = 0 if doc.get("kind") in {"guidance", "strategy"} else 1
        # Document replacement > final status > relevance. Recency only breaks
        # ties within a family below; there is no universal newest-is-authority.
        status = 0 if doc.get("status") == "final" else 1
        replaced = 1 if doc_id in full_replacements else 0
        revised = 1 if doc_id in revisions else 0
        return (replaced, role if guidance_query and policy_mode == "current" else 0, status, revised, relevance, doc_id)

    eligible.sort(key=ordering)
    if policy_mode in {"current", "latest"}:
        # Sort members only within their current family positions, preserving
        # relevance between unrelated families. Latest-report intent may use dates.
        positions = {}
        for i, doc_id in enumerate(eligible):
            positions.setdefault(index[doc_id]["family_id"], []).append(i)
        for places in positions.values():
            members = [eligible[i] for i in places]
            members.sort(key=lambda doc_id: (
                doc_id in full_replacements, not _active(index[doc_id], when),
                index[doc_id].get("status") != "final", doc_id in revisions,
                -(_document_date(index[doc_id]).toordinal() if _document_date(index[doc_id]) else 0),
                rank.get(doc_id, len(rank)), doc_id))
            for i, doc_id in zip(places, members):
                eligible[i] = doc_id
    selected = eligible[:top_k]
    # A partial update requires its base; reserve room rather than marking it obsolete.
    for edge in active_edges:
        if policy_mode in {"current", "latest"} and edge["type"] in {"amendment_of", "supplement_to"} and (
            edge["document_id"] in selected or edge["target_id"] in selected):
            for doc_id in (edge["target_id"], edge["document_id"]):
                if doc_id not in selected:
                    selected.append(doc_id)
    selected_edges = [edge for edge in active_edges if edge["document_id"] in selected or edge["target_id"] in selected]
    if intent.mode == "comparison" and intent.years:
        selected_edges = [edge for edge in selected_edges if not _document_date(index[edge["document_id"]]) or
                          str(_document_date(index[edge["document_id"]]).year) <= max(intent.years)]
    all_evidence = _doc_artifacts(store, list(dict.fromkeys(selected + [edge["document_id"] for edge in selected_edges])))
    for doc_id in selected:
        possible = [a for a in all_evidence if a.document_id == doc_id]
        if possible:
            best[doc_id] = max(possible, key=lambda a: sum(term in a.original_text_chunk.casefold() for term in terms))
    artifacts = [best[doc_id] for doc_id in selected if doc_id in best]
    selected = [a.document_id for a in artifacts]
    decisions = []
    for doc_id in selected:
        doc = index[doc_id]
        if doc_id in conflicted:
            reason = "Conflicting replacement relationships prevent establishing a unique current version."
        elif doc_id in full_replacements:
            reason = f"Explicitly replaced/withdrawn by {full_replacements[doc_id]}; retained as historical context."
        elif doc_id in revisions:
            reason = f"A documented revision exists in {revisions[doc_id]}; complete replacement is not established."
        elif not _active(doc, when):
            reason = "Draft, withdrawn or future-dated source; not established applicable guidance."
        elif any(edge["document_id"] == doc_id and edge["type"] in {"supersedes", "replaces"} for edge in active_edges):
            reason = "Preferred within this family because explicit replacement evidence identifies this applicable successor."
        elif any(edge["document_id"] == doc_id and edge["type"] in {"amendment_of", "supplement_to"} for edge in active_edges):
            reason = "Partial update: read together with its base document."
        elif any(edge["document_id"] == doc_id and edge["type"] == "revision_of" for edge in active_edges):
            reason = "Documented revision; preferred for the revised material without assuming every earlier section is obsolete."
        else:
            reason = "Selected by topic, available status and within-family date; formal currentness is not established by recency alone."
        successor = full_replacements.get(doc_id) or revisions.get(doc_id)
        reason_edge = next((edge for edge in active_edges if edge["target_id"] == doc_id and
                            edge["document_id"] == successor and edge["type"] in (
                                {"supersedes", "replaces", "withdraws"} if doc_id in full_replacements else {"revision_of"})), None)
        decisions.append({"document_id": doc_id, "reason": reason,
                          "evidence": reason_edge["evidence"] if reason_edge else None})
        for warning in doc.get("warnings", []):
            for raw, readable in (("publication_date", "publication date"), ("revision_date", "revision date"),
                                  ("effective_date", "effective date"), ("metadata_inferred", "catalog metadata"),
                                  ("title_inferred", "title cue"), ("explicit", "document text")):
                warning = warning.replace(raw, readable)
            uncertainties.append(warning)
    return {"intent": intent.mode, "question": question, "as_of": when.isoformat(),
            "selected_document_ids": selected, "artifacts": artifacts,
            "documents": {doc_id: index[doc_id] for doc_id in selected},
            "decisions": decisions, "relationships": selected_edges,
            "uncertainties": list(dict.fromkeys(uncertainties)), "evidence": all_evidence}


def lifecycle_label(document):
    fields = []
    for key, label in (("publication_date", "Published"), ("revision_date", "Revised"), ("effective_date", "Effective")):
        if document.get(key):
            signal = document.get("dates", {}).get(key, {})
            origin = {"explicit": "document text", "metadata_inferred": "catalog metadata",
                      "title_inferred": "title cue"}.get(signal.get("source"), "source unknown")
            fields.append(f"{label} {document[key]} ({origin})")
    if document.get("version"):
        fields.append(f"Version {document['version']}")
    if document.get("status") not in {None, "unknown"}:
        fields.append(str(document["status"]).capitalize())
    return " · ".join(fields) or "Date and document status not established"


def is_lifecycle_assertion(text):
    """Reserve authoritative lifecycle wording for the verified local renderer."""
    # Keep the status predicate attached to a document/guidance subject. A broad
    # "report ... obsolete" search would wrongly reject ordinary statements such
    # as "The report says obsolete equipment should be replaced."
    subject = r"(?:guidance|guidelines?|recommendations?|documents?|reports?|strateg(?:y|ies)|plans?|polic(?:y|ies)|versions?|editions?|DOC\d+)"
    status_predicate = re.search(
        rf"\b{subject}\b(?:\s+(?:from|of|dated)\s+(?:19|20)\d{{2}})?\s+"
        r"(?:is|are|was|were|remains?|became|becomes?|(?:has|have|had)\s+(?:now\s+)?been)\s+"
        r"(?:(?:still|now|currently|already|formally|officially|fully|partially|not|considered|deemed|no longer)\s+)*"
        r"(?:replaced|withdrawn|obsolete|outdated|rescinded|revoked|superseded|current|valid|in force|in effect|up[- ]to[- ]date|latest|newest)\b",
        text, re.I,
    )
    return bool(status_predicate or re.search(
        r"\bsupersed\w*\b|\brescinds?\b|\b(?:current|latest|newest|most recent)\s+(?:(?:applicable|formal|official)\s+)?(?:guidance|recommendations?|version|report|strategy)\b"
        r"|\b(?:replaces?|replaced|amendment to|supplement to|revision of|revised version of)\b.{0,100}\b(?:guidance|document|report|strategy|plan|DOC\d+)\b",
        text, re.I))


def render_temporal_answer(selection, store):
    """Render verified lifecycle decisions and verbatim guidance excerpts.

    The application owns every temporal assertion; no LLM prose can invent a
    revision edge or place an unverified currentness claim in a preamble.
    """
    if not selection["artifacts"]:
        return ("No matching dated/versioned source was found for this temporal request in the indexed corpus.", "", [])
    lines, sources = [], []
    def cite(artifact):
        if artifact not in sources:
            sources.append(artifact)
        return _citation(artifact)
    terms = _terms(selection["question"])
    all_evidence = selection["evidence"]
    for artifact, decision in zip(selection["artifacts"], selection["decisions"]):
        doc = selection["documents"][artifact.document_id]
        proof = decision.get("evidence") or {}
        reason_source = next((a for a in all_evidence if a.document_id == proof.get("document_id") and
                              a.page_number == proof.get("page_number") and proof.get("exact_span") and
                              proof["exact_span"] in a.original_text_chunk), artifact)
        lines += [f"**{artifact.document_id} — {artifact.title}**", "",
                  f"{lifecycle_label(doc)}. {cite(artifact)}",
                  f"{decision['reason']} {cite(reason_source)}", ""]
        # Date/status statements link to the exact metadata evidence location.
        signals = list(doc.get("dates", {}).values()) + [doc.get("status_evidence"), doc.get("version_evidence")]
        metadata_spans = set()
        for signal in signals:
            if not isinstance(signal, dict):
                continue
            evidence = signal.get("evidence") or signal
            span = evidence.get("exact_span") or ""
            source = next((a for a in all_evidence if a.document_id == artifact.document_id and
                           a.page_number == evidence.get("page_number") and span and span in a.original_text_chunk), None)
            if source:
                if span in metadata_spans:
                    continue
                metadata_spans.add(span)
                lines.append(f'- Version/date evidence: “{span}” {cite(source)}')
        spans = []
        for candidate in all_evidence:
            if candidate.document_id != artifact.document_id:
                continue
            for span in re.split(r"(?<=[.!?])\s+|\n\s*\n", candidate.original_text_chunk):
                span = span.strip()
                if not 35 <= len(span) <= 750 or re.search(r"table of contents|references cited|https?://", span, re.I):
                    continue
                matches = sum(term in span.casefold() for term in terms)
                if terms and not matches:
                    continue
                action = bool(re.search(r"recommend|should|must|shall|priorit|control|protect|manage|harvest|restor|monitor", span, re.I))
                normative = bool(re.search(r"\b(?:recommend\w*|should|must|shall|priorit\w*)\b", span, re.I))
                spans.append((normative, matches, action, span, candidate))
        spans.sort(key=lambda item: (-item[0], -item[1], -item[2]))
        seen = set()
        for _, _, _, span, candidate in spans:
            if span in seen or span in metadata_spans:
                continue
            seen.add(span)
            lines.append(f'- Source excerpt: “{span}” {cite(candidate)}')
            if len(seen) == 2:
                break
        if not seen:
            lines.append(f'- Source excerpt: “{artifact.original_text_chunk[:600]}” {cite(artifact)}')
        lines.append("")
    for edge in selection["relationships"]:
        evidence = edge["evidence"]
        artifact = next((a for a in all_evidence if a.document_id == edge["document_id"] and
                         a.page_number == evidence["page_number"] and evidence["exact_span"] in a.original_text_chunk), None)
        if artifact:
            kind = {"revision_of": "is an explicit revision of (not proof of complete replacement)",
                    "amendment_of": "amends portions of", "supplement_to": "supplements",
                    "supersedes": "explicitly supersedes", "replaces": "explicitly replaces",
                    "withdraws": "explicitly withdraws"}.get(edge["type"], edge["type"])
            lines += [f"- {edge['document_id']} {kind} {edge['target_id']}: “{evidence['exact_span']}” {cite(artifact)}"]
    if selection["uncertainties"]:
        lines += ["", *[f"- {message}" for message in selection["uncertainties"]]]
    preamble = f"{selection['intent'].capitalize()} source review within the indexed corpus, as of {selection['as_of']}. Dates alone do not prove authority or replacement."
    return "\n".join(lines).strip(), preamble, sources
