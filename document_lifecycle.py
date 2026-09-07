"""Conservative document lifecycle extraction with a revisioned SQLite index.

Dates and relationships identify document versions; this module never labels a
document current merely because it is newer. Explicit signals retain their exact
canonical source span. Title/catalog candidates remain distinguishable from them.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
import hashlib
import json
import re


EXTRACTOR_VERSION = "v3-lifecycle"
YEAR = r"(?:19|20)\d{2}"
MONTHS = {name.casefold(): index for index, name in enumerate((
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"), 1)}
MONTH_PATTERN = "(?:" + "|".join(MONTHS) + ")"
DATE_PATTERN = rf"(?:{YEAR}-\d{{2}}-\d{{2}}|{MONTH_PATTERN}\s+\d{{1,2}},?\s+{YEAR}|\d{{1,2}}\s+{MONTH_PATTERN}\s+{YEAR}|{YEAR}-\d{{2}}|{MONTH_PATTERN}\s+{YEAR}|{YEAR})"
DATE_LABEL = re.compile(
    rf"(?P<label>originally\s+published|publication\s+date|date\s+of\s+publication|published|issued|last\s+revised|revision\s+date|revised|updated|effective\s+date|effective)"
    rf"\s*(?::|as of|on|in)?\s*(?P<date>{DATE_PATTERN})(?![\d-])", re.I)
RELATION = re.compile(
    r"(?P<supersedes>supersedes)\s+|(?P<replaces>replaces)\s+|(?P<withdraws>withdraws|rescinds)\s+"
    r"|(?P<amendment>amends)\s+|(?P<supplement>supplements)\s+"
    r"|(?P<revision>(?:(?:is|serves as|constitutes|represents)\s+(?:(?:a|the)\s+)?)?"
    r"(?:(?:comprehensive|comp\s+rehensive)\s+)?(?:revised\s+version|revision|update)\s+(?:of|to))\s+"
    r"|(?P<amendment_of>(?:(?:is|constitutes)\s+(?:(?:an?|the)\s+)?)?amendment\s+to)\s+"
    r"|(?P<supplement_to>(?:(?:is|constitutes)\s+(?:(?:an?|the)\s+)?)?(?:supplement|addendum)\s+to)\s+", re.I)
NEGATED_OR_PROSPECTIVE = re.compile(
    r"\b(?:not|never|neither|without|might|could|would|should|will|if|planned|proposed|scheduled)\b"
    rf"|\bmay\b(?!\s+(?:\d{{1,2}},?\s+)?{YEAR}\b)"
    r"|\b(?:doesn|didn|isn|wasn)['\u2019]t\b", re.I)


def parse_date(value: str) -> dict | None:
    """Parse supported dates without inventing a day/month for year-only text."""
    value = value.strip()
    year = month = day = None
    match = re.fullmatch(rf"({YEAR})(?:-(\d{{2}})(?:-(\d{{2}}))?)?", value)
    if match:
        year, month, day = (int(part) if part else None for part in match.groups())
    else:
        match = re.fullmatch(rf"({MONTH_PATTERN})\s+(?:(\d{{1,2}}),?\s+)?({YEAR})", value, re.I)
        if match:
            month, day, year = MONTHS[match[1].casefold()], int(match[2]) if match[2] else None, int(match[3])
        else:
            match = re.fullmatch(rf"(\d{{1,2}})\s+({MONTH_PATTERN})\s+({YEAR})", value, re.I)
            if match:
                day, month, year = int(match[1]), MONTHS[match[2].casefold()], int(match[3])
    if year is None:
        return None
    try:
        date(year, month or 1, day or 1)
        if month == 0 or day == 0:
            return None
    except ValueError:
        return None
    result = str(year)
    if month is not None:
        result += f"-{month:02d}"
    if day is not None:
        result += f"-{day:02d}"
    return {"value": result, "precision": "day" if day is not None else "month" if month is not None else "year"}


def normalize_title(title: str) -> str:
    """Remove date/version markers while retaining subject and document type."""
    value = title.casefold()
    value = re.sub(r"\b(?:version|ver\.?|v)\s*\d+(?:\.\d+)*\b", " ", value)
    value = re.sub(r"\b(?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\s+edition\b", " ", value)
    value = re.sub(rf"\b{YEAR}\b", " ", value)
    value = re.sub(r"\b(?:revised|revision|updated|update)\b", " ", value)
    return " ".join(re.findall(r"[\w]+", value))


def _family_title(title):
    value = normalize_title(title)
    value = re.sub(r"^(?:draft|final)\s+|\s+(?:draft|final)$", "", value)
    return value.strip()


def _aliases(title):
    words = re.findall(r"[A-Za-z]+", re.sub(rf"\b{YEAR}\b", "", title))
    words = [word for word in words if word.casefold() not in {"the", "a", "an", "of", "and", "for", "revised", "updated", "draft", "final"}]
    result = {normalize_title(title), _family_title(title)}
    # Acronyms are usable only as own-document identifiers or with a matching
    # target year and a unique same-issuer target, never as semantic similarity.
    result.update("".join(word[0] for word in words[-length:]).casefold()
                  for length in range(3, min(8, len(words)) + 1))
    return {value for value in result if value}


def _sentences(chunks):
    for chunk in chunks:
        text = chunk["original_text_chunk"]
        for match in re.finditer(r"[^\n]+", text):
            line = match[0]
            for sentence in re.split(r"(?<=[.!?])\s+", line):
                span = sentence.strip()
                if span:
                    yield span, {"document_id": chunk["document_id"], "page_number": str(chunk["page_number"]), "exact_span": span}


def _front_page(evidence):
    match = re.match(r"^(\d+)", evidence["page_number"])
    return bool(match and int(match[1]) <= 3)


def _own_subject(prefix, document):
    value = prefix.strip(" ,:;-")
    if not value or value.startswith(('"', "'", "\u201c", "\u2018")):
        return False
    if re.fullmatch(r"(?:this|our|the present|the current)\s+(?:\d{4}\s+)?(?:document|guidance|report|strategy|plan|version|edition|amendment|supplement|update)", value, re.I):
        return True
    value = re.sub(r"^(?:the|this)\s+", "", value, flags=re.I)
    value = re.sub(r"^document\s+", "", value, flags=re.I)
    if value == document["document_id"]:
        return True
    normalized = normalize_title(value)
    return normalized in _aliases(document["title"])


def _relation_clause(span, match, document, evidence):
    prefix = span[:match.start()].strip()
    if _own_subject(prefix, document):
        return prefix, evidence
    # Some PDF chunks flatten headings/lists into the start of a sentence. An
    # explicit dated own-document acronym can still anchor the following clause.
    anchor = re.search(rf"\b(?:The|This)\s+{YEAR}\s+[A-Za-z]{{3,8}}\s*$", span[:match.start()])
    if anchor and _own_subject(anchor[0], document):
        preceding = span[:anchor.start()]
        if re.search(r"\b(?:says|said|states|claimed|quotes|reported|according to)\b[^.!?]{0,120}$", preceding, re.I):
            return None
        return anchor[0], {**evidence, "exact_span": span[anchor.start():]}
    return None


def _signal(value, source, evidence=None, **extra):
    return {"value": value, "source": source, "confidence": "high" if source == "explicit" else "low",
            "evidence": evidence, **extra}


def _issuer_matches(publisher, agency):
    if not agency.strip():
        return False
    clean = normalize_title(publisher)
    name = normalize_title(agency)
    words = re.findall(r"[A-Za-z]+", publisher)
    words = [word for word in words if word.casefold() not in {"of", "the", "and"}]
    acronyms = {"".join(word[0] for word in words[-length:]).casefold()
                for length in range(3, min(8, len(words)) + 1)}
    return name in clean or name.replace(" ", "") in acronyms


def _version_value(label):
    numeric = re.search(r"\d+(?:\.\d+)*", label)
    if numeric:
        return numeric[0]
    ordinals = {word: str(index) for index, word in enumerate(("first", "second", "third", "fourth", "fifth", "sixth"), 1)}
    return next((number for word, number in ordinals.items() if word in label.casefold()), label)


def _frontmatter_signals(metadata, chunks):
    """Read recognizable imprint/byline/citation labels, not arbitrary years."""
    signals, versions = [], []
    for item in chunks:
        page = str(item["page_number"])
        if page not in {"1", "2", "3", "Web"}:
            continue
        text = item["original_text_chunk"][:6000]

        def add(match, value, basis, confidence="high"):
            parsed = parse_date(value)
            if parsed:
                signals.append(_signal(parsed["value"], "explicit", {
                    "document_id": item["document_id"], "page_number": page, "exact_span": match[0]},
                    precision=parsed["precision"], basis=basis))
                signals[-1]["confidence"] = confidence

        if page == "Web":
            for match in re.finditer(rf"\bBy\s+([^|\n]{{1,100}})\s*\|\s*({DATE_PATTERN})(?![\d-])", text, re.I):
                add(match, match[2], "web_byline")
            continue
        for match in re.finditer(rf"\bPrepared\s+by\s+[^\n]{{1,180}}?\s+({DATE_PATTERN})(?![\d-])", text, re.I):
            if normalize_title(metadata["title"]) in normalize_title(text[:1000]):
                add(match, match[1], "prepared_by_cover_date")
        for match in re.finditer(rf"([A-Z][A-Za-z. ]{{3,95}}),\s*[A-Za-z .'-]+,\s*[A-Za-z .'-]+:\s*({DATE_PATTERN})(?![\d-])", text):
            if _issuer_matches(match[1], metadata.get("agency", "")):
                add(match, match[2], "publisher_imprint")
        for match in re.finditer(rf"Suggested\s+Citation\s*:?\s*.{{1,350}}?[.,]\s*({YEAR})[.,]\s*(.{{8,240}}?)(?:\.\s|$)", text, re.I):
            title_words = set(normalize_title(metadata["title"]).split()) - {"pdf", "html"}
            if len(title_words) >= 3 and title_words <= set(normalize_title(match[2]).split()):
                add(match, match[1], "suggested_self_citation")
        for match in re.finditer(rf"\bCopyright\s*(?:\u00a9|\(c\))?\s*([^\n]{{3,100}}?)\s+({YEAR})\b", text, re.I):
            if _issuer_matches(match[1], metadata.get("agency", "")):
                add(match, match[2], "copyright_year_proxy", "medium")
        for match in re.finditer(r"\b(?:The\s+)?([^\n.]{3,110}?),\s*(\d+(?:st|nd|rd|th))\s+edition\b", text, re.I):
            if _issuer_matches(match[1], metadata.get("agency", "")):
                versions.append(_signal(_version_value(match[2]), "explicit", {
                    "document_id": item["document_id"], "page_number": page, "exact_span": match[0]}))
    return signals, versions


def _kind(title):
    for name, pattern in (("meeting", r"\b(?:meeting|minutes|workshop)\b"),
                          ("research", r"\b(?:research|study|trial|experiment)\b"),
                          ("guidance", r"\b(?:guidance|guideline|guidelines|manual|protocol)\b"),
                          ("strategy", r"\b(?:strategy|plan)\b"),
                          ("report", r"\b(?:report|review|assessment)\b")):
        if re.search(pattern, title, re.I):
            return name
    return "unknown"


def _own_advance_header(span):
    marker = re.search(r"\bunedited\s+advance\s+version\b", span, re.I)
    if not marker:
        return False
    prefix = span[:marker.start()].strip()
    if not prefix:
        return True
    if re.fullmatch(r"(?:This|The present)\s+(?:document|report|guidance|assessment)\s+is\s+(?:an?\s+)?", prefix + " ", re.I):
        return True
    # A short cover-heading suffix is different from discussing another work.
    if re.search(r"\b(?:discuss\w*|cit\w*|refer\w*|external|previous|another|quot\w*|according|review\s+of)\b", prefix, re.I):
        return False
    return bool(len(prefix) <= 300 and re.search(r"\b(?:report|guidance|assessment|summary|strategy|plan)\b", prefix, re.I)
                and re.search(r"[-\u2013\u2014:\ufffd](?:C)?\s*$", prefix))


def _extract_document(metadata, chunks):
    docid = metadata["document_id"]
    result = {"document_id": docid, "title": metadata["title"], "agency": metadata.get("agency", ""),
              "topic": metadata.get("topic", ""), "publication_date": None, "revision_date": None,
              "effective_date": None, "version": None, "status": "unknown", "kind": _kind(metadata["title"]),
              "family_id": "document:" + docid, "family_source": "unknown", "dates": {},
              "relations": [], "warnings": [], "signals": [], "status_evidence": None, "version_evidence": None}
    candidates = defaultdict(list)
    catalog = parse_date(str(metadata.get("year", "")))
    if catalog:
        candidates["publication_date"].append(_signal(catalog["value"], "metadata_inferred", precision=catalog["precision"]))
    title_years = list(dict.fromkeys(re.findall(rf"\b{YEAR}\b", metadata["title"])))
    if len(title_years) == 1:
        candidates["publication_date"].append(_signal(title_years[0], "title_inferred", precision="year"))
    title_status = re.search(r"^\s*(draft|final|withdrawn)\b", metadata["title"], re.I)
    if title_status is None:
        title_status = re.search(r"\b(draft|final|withdrawn)\s*(?:version\s*)?(?:\d{4})?\s*[)\]]?\s*$", metadata["title"], re.I)
    if title_status:
        label = title_status[1].casefold()
        result["status"], result["status_evidence"] = label, _signal(label, "title_inferred")
    title_version = re.search(r"\b(?:version\s*:?\s*\d+(?:\.\d+)*|(?:first|second|third|fourth|fifth)\s+edition)\b", metadata["title"], re.I)
    if title_version:
        result["version"] = _version_value(title_version[0])
        result["version_evidence"] = _signal(result["version"], "title_inferred")
    front_dates, front_versions = _frontmatter_signals(metadata, chunks)
    if front_dates:
        candidates["publication_date"].extend(front_dates)
    if front_versions:
        result["version"], result["version_evidence"] = front_versions[0]["value"], front_versions[0]
    own_aliases = _aliases(metadata["title"])
    for span, evidence in _sentences(chunks):
        front = _front_page(evidence)
        if NEGATED_OR_PROSPECTIVE.search(span):
            continue
        for match in DATE_LABEL.finditer(span):
            prefix = span[:match.start()].strip()
            prefix = re.sub(r"\s+(?:was|is)$", "", prefix, flags=re.I)
            if not ((front and not prefix) or _own_subject(prefix, metadata)):
                continue
            parsed = parse_date(match["date"])
            if not parsed:
                result["warnings"].append("Invalid labelled date was not indexed: " + match["date"])
                continue
            label = match["label"].casefold()
            field = "revision_date" if re.search(r"revis|updat", label) else "effective_date" if "effective" in label else "publication_date"
            candidates[field].append(_signal(parsed["value"], "explicit", evidence, precision=parsed["precision"]))
        status = re.match(r"^(?:document\s+)?status\s*:\s*(draft|final|withdrawn)\b", span, re.I) if front else None
        if status is None:
            status = re.match(r"^(?:This|The present)\s+(?:document|guidance|report|strategy|plan)\s+is\s+(?:now\s+)?(draft|final|withdrawn)\b", span, re.I)
        if status is None and front:
            status = re.fullmatch(r"(Draft|Final|Withdrawn)(?:\s+(?:guidance|report|version))?[.!]?", span, re.I)
        if status:
            status_value = status[1].casefold()
            if result["status_evidence"] and result["status_evidence"]["source"] == "explicit" and result["status"] != status_value:
                result["warnings"].append("Conflicting explicit document status signals")
                result["status"] = "unknown"
            else:
                result["status"] = status_value
            result["status_evidence"] = _signal(status_value, "explicit", evidence)
        if front and _own_advance_header(span):
            result["status"] = "draft"
            result["status_evidence"] = _signal("draft", "explicit", evidence)
            result["warnings"].append("Unedited advance version treated as draft, not established final guidance")
        version = re.search(r"\b(?:version\s*:?\s*\d+(?:\.\d+)*|(?:first|second|third|fourth|fifth)\s+edition)\b", span, re.I)
        if version and ((front and not span[:version.start()].strip()) or _own_subject(re.sub(r"\s+is$", "", span[:version.start()].strip(), flags=re.I), metadata)):
            result["version"] = _version_value(version[0])
            result["version_evidence"] = _signal(result["version"], "explicit", evidence)
        # A year adjacent to this document's own front-cover title is an explicit
        # cover date, distinct from any reference years inside a later report.
        if evidence["page_number"] == "1":
            years = list(dict.fromkeys(re.findall(rf"\b{YEAR}\b", span)))
            if len(years) == 1 and normalize_title(metadata["title"]) in normalize_title(span):
                candidates["publication_date"].append(_signal(years[0], "explicit", evidence, precision="year"))
        # Own-version statements remain valid beyond front matter. A target date
        # or scheduled future revision is never treated as the source's date.
        dated_own = re.search(rf"\bthis\s+version\s+of\s+(?:the\s+)?([A-Z]{{3,8}})\s*\(({YEAR})\)", span, re.I)
        if dated_own and dated_own[1].casefold() in own_aliases:
            own_version = _signal(dated_own[2], "explicit", evidence, basis="own_version_year")
            result["signals"].append({"field": "version_year", **own_version})
            if result["version"] is None:
                result["version"], result["version_evidence"] = dated_own[2], own_version
        for match in RELATION.finditer(span):
            own = _relation_clause(span, match, metadata, evidence)
            if own is None:
                continue
            prefix, relation_evidence = own
            dated_prefix = re.search(rf"\b({YEAR})\b", prefix)
            if dated_prefix and match.lastgroup == "revision":
                candidates["revision_date"].append(_signal(dated_prefix[1], "explicit", relation_evidence, precision="year"))
    for field, options in candidates.items():
        options.sort(key=lambda item: (item["source"] != "explicit", item["source"] == "title_inferred", -len(item["value"])))
        chosen = options[0]
        result[field], result["dates"][field] = chosen["value"], chosen
        if chosen.get("basis") == "copyright_year_proxy":
            result["warnings"].append("Publication year uses an explicit copyright-year proxy; a separate publication date was not established")
        result["signals"].extend({"field": field, **item} for item in options)
        for other in options[1:]:
            shorter = min(len(chosen["value"]), len(other["value"]))
            if chosen["value"][:shorter] != other["value"][:shorter]:
                result["warnings"].append(f"Conflicting {field}: {chosen['value']} ({chosen['source']}) versus {other['value']} ({other['source']})")
    version_years = {item["value"][:4] for item in result["signals"] if item["field"] in {"version_year", "revision_date"} and item["source"] == "explicit"}
    if catalog and any(year != catalog["value"][:4] for year in version_years):
        result["warnings"].append("The document's revision/version year differs from its catalog publication year; these dates describe different lifecycle events")
    return result


def _target_ids(reference, owner, documents):
    ids = re.findall(r"\bDOC\d{3,}\b", reference)
    if ids:
        return [docid for docid in dict.fromkeys(ids) if docid in documents and docid != owner["document_id"]]
    candidates = []
    normalized = normalize_title(reference)
    for docid, target in documents.items():
        if docid == owner["document_id"]:
            continue
        title = normalize_title(target["title"])
        full_title = bool(title and re.search(r"(?<!\w)" + re.escape(title) + r"(?!\w)", normalized))
        years = {value[:4] for value in (target["publication_date"], target["revision_date"]) if value}
        year_alias = False
        if target["agency"] and target["agency"].casefold() == owner["agency"].casefold():
            for year in years:
                for alias in _aliases(target["title"]):
                    if " " not in alias and re.search(r"\b" + year + r"\s+" + re.escape(alias) + r"\b", reference, re.I):
                        year_alias = True
        if full_title:
            mentioned_years = set(re.findall(rf"\b{YEAR}\b", reference))
            if mentioned_years and not mentioned_years.intersection(years):
                full_title = False
        if full_title or year_alias:
            candidates.append(docid)
    # Multiple explicit references joined by 'and' can resolve independently;
    # ambiguous same-title versions without a disambiguating year stay unresolved.
    if len(candidates) > 1:
        titles = [normalize_title(documents[docid]["title"]) for docid in candidates]
        if len(titles) != len(set(titles)) or not re.search(r"\b(?:and|both)\b", reference, re.I):
            return []
    grouped = defaultdict(list)
    for docid in candidates:
        target = documents[docid]
        grouped[(normalize_title(target["title"]), target["publication_date"])].append(docid)
    return [members[0] for members in grouped.values() if len(members) == 1]


def extract_lifecycles(metadata: list[dict], chunks: list[dict]) -> dict[str, dict]:
    """Pure extraction from canonical document metadata and source chunks."""
    by_doc = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk["document_id"]].append(chunk)
    records = {item["document_id"]: _extract_document(item, by_doc[item["document_id"]]) for item in metadata}
    for docid, owner in records.items():
        seen = set()
        for span, evidence in _sentences(by_doc[docid]):
            if NEGATED_OR_PROSPECTIVE.search(span) or span.startswith(('"', "'", "\u201c", "\u2018")):
                continue
            for match in RELATION.finditer(span):
                own = _relation_clause(span, match, owner, evidence)
                if own is None:
                    continue
                _, relation_evidence = own
                kind = {"revision": "revision_of", "amendment": "amendment_of", "supplement": "supplement_to"}.get(match.lastgroup, match.lastgroup)
                reference = span[match.end():]
                targets = _target_ids(reference, owner, records)
                if not targets:
                    owner["warnings"].append("An explicit lifecycle reference could not be uniquely resolved to an indexed document")
                for target in targets:
                    key = (kind, target, relation_evidence["page_number"], relation_evidence["exact_span"])
                    if key not in seen:
                        owner["relations"].append({"type": kind, "target_id": target, "source": "explicit", "evidence": relation_evidence})
                        seen.add(key)
                    # Withdrawal is a dated directional relation. Do not mutate
                    # the target's intrinsic status independent of query time.
    parent = {docid: docid for docid in records}

    def find(docid):
        while parent[docid] != docid:
            docid = parent[docid]
        return docid

    def union(first, second, source):
        left, right = find(first), find(second)
        if left != right:
            parent[max(left, right)] = min(left, right)

    families = defaultdict(list)
    for docid, record in records.items():
        if record["agency"].strip() and _family_title(record["title"]):
            families[(_family_title(record["title"]), record["agency"].strip().casefold())].append(docid)
        for relation in record["relations"]:
            union(docid, relation["target_id"], "explicit")
    for members in families.values():
        for other in members[1:]:
            union(members[0], other, "title_inferred")
    components = defaultdict(list)
    for docid in records:
        components[find(docid)].append(docid)
    for members in components.values():
        if len(members) < 2:
            continue
        digest = hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:16]
        explicit_members = {docid for record in records.values() for relation in record["relations"]
                            for docid in (record["document_id"], relation["target_id"])}
        for docid in members:
            records[docid]["family_id"] = "family:" + digest
            records[docid]["family_source"] = "explicit" if docid in explicit_members else "title_inferred"
    for record in records.values():
        record["warnings"] = list(dict.fromkeys(record["warnings"]))
    return records


def _ensure_schema(store):
    if getattr(store, "_v3_lifecycle_schema_ready", False):
        return
    connection = store.connection
    connection.execute("SAVEPOINT lifecycle_schema")
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS document_lifecycle_state (singleton INTEGER PRIMARY KEY CHECK(singleton=1), revision INTEGER NOT NULL, built_revision INTEGER NOT NULL, extractor_version TEXT NOT NULL)")
        connection.execute("INSERT OR IGNORE INTO document_lifecycle_state VALUES (1,0,-1,'')")
        connection.execute("CREATE TABLE IF NOT EXISTS document_lifecycles (document_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        for table in ("documents", "knowledge_artifacts"):
            for event in ("INSERT", "UPDATE", "DELETE"):
                connection.execute(f"CREATE TRIGGER IF NOT EXISTS lifecycle_{table}_{event.lower()} AFTER {event} ON {table} BEGIN UPDATE document_lifecycle_state SET revision=revision+1 WHERE singleton=1; END")
        connection.execute("RELEASE SAVEPOINT lifecycle_schema")
    except Exception:
        connection.execute("ROLLBACK TO SAVEPOINT lifecycle_schema")
        connection.execute("RELEASE SAVEPOINT lifecycle_schema")
        raise
    if not connection.in_transaction:
        store._v3_lifecycle_schema_ready = True


def rebuild_lifecycles(store) -> dict[str, dict]:
    """Atomically rebuild the derived index; canonical data/vectors stay intact."""
    with store._lock:
        _ensure_schema(store)
        connection = store.connection
        connection.execute("SAVEPOINT lifecycle_rebuild")
        try:
            rows = connection.execute("SELECT document_id,title,year,agency,topic FROM documents ORDER BY document_id").fetchall()
            metadata = [dict(row) for row in rows]
            chunks = [dict(row) for row in connection.execute("SELECT document_id,title,page_number,original_text_chunk FROM knowledge_artifacts ORDER BY artifact_id")]
            known = {item["document_id"] for item in metadata}
            for chunk in chunks:
                if chunk["document_id"] not in known:
                    metadata.append({"document_id": chunk["document_id"], "title": chunk["title"], "year": "", "agency": "", "topic": ""})
                    known.add(chunk["document_id"])
            result = extract_lifecycles(metadata, chunks)
            connection.execute("DELETE FROM document_lifecycles")
            connection.executemany("INSERT INTO document_lifecycles VALUES (?,?)", [(docid, json.dumps(payload, ensure_ascii=False, sort_keys=True)) for docid, payload in result.items()])
            connection.execute("UPDATE document_lifecycle_state SET built_revision=revision,extractor_version=? WHERE singleton=1", (EXTRACTOR_VERSION,))
            connection.execute("RELEASE SAVEPOINT lifecycle_rebuild")
            return result
        except Exception:
            connection.execute("ROLLBACK TO SAVEPOINT lifecycle_rebuild")
            connection.execute("RELEASE SAVEPOINT lifecycle_rebuild")
            raise


def get_lifecycles(store, force: bool = False) -> dict[str, dict]:
    """Read the index cheaply; canonical mutations trigger one lazy rebuild."""
    with store._lock:
        _ensure_schema(store)
        state = store.connection.execute("SELECT revision,built_revision,extractor_version FROM document_lifecycle_state WHERE singleton=1").fetchone()
        if force or state[0] != state[1] or state[2] != EXTRACTOR_VERSION:
            return rebuild_lifecycles(store)
        return {row[0]: json.loads(row[1]) for row in store.connection.execute("SELECT document_id,payload FROM document_lifecycles ORDER BY document_id")}
