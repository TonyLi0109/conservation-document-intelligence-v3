# Minimal RCA corrections

Implemented in the existing backend; no LangChain/LlamaIndex dependency was introduced. Helpers operate on dictionaries and existing immutable canonical artifacts. Framework adapters can supply document metadata and stable chunk IDs without changing the canonical provenance handles.

## 1. Explicit planning periods

`document_lifecycle.py` extracts only own-plan wording or frontmatter planning labels, retaining the exact source span. It ignores unrelated ranges, reversed intervals, and conflicts. The extractor version is bumped, so the derived JSON lifecycle index refreshes automatically on its next use; no corpus re-ingestion is needed.

Exact extraction implementation:

```python
PLANNING_RANGE = re.compile(r"\b(?P<start>20\d{2})\s*(?:[-\u2013\u2014]|to|through)\s*(?P<end>20\d{2})\b", re.I)
PLANNING_OWN = re.compile(
    r"\b(?:this|the present)\s+(?:\w+\s+){0,4}plan\s+"
    r"(?:includes?|covers?|spans?|is for|applies to)\s+(?:the\s+)?(?:years?\s+)?$", re.I)


def extract_planning_period(chunks):
    """Keep verbatim evidence; reject reversed, external, and conflicting ranges."""
    periods = []
    for span, evidence in _sentences(chunks):
        for match in PLANNING_RANGE.finditer(span):
            prefix = span[:match.start()]
            own = PLANNING_OWN.search(prefix)
            label = re.fullmatch(r"\s*(?:planning period|planning horizon|plan period|plan timeframe)\s*:?\s*", prefix, re.I)
            if not (own or label and _front_page(evidence)):
                continue
            if int(match['start']) > int(match['end']):
                continue
            start = own.start() if own else 0
            periods.append({'start_year': int(match['start']), 'end_year': int(match['end']),
                            'source': 'explicit', 'confidence': 'high',
                            'evidence': {**evidence, 'exact_span': span[start:match.end()]}})
    unique = {(p['start_year'], p['end_year']) for p in periods}
    return (periods[0] if len(unique) == 1 else None), len(unique) > 1


```

Inside `_extract_document`:

```python
    result['planning_period'], planning_conflict = extract_planning_period(chunks)
    if planning_conflict:
        result['warnings'].append('Conflicting explicit planning periods; no operational horizon selected')
```

Exact active-horizon predicate in `temporal.py`:

```python
def active_planning_horizon(document, when):
    """Inclusive calendar-year horizon; never imply publication or supersession."""
    period = document.get('planning_period')
    return bool(period and period.get('source') == 'explicit'
                and period['start_year'] <= when.year <= period['end_year']
                and _active(document, when))


```

Current-mode ordering adds the horizon preference after explicit replacement handling. No boost applies to expired/future periods, draft/withdrawn documents, or known revised sources. The end year is inclusive through December 31; it is never a publication date. Latest/historical date sorting retains its previous meaning. Within-family ordering includes the same horizon preference so a later pass cannot undo it. Planning evidence is displayed with its source span in temporal answers.

## 2. Corpus-aware routing

`document_targets.py` resolves exact, unambiguous corpus titles and document IDs. It masks only recognized mentions and their matching adjacent catalog years. Conflicting edition years and ambiguous names remain unresolved. Independent temporal wording (for example `as of 2020` or `current`) survives masking.

Exact resolver and comparison helper:

```python
"""Conservative corpus-title resolution shared by routing and retrieval."""
import re
import unicodedata


def normalize(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def resolve_document_targets(query, documents):
    """Return unique exact title/ID matches and text with those mentions masked.

    No fuzzy entity inference: ambiguous title/year matches remain unresolved.
    Mask only the recognized entity and its adjacent catalog year, preserving
    independent 'as of 2020', 'latest', or other lifecycle requests.
    """
    text = normalize(query)
    candidates = {}
    for document in documents:
        title = normalize(document['title'])
        year = str(document.get('year') or '')
        aliases = {title}
        if re.fullmatch(r'(?:19|20)\d{2}', year):
            aliases.add(re.sub(r'^' + year + r'\s+', '', title))
        for alias in aliases:
            if len(alias.split()) < 3:
                continue
            for match in re.finditer(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', text):
                start, end = match.span()
                prefix = re.search(r'\b((?:19|20)\d{2})\s+$', text[:start])
                # A conflicting edition year must not resolve to this catalog row.
                if prefix:
                    if prefix[1] != year:
                        continue
                    start = prefix.start()
                candidates.setdefault((start, end), set()).add(document['document_id'])
        for match in re.finditer(r'(?<!\w)' + re.escape(document['document_id'].casefold()) + r'(?!\w)', text):
            candidates.setdefault(match.span(), set()).add(document['document_id'])
    selected, spans = [], []
    for (start, end), ids in sorted(candidates.items(), key=lambda pair: -(pair[0][1]-pair[0][0])):
        if len(ids) != 1 or any(start < b and end > a for a, b in spans):
            continue
        spans.append((start, end))
        selected.extend(ids)
    chars = list(text)
    for start, end in spans:
        chars[start:end] = ' ' * (end-start)
    return list(dict.fromkeys(selected)), ''.join(chars)


def comparison_targets(query, documents):
    ids, residual = resolve_document_targets(query, documents)
    comparing = re.search(r'\b(?:compar\w*|differ\w*|versus|vs|between|contrast\w*)\b', residual)
    return ids if len(ids) > 1 and comparing else []
```

`detect_temporal_intent(question, *, documents=())` now begins with:

```python
    from document_targets import resolve_document_targets
    targets, residual = resolve_document_targets(question, documents)
    # Entity years describe editions, not a request to rank versions by time.
    text = residual if targets else question.casefold()
```

The existing temporal rules run on `text`. `main.ask_chatbot_with_context` passes corpus metadata to this function and records `multi_doc_synthesis` when multiple recognized titles request a comparison without independent temporal intent. It passes resolved target IDs to retrieval. The search entry point uses the same classifier. A route of `none` means use normal synthesis, not an empty answer.

## 3. Diversity with coverage reservation

Exact selector in `retrieval.py`:

```python
def select_diverse_evidence(ranked_ids, artifacts, top_k, target_ids, *, duplicate_threshold=0.92, decay=1.0):
    """Reserve one eligible chunk per explicit target, then decay repeat votes.

    The score is an ordinal utility, not a provenance confidence. Preserve
    canonical objects. Cross-document duplicates remain independently citable.
    Coverage is bounded by top_k and available candidates; report any gap.
    """
    remaining = list(ranked_ids)
    rank = {item: position for position, item in enumerate(remaining, 1)}
    selected, suppressed, decisions = [], [], []
    counts = Counter()
    signatures = {}
    while remaining and len(selected) < top_k:
        unseen = set(target_ids) - set(counts)
        eligible = [i for i in remaining if artifacts[i].document_id in unseen]
        pool = eligible or remaining
        def utility(item):
            return 1.0 / (60 + rank[item]) / (1.0 + decay * counts[artifacts[item].document_id])
        item = max(pool, key=lambda i: (utility(i), -rank[i]))
        remaining.remove(item)
        doc_id = artifacts[item].document_id
        signature = _signature(artifacts[item].original_text_chunk)
        if any(_duplicates(signature, previous, duplicate_threshold) for previous in signatures.get(doc_id, [])):
            suppressed.append(item)
            continue
        decisions.append({'artifact_id': item, 'document_id': doc_id,
                          'marginal_score': utility(item), 'previous_count': counts[doc_id],
                          'coverage_reservation': bool(eligible)})
        selected.append(item)
        counts[doc_id] += 1
        signatures.setdefault(doc_id, []).append(signature)
    return selected, suppressed, {'selection': decisions,
        'missing_target_ids': sorted(set(target_ids) - set(counts)),
        'capacity_limited': top_k < len(set(target_ids)), 'decay': decay}


```

Integration: recognized comparison targets establish a hard scope. Scoped document candidates are preserved across fusion/rerank truncation. The selector runs after relevance/entity filtering, only for explicit multi-document comparisons. One eligible chunk per available target is reserved before repeating documents, then the marginal score decays as `base_rank_score / (1 + decay * selected_count)`.

Within-document duplicate suppression uses the existing threshold. Identical text from different target documents remains independently citable. Missing targets and insufficient top-k capacity are reported in retrieval diagnostics; the selector does not fabricate evidence or exceed top-k. Other queries keep the existing selection path. Provenance validation and thresholds are unchanged.

## Validation

The focused regression suite passed 173 tests, including synthetic end-to-end synthesis through the existing provenance validator and a deliberately imbalanced 80-to-1 candidate corpus.

Offline corpus results (no provider calls, temporary database copies):

- Q1: DOC002 now has planning period 2023-2028 with exact source evidence on page 1 and remains first.
- Q4: title-aware temporal classification returns `none`, allowing normal synthesis.
- Q4 top five after reranking: DOC036, DOC001, DOC036, DOC001, DOC036. No target coverage gaps.

Run `python scripts/debug_q1.py` and `python scripts/debug_q4.py` from `new-V3`. These diagnostics do not make live generator calls; dense ablations require the matching cached embedding.
