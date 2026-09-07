"""Provider-independent hybrid retrieval of immutable canonical evidence.

SQLite BM25 and the existing cosine index vote by rank (RRF), not raw score.
Document metadata recovers candidate pages before bounded deterministic
reranking. No lifecycle authority is inferred here; temporal.py owns that policy.
"""
from __future__ import annotations

from collections import Counter
import re
import time
import unicodedata

from config import SETTINGS


_STOP = set("a an and are as at be been being by can could did do does for from had has have how i if in into is it its me of on or our please should so than that the their them these they this those to was we were what when where which who why will with would you your".split())
_STOP.update("find report reports document documents mentioned discuss discusses discussed tell show shows provide evidence data information some about across available question topic say says said method methods effective effectiveness".split())
_WORD = re.compile(r"[\w]+", re.UNICODE)


def _normalize(text):
    return " ".join(_WORD.findall(unicodedata.normalize("NFKC", text).casefold()))


def _tokens(text):
    return [t for t in _WORD.findall(text.casefold()) if t not in _STOP and len(t) > 1]


def _stem(token):
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "s"):
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token


def _contains(text, phrase):
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.I))


def classify_query(query):
    if re.search(r'"[^"\n]+"|\bDOC\d+\b', query, re.I):
        return "exact_lookup"
    if re.search(r"\b(?:find|locate|looking for|remember|which|what|list)\b.{0,80}\b(?:report|document|publication|plan|title)s?\b", query, re.I):
        return "report_lookup"
    if re.search(r"\b(?:data|quantitative|effective\w*|percent\w*|rate|count|how (?:many|much)|measur\w*|before and after|confidence interval|results)\b", query, re.I):
        return "quantitative_question"
    from chat_context import TOPIC_ALIASES
    if any(_normalize(query) == _normalize(alias) for aliases in TOPIC_ALIASES.values() for alias in aliases):
        return "entity_lookup"
    return "semantic_question"


def enrich_query(query, entity=None):
    """Expand only existing explicit aliases, bounded by the current query scope."""
    from chat_context import TOPIC_ALIASES, BROADER_SCOPE
    expanded, groups = [], []
    if not BROADER_SCOPE.search(query):
        for canonical, aliases in TOPIC_ALIASES.items():
            if any(_contains(query, a) for a in aliases) or entity and (
                entity.casefold() == canonical.casefold() or any(entity.casefold() == a.casefold() for a in aliases)
            ):
                groups.append(tuple(dict.fromkeys((canonical, *aliases))))
                expanded.extend(a for a in aliases if not _contains(query, a))
    terms = list(dict.fromkeys(_tokens(query)))
    # Quotes are retained as literal phrases by the index's safe MATCH builder.
    phrases = re.findall(r'"([^"\n]{2,200})"', query)
    enriched = " ".join([*terms, *expanded, *('"' + p + '"' for p in phrases)])
    return enriched, groups, expanded


def reciprocal_rank_fusion(rankings, k=60):
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("RRF k must be a positive integer")
    votes, first = {}, {}
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), 1):
            votes[item] = votes.get(item, 0.0) + 1.0 / (k + rank)
            first.setdefault(item, len(first))
    return sorted(votes, key=lambda item: (-votes[item], first[item]))


def quantitative_strength(text, topic_terms=(), entity_groups=()):
    """Find a measured outcome in source text, excluding isolated years/page IDs.

    This is an evidence-selection cue, not proof of a claim or effect size.
    Numbers and outcomes must occur together locally, with query-topic overlap.
    """
    number = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*%|\s*percent|\s*(?:tons?|fish|birds?|species|acres?|hectares?|individuals?|sites?|million|billion|kg|km|miles?|fold)\b)?", re.I)
    outcomes = re.compile(r"decreas|increas|reduc|remov|captur|mortality|surviv|densit|abundan|vulnerab|loss|lost|declin|survey|estimated|confidence|±|\bCI\b|effic|rate|index|count|measur|recorded|reported|resulted", re.I)
    generic = {"invasive", "control", "management", "monitoring", "study", "studies", "research", "method", "methods"}
    topic = {_stem(t) for t in topic_terms if not t.isdigit() and t not in generic}
    strength = 0
    for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", text):
        # Long extracted table paragraphs are examined in bounded local windows.
        for match in number.finditer(sentence):
            value = match[0]
            if value.isdigit() and (1900 <= int(value) <= 2099):
                continue
            window = sentence[max(0, match.start() - 180):match.end() + 180]
            if re.search(r"https?://|\b(?:doi|accessed|vol\.|v\.|no\.|pp\.)\b", window, re.I):
                continue
            if not outcomes.search(window):
                continue
            if entity_groups and not any(any(_contains(_normalize(window), _normalize(a)) for a in aliases) for aliases in entity_groups):
                continue
            if topic and not topic.intersection(_stem(t) for t in _tokens(window)):
                continue
            if re.search(r"\b(?:page|p\.|figure|fig\.|table|version|edition|section)\s*$", sentence[max(0, match.start() - 20):match.start()], re.I):
                continue
            units = bool(re.search(r"%|percent|tons?|fish|birds?|species|acres?|hectares?|individuals?|sites?|million|billion|kg|km|miles?|fold", value, re.I))
            strength = max(strength, 2 if units else 1)
    return strength


def quantitative_signal(text, topic_terms=(), entity_groups=()):
    return quantitative_strength(text, topic_terms, entity_groups) > 0


def _signature(text):
    words = _normalize(text).split()
    shingles = {tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))}
    # Preserve changed quantitative findings and contrary outcomes during dedup.
    numbers = tuple(re.findall(r"\d+(?:[.,]\d+)*\s*%?", text))
    negation = tuple(sorted(Counter(re.findall(r"\b(?:no|not|never|without|failed|ineffective|limited|significant|insignificant|increas\w*|decreas\w*|higher|lower|improv\w*|worsen\w*)\b", text.casefold())).items()))
    measurement_context = tuple(
        _normalize(text[max(0, m.start() - 40):m.end() + 40])
        for m in re.finditer(r"\d+(?:[.,]\d+)*\s*(?:%|(?:percent|tons?|kg|acres?|hectares?|fish|birds?|species)\b)", text, re.I)
    )
    return " ".join(words), shingles, numbers, negation, measurement_context


def _duplicates(left, right, threshold):
    if left[0] == right[0]:
        return True
    if left[2:] != right[2:] or not left[1] or not right[1]:
        return False
    return len(left[1] & right[1]) / min(len(left[1]), len(right[1])) >= threshold


def retrieve_evidence(store, query, *, query_embedding=None, top_k=5,
                      mode="hybrid_rerank", diagnostics=None, entity=None,
                      document_ids=None):
    """Return final K canonical chunks; providers are called only by the caller.

    Modes expose ablations: dense and lexical are raw rankers; hybrid adds equal
    RRF votes including document-first candidates; hybrid_rerank additionally
    applies inspectable relevance/outcome signals and conservative deduplication.
    Explicit document_ids is a hard caller constraint; inferred metadata is soft.
    """
    from retrieval_index import ensure_retrieval_index, lexical_candidates, document_candidates
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if mode not in {"dense", "lexical", "hybrid", "hybrid_rerank"}:
        raise ValueError("Unknown retrieval mode")
    if mode == "dense" and query_embedding is None:
        raise ValueError("Dense retrieval requires a query embedding")
    if document_ids is not None and (isinstance(document_ids, str) or any(not isinstance(d, str) for d in document_ids)):
        raise ValueError("document_ids must be a sequence of document IDs")
    settings = SETTINGS.retrieval
    started = time.perf_counter()
    enriched, groups, expansions = enrich_query(query, entity)
    strict_groups = [g for g in groups if g[0] not in {"climate change", "wetland restoration"}
                     or entity and any(entity.casefold() == a.casefold() for a in g)]
    kind = classify_query(query)
    trace = {"query": query, "query_type": kind, "expanded_query": enriched,
             "alias_expansions": expansions, "mode": mode,
             "dense_available": query_embedding is not None}
    allow = set(document_ids) if document_ids is not None else None
    # An explicitly requested stable document ID scopes the lookup precisely.
    named_ids = set(re.findall(r"\bDOC\d+\b", query.upper()))
    if named_ids:
        allow = named_ids if allow is None else allow & named_ids
    if allow is not None and not allow:
        if diagnostics is not None:
            diagnostics.update({**trace, "final_artifact_ids": [], "total_ms": 0.0})
        return []
    dense, lexical, doc_rank, document_votes = [], [], [], []
    if mode != "dense":
        ensure_retrieval_index(store)
    indexed = time.perf_counter()
    if mode != "lexical" and query_embedding is not None:
        dense = store.vector_store.search(query_embedding, max(top_k, settings.dense_top_k))
        if allow is not None:
            # Hard scopes must search within all allowed chunks, not filter an
            # already truncated global Top-K and miss the requested document.
            scoped_ids = {int(r[0]) for r in store.connection.execute(
                f"SELECT artifact_id FROM knowledge_artifacts WHERE document_id IN ({','.join('?' for _ in allow)})", sorted(allow))}
            dense = [i for i in store.vector_store.search(query_embedding, store.artifact_count) if i in scoped_ids][:max(top_k, settings.dense_top_k)]
    dense_end = time.perf_counter()
    if mode != "dense":
        lexical = lexical_candidates(store, enriched or query, max(top_k, settings.lexical_top_k), document_ids=allow)
    lexical_end = time.perf_counter()
    if mode in {"hybrid", "hybrid_rerank"}:
        doc_rank = sorted(allow) if allow is not None else document_candidates(store, enriched or query, settings.document_top_k)
        for doc_id in doc_rank:
            hits = lexical_candidates(store, enriched or query, settings.document_chunk_k, document_ids=[doc_id])
            if not hits and (allow or kind in {"exact_lookup", "report_lookup"}):
                hits = [int(r[0]) for r in store.connection.execute(
                    "SELECT artifact_id FROM knowledge_artifacts WHERE document_id=? ORDER BY artifact_id LIMIT ?",
                    (doc_id, settings.document_chunk_k))]
            if kind in {"exact_lookup", "report_lookup"}:
                first = store.connection.execute(
                    "SELECT artifact_id,title FROM knowledge_artifacts WHERE document_id=? ORDER BY artifact_id LIMIT 1", (doc_id,)).fetchone()
                if first and _normalize(first["title"]) in _normalize(query):
                    hits = list(dict.fromkeys([int(first["artifact_id"]), *hits]))[:settings.document_chunk_k]
            document_votes.append(hits)
        # Round-robin document representatives: long reports don't get more votes
        # simply because they have more chunks or repeat the title on every page.
        document_pool = [row[i] for i in range(settings.document_chunk_k) for row in document_votes if len(row) > i]
        # Metadata recovery extends lexical recall. It is not an independent
        # extra vote for chunks that already matched the same title in BM25.
        recovered_lexical = list(dict.fromkeys([*lexical, *document_pool]))
        fused = reciprocal_rank_fusion([recovered_lexical, dense], settings.rrf_k)[:max(top_k, settings.fusion_top_k)]
    else:
        document_pool = []
        fused = (dense if mode == "dense" else lexical)[:max(top_k, settings.fusion_top_k)]
    retrieved_end = time.perf_counter()
    if mode != "hybrid_rerank":
        final = fused[:top_k]
        result = store._artifacts_by_ranked_ids(final)
        trace.update({"dense_candidates": dense, "lexical_candidates": lexical,
                      "document_candidates": doc_rank, "document_chunk_candidates": document_pool,
                      "fused_ranking": fused, "reranked_ids": fused, "metadata_signals": {},
                      "duplicate_suppressed_ids": [], "final_artifact_ids": final,
                      "final_document_ids": [a.document_id for a in result],
                      "index_ms": round((indexed - started) * 1000, 3),
                      "dense_ms": round((dense_end - indexed) * 1000, 3),
                      "lexical_ms": round((lexical_end - dense_end) * 1000, 3),
                      "retrieval_ms": round((retrieved_end - indexed) * 1000, 3),
                      "rerank_ms": 0.0, "selection_ms": round((time.perf_counter() - retrieved_end) * 1000, 3),
                      "total_ms": round((time.perf_counter() - started) * 1000, 3)})
        if diagnostics is not None:
            diagnostics.update(trace)
        return result
    reranked = fused[:max(top_k, settings.rerank_top_k)] if mode == "hybrid_rerank" else fused
    artifacts = store._artifacts_by_ranked_ids(reranked)
    by_id = dict(zip(reranked, artifacts))
    metadata = {str(row["document_id"]): dict(row) for row in store.connection.execute("SELECT document_id,title,agency,topic,year FROM documents")}
    first_ids = {r[0]: r[1] for r in store.connection.execute("SELECT document_id,MIN(artifact_id) FROM knowledge_artifacts GROUP BY document_id")}
    terms = {_stem(t) for t in _tokens(query)}
    quoted = re.findall(r'"([^"\n]{2,200})"', query)
    features = {}
    for rank, artifact_id in enumerate(reranked):
        artifact = by_id[artifact_id]
        title = _normalize(artifact.title)
        body = _normalize(artifact.original_text_chunk)
        text_terms = {_stem(t) for t in _tokens(artifact.original_text_chunk)}
        title_terms = {_stem(t) for t in _tokens(artifact.title)}
        doc = metadata.get(artifact.document_id, {})
        agency = str(doc.get("agency") or "")
        references = len(re.findall(r"https?://|\bdoi\b|\baccessed\b", artifact.original_text_chunk, re.I)) >= 4
        measured = quantitative_strength(artifact.original_text_chunk, terms, strict_groups) if kind == "quantitative_question" and not references else 0
        features[artifact_id] = {
            "entity_match": sum(any(_contains(body + ' ' + title, _normalize(a)) for a in aliases) for aliases in strict_groups),
            "title_coverage": len(terms & title_terms) / max(1, len(terms)),
            "body_coverage": len(terms & text_terms) / max(1, len(terms)),
            "exact_title": len(title.split()) >= 3 and title in _normalize(query),
            "phrase_match": sum(_normalize(p) in body or _normalize(p) in title for p in quoted),
            "quantitative": measured > 0,
            "measurement_strength": measured,
            "reference_heavy": references,
            "first_page": first_ids.get(artifact.document_id) == artifact_id,
            "agency_match": bool(agency and _contains(_normalize(query), _normalize(agency))),
            "fused_rank": rank + 1,
        }
    if mode == "hybrid_rerank":
        # Clear requested entities are a scope signal; preserve semantic-only
        # matches when no candidate supplies a lexical entity association.
        if strict_groups:
            reranked = [i for i in reranked if features[i]["entity_match"]]
        lookup = kind in {"exact_lookup", "report_lookup"}
        quantitative = kind == "quantitative_question"
        def key(i):
            f = features[i]
            return (-f["exact_title"], -f["first_page"] if f["exact_title"] and lookup else 0,
                    -f["phrase_match"], f["reference_heavy"],
                    -f["body_coverage"] if lookup and not f["exact_title"] else 0,
                    -f["agency_match"],
                    -f["measurement_strength"] if quantitative else 0,
                    f["fused_rank"])
        reranked.sort(key=key)
    rerank_end = time.perf_counter()
    final, signatures, suppressed = [], [], []
    for artifact_id in reranked:
        signature = _signature(by_id[artifact_id].original_text_chunk)
        if mode == "hybrid_rerank" and any(_duplicates(signature, prior, settings.duplicate_threshold) for prior in signatures):
            suppressed.append(artifact_id)
            continue
        final.append(artifact_id)
        signatures.append(signature)
        if len(final) >= top_k:
            break
    finished = time.perf_counter()
    trace.update({"dense_candidates": dense, "lexical_candidates": lexical,
                  "document_candidates": doc_rank, "document_chunk_candidates": document_pool,
                  "fused_ranking": fused, "reranked_ids": reranked,
                  "metadata_signals": features, "duplicate_suppressed_ids": suppressed,
                  "final_artifact_ids": final, "final_document_ids": [by_id[i].document_id for i in final],
                  "index_ms": round((indexed - started) * 1000, 3),
                  "dense_ms": round((dense_end - indexed) * 1000, 3),
                  "lexical_ms": round((lexical_end - dense_end) * 1000, 3),
                  "retrieval_ms": round((retrieved_end - indexed) * 1000, 3),
                  "rerank_ms": round((rerank_end - retrieved_end) * 1000, 3),
                  "selection_ms": round((finished - rerank_end) * 1000, 3),
                  "total_ms": round((finished - started) * 1000, 3)})
    if diagnostics is not None:
        diagnostics.update(trace)
    return [by_id[i] for i in final]
