# Immediately available Wiki quality

## Root cause and path comparison

`app.render_wiki_tab` selects an entity and calls
`generate_extractive_wiki_concept`. Previously this read any cached artifact,
regardless of compiler version, or built a template page. The summary was a
generic description of the compiler, facts and evidence were limited to the
first span of each of five chunks, and related entities were always empty.

The explicit refresh calls `generate_wiki_concept(force_refresh=True)`. It used
the same keyword retrieval and five-chunk limit, but exposed up to eight allowed
spans per chunk to the LLM. Its prompt asked for a summary, facts, relationships
with exact supporting quotes, and evidence. Only this path could synthesize
an introduction or populate relationships. It uses the UI-selected model (or
configured default), temperature zero and a 3,000-token default output budget.
An embedding search is available only if explicit refresh finds no keyword hits.

Both paths already persisted the same five-field concept schema in
`compiled_knowledge`, `compiled_facts`, `compiled_relationships`,
`compiled_relationship_evidence`, and `compiled_evidence`. These tables link to
canonical chunks; document metadata is resolved from trusted storage. There is
no hidden richer precomputed artifact the UI was ignoring, and no separately
persisted ingestion-time entity extraction to reuse. The curated entity catalog
is a vocabulary filtered against corpus text/titles, not a relationship graph.
Existing LLM compilations were already displayed when cached.

Preparation is an explicit `main.py --precompile-wiki` step. Ingestion clears
derived knowledge when replacing the corpus; it does not itself run the Wiki
compiler. Previously preparation only filled missing pages, leaving old
placeholder pages indefinitely.

## Implemented flow

1. `_prepare_evidence` retrieves a bounded keyword candidate pool (four times
   the chunk limit), ranks usable source spans, and prioritizes document
   diversity before filling remaining slots. Both builders use it. Default
   final capacity is 12 chunks and eight spans per chunk, configurable through
   `V3_WIKI_TOP_K` and `V3_WIKI_SPANS_PER_ARTIFACT`.
2. `_extractive_payload` builds the common five-field artifact. Definitions and
   substantive topic statements lead the introduction and facts. Facts are
   deduplicated; quotes from distinct source locations retain provenance.
   Explicit predicate patterns over complete curated entity names can identify
   membership, occurrence, and management relationships. Negated statements and
   mere keyword/chunk co-occurrence do not create these edges.
3. `_finish_compilation` applies the shared validator, merges and deduplicates
   evidence and local relationships, and ensures relationship quotes appear in
   supporting evidence. The existing database write validates canonical source
   ownership; reads revalidate persisted quotes and restore trusted metadata.
4. The deterministic builder persists version `v3.3-extractive`. Preparation
   upgrades older deterministic artifacts while preserving all AI refreshes.
5. Selection reads the persisted artifact. Missing/stale local artifacts have a
   deterministic repair path. The UI no longer has a hot-reload alias fallback
   from the local loader to the LLM builder.
6. Explicit refresh uses the same evidence and local artifact as starting
   material, asks the model for improved synthesis, then uses the same finalizer
   and database. It persists compiler version `v3.3`. Timeout failures retain
   the previous page; invalid JSON or quotes cannot replace valid storage and
   the UI retains its existing page.

This preserves the architecture and database schema. The same explicit refresh
function can be called by a future offline batch job before deployment; its
results will be immediately usable by the existing page loader. No global
paid batch generation was added or run.

## Intentional differences and limits

Local compilation remains extractive, so prose can retain PDF extraction
artifacts and be less fluent than an LLM synthesis. Its introduction reuses
selected facts; glossary labels receive a small grammatical normalization.
Relationship patterns are intentionally conservative and may return fewer
entities than the model. They do not infer relationships from co-occurrence.
The bounded retrieval pool can miss useful evidence outside its candidate set.
Quotes prove source provenance; they do not independently verify every semantic
claim in LLM-written summaries/facts, matching the existing contract.

Evidence display can be longer (up to 96 selected spans at defaults, plus any
additional validated model relationship quotes). This increases artifact and
render size. Only explicit AI refresh has the larger prompt cost. Model settings
other than the compiler version and shared retrieval limit are unchanged.

## Verification and performance

`tests/test_wiki_quality.py` adds nine offline regression tests covering:

- A representative Invasive carp introduction, facts, relationships and rich evidence.
- Source URLs, document/page IDs, printed labels and exact quote preservation.
- Cache round trips and actual entity-selection renderer execution with API
  generation and retrieval forbidden on cache hits.
- Successful mocked regeneration and structural/evidence compatibility.
- Old local artifact upgrades, idempotent preparation and preservation of AI pages.
- Invalid refresh rejection without replacing the cache, and missing evidence.
- Negation, unrelated entity overlap and deduplication without losing distinct sources.

Full suite: `python -m pytest -q` — **27 passed**. `git diff --check` passed.
No live model/embedding requests were made for verification.

On the local 964-chunk corpus, Invasive carp now has eight facts, 96 supporting
excerpts from nine documents, and an explicit Silver carp membership relation.
Its introduction starts with the corpus's definition of invasive carp and adds
context on containment/spread. Twelve deterministic pages in the bundled
database were upgraded; three existing AI-refreshed pages were preserved.

Fifty cached loader calls measured a median of **1.16 ms** and maximum of
**1.88 ms**. A local Invasive carp rebuild took **101 ms**. These are local
Python/database measurements, excluding Streamlit rendering, entity-catalog
queries, cold process startup, and network/browser time. Prepared selection
performs no retrieval, embedding, or LLM request. The existing catalog-query
and session-state behavior is unchanged.

## Refresh and manual verification

From `new-V3`, run after ingestion or when updating an older deployment:

```powershell
python main.py --precompile-wiki
python -m pytest -q
streamlit run app.py
```

No schema migration is needed. The included seed database has already been
refreshed. Restart any running Streamlit process after replacing its seed;
hosted deployments use a runtime database copy and the UI caches the selected
result in session state.

1. Open **Wiki → Species → Invasive carp**. The introduction should define the
   group rather than describe the compiler. Inspect the eight facts, Silver carp
   relationship and expanded supporting evidence.
2. Open several evidence expanders and check the source titles, pages, URLs and
   verbatim excerpts against the corpus.
3. Switch to another entity and back. The page should appear from cache without
   a generation spinner or API request.
4. Click **Regenerate from evidence** with a configured API key. This is the only
   action that requests synthesis. Compare the same sections, retained source
   evidence, and any richer model-written prose/relationships.
5. Switch away and back again; the refreshed artifact should now be the initial
   display. If refresh fails, the existing page should remain available.
