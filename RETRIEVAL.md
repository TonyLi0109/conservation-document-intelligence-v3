# V3 retrieval

V3 combines indexed lexical search, the existing dense vector search, document metadata retrieval, and bounded deterministic reranking. The output remains a list of canonical `KnowledgeArtifact` objects: source IDs, text, pages, and citation URLs come from SQLite ingestion records. The retrieval layer adds no answer-generation or external reranker API calls.

## What was inspected

Before this change, `database.py` offered separate semantic and keyword routes. Semantic retrieval used the persisted embeddings and NumPy cosine search. Keyword retrieval scanned canonical title/body text and scored substring frequency, term coverage, and title matches. The legacy document matcher required the subject terms to appear together in one chunk. These interfaces remain available for compatibility and the historical benchmark.

Two comparison repositories were available locally beside `new-V3`. Their authorship was not verified, so this document identifies implementations by repository and path rather than attributing them to Charles.

| Local comparison implementation | Useful mechanism | V3 decision |
| --- | --- | --- |
| `conservation-document-intelligence-main/conservation-document-intelligence-main/src/conservation_intelligence/repository.py`: `keyword_search`, `reciprocal_rank_fusion` | SQLite FTS5 BM25; fusion by canonical chunk rank; duplicate IDs counted once per ranking | Adapt these small mechanisms to V3's integer artifact IDs and existing store. |
| Same repository, `chatbot.py`: `_rank_by_scope_coverage`, `_entity_action_proximity_score`, `_select_facet_balanced_evidence` | Explicit query facets, nearby subject/action signals, and selection of complementary evidence | Use bounded query-aware features. Its extensive topic-specific rules and rigid per-document cap were not imported. |
| Same repository, `repository.py`: `fetch_adjacent_chunks`, `evidence_quality_issues` | Document-local neighbor lookup and bibliography/table-of-contents screening | Reference-density screening is implemented. Automatic neighbor expansion is not part of this change; V3's global integer IDs must not be treated as document-local neighbor IDs. |
| `conservation-intelligence-master/conservation-intelligence-master/scripts/semantic_search.py` | Smaller embedding windows, overfetching, and grouping by `original_chunk_id` before canonical SQLite hydration | A possible future long-chunk representation change. No Chroma, SentenceTransformer model, or new embedding-window migration was introduced. Its use of “hybrid” elsewhere also includes deterministic-plus-LLM answering; it should not automatically be read as lexical/dense rank fusion. |

V3's existing chunking, source registry, contextual question resolution, lifecycle policies, Wiki compilation, and provenance validator remain the architectural boundaries.

## Candidate generation and ranking

`retrieval_index.py` owns two persisted FTS5 external-content indexes:

- `retrieval_chunks_fts`: canonical `knowledge_artifacts` title/body text, with `artifact_id` as FTS row ID and `document_id` unindexed.
- `retrieval_documents_fts`: catalog title, agency, topic, and year, with `document_id` unindexed. Metadata candidates seed report-specific chunk searches; metadata alone is not answer evidence.

Both use `porter unicode61` tokenization and SQLite's default equal BM25 column weights. FTS normalizes token matching without rewriting canonical text. Query input becomes safely quoted OR terms; bounded, explicitly quoted phrases are additional literal phrase alternatives. User-supplied FTS operators or SQL syntax are never executed. Document scopes use bound SQL parameters. SQLite documents the external-content trigger pattern and the ascending order of its BM25 scores in the [official FTS5 documentation](https://www.sqlite.org/fts5.html#external_content_tables).

`retrieval.py:retrieve_evidence` performs the following steps:

1. Remove conversational filler from the lexical query and identify exact, report, quantitative, entity, or general semantic intent. Retain quoted phrases and stable document IDs. Alias expansion is limited to the curated groups already used by `chat_context.py`, including invasive/Asian carp and zebra mussels/`Dreissena polymorpha`. Broader-scope follow-ups do not automatically inherit those expansions.
2. Retrieve BM25 chunk candidates and, when the caller provides an embedding, dense candidates from the existing vector cache. An explicit document scope is applied before truncating scoped results. Query embeddings remain a caller responsibility.
3. Retrieve matching document metadata, then search for relevant chunks within each candidate document. Append previously missing document representatives to the lexical ranking, in round-robin document order. This can recover an interior passage from a long report without loading all of that report's text. Exact/report lookups may include the first canonical chunk to recover identifying front matter.
4. Fuse the recovered lexical ranking and dense ranking using reciprocal rank fusion (RRF). Metadata recovery does not give already-matched title chunks a second lexical vote. An absent dense ranking contributes no votes.
5. For `hybrid_rerank`, inspect only the bounded leading candidates, enforce explicit entity scope where applicable, rerank by query-relevant features, and suppress redundant evidence before selecting final K chunks.

For artifact `a`, fusion uses:

```text
RRF(a) = sum(1 / (k + rank_r(a)))
```

The sum includes each ranking containing `a`, with ranks starting at one. Repeated IDs within one ranking get one vote, and ties retain stable first-seen order. The default `k = 60` follows the starting value in [Cormack, Clarke, and Büttcher's RRF paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). It is not a claim that this value is optimal for this corpus. RRF avoids adding incomparable raw cosine and BM25 scores.

The reranker is deterministic Python code, with no learned cross-encoder or LLM judge. It considers exact titles, quoted phrases, reference density, body coverage for remembered-report queries, explicit agency mentions, quantitative outcome cues, and the fused rank. Exact report titles can prioritize identifying front matter. Named entity filtering uses explicit curated aliases; broad themes such as climate change receive softer treatment unless the caller explicitly supplies them as the active entity.

Quantitative cues require a nearby number and outcome language with useful topical association. Generic words such as “management” do not establish subject relevance by themselves. Years, page/figure identifiers, and reference-like contexts are excluded where recognized. A cue helps select evidence; it does not prove causal effectiveness, validate a numerical claim, or make results from different methods directly comparable. Null and adverse outcomes remain eligible evidence.

## Defaults and modes

`config.py:RetrievalSettings` centralizes deployment overrides. Defaults are starting settings, not a fitted claim of optimality.

| Environment variable | Default | Purpose |
| --- | ---: | --- |
| `V3_TOP_K` | 5 | Final answer evidence limit |
| `V3_DENSE_TOP_K` | 48 | Dense candidate limit |
| `V3_LEXICAL_TOP_K` | 48 | BM25 chunk candidate limit |
| `V3_DOCUMENT_TOP_K` | 8 | Metadata candidate documents |
| `V3_DOCUMENT_CHUNK_K` | 6 | Candidate chunks per metadata-selected document |
| `V3_FUSION_TOP_K` | 96 | Retained fused candidates |
| `V3_RERANK_TOP_K` | 64 | Candidates inspected by deterministic reranking |
| `V3_RRF_K` | 60 | RRF rank smoothing constant |
| `V3_DUPLICATE_THRESHOLD` | 0.85 | Five-token shingle containment threshold, subject to additional evidence checks |

When a caller explicitly requests a larger K, candidate limits are raised to at least that K where needed. The document candidate controls bound document-first expansion; they do not cap the number of final chunks from a strong source.

| Retrieval mode | Behavior |
| --- | --- |
| `lexical` | BM25 ranking only |
| `dense` | Existing cosine ranking; a query embedding is required |
| `hybrid` | Rank fusion of lexical candidates extended by document recovery, and available dense candidates |
| `hybrid_rerank` | Hybrid candidates plus deterministic reranking and evidence deduplication |

Production uses `hybrid_rerank` by default. Without an embedding it still runs the lexical/document path. That fallback is not dense semantic search. The ablation's `lexical_rerank` label means lexical ranking with document recovery plus reranking, with no dense vector; it is not another production API mode.

## Integration and provenance

`main.py:ask_chatbot_with_context` resolves the recent conversation into a standalone question before retrieval. A true follow-up may supply its active entity; a partial-context or broader-topic question does not become an unconditional filter for the old species. Conversation text guides intent, while retrieved canonical chunks supply evidence. Page refresh and thread handling remain owned by the existing conversation modules.

The normal chatbot path requests the existing query embedding and uses the hybrid retriever. If embedding generation is unavailable, the indexed lexical/document path remains usable. Existing narrow deterministic answers and temporal answers retain their routes. Document-discovery questions use document-oriented selection over canonical retrieved chunks, rather than generating unsupported report names.

`main.py:search_corpus` exposes the modes to `app.py`. The Search tab offers Hybrid Search, Keyword Search, and Semantic Search. Choosing a chat generation model does not choose a retrieval embedding model. The app prepares the derived index once when opening its writable runtime corpus copy.

Temporal questions go through `temporal.py:select_temporal_evidence`. Hybrid lexical/document retrieval provides candidates; the lifecycle index and verified, dated relationships determine applicability. A newer publication, matching title, or higher retrieval score cannot establish that a report supersedes another. Amendments, supplements, historical comparisons, future effective dates, and unresolved currentness retain the lifecycle policy described in [DOCUMENT_LIFECYCLE.md](DOCUMENT_LIFECYCLE.md).

`wiki_compiler.py:_prepare_evidence` uses the shared retriever without requesting an embedding during local compilation. Its existing sentence scoring, document-aware source selection, exact-span checks, and relationship validation still run afterward. Cached Wiki page opening remains API-free. The optional provider regeneration path retains its existing semantic fallback when local candidates are absent. Adding the lexical index does not invalidate existing compiled Wiki pages.

Final candidate IDs are hydrated through `KnowledgeStore._artifacts_by_ranked_ids`. `KnowledgeArtifact` is unchanged: no ranking score, generated excerpt, alias, or inferred date becomes canonical provenance. The answer validator still resolves opaque evidence handles and verifies exact source spans. Original PDF page conventions, logical page labels, trusted source URLs, and the literal `Web` location remain intact.

Deduplication operates only on selected evidence. Exact normalized copies and sufficiently overlapping chunks can be suppressed, with guards for changed numbers, negation/directional outcomes, and supported measurement contexts. The surrounding quantitative wording helps preserve distinct measurements with the same number. These deterministic checks are conservative heuristics, not a semantic equivalence proof. Canonical rows, vectors, and citations are never deleted to reduce result repetition. Multiple useful chunks from the same document remain allowed; there is no mandatory one-chunk-per-document rule in the shared retriever. Duplicate copies such as DOC007/DOC008 should not be treated as independent corroboration.

## Index initialization and maintenance

Run from `new-V3`, replacing the example path with the intended SQLite corpus:

```powershell
python main.py --database "C:\path\to\corpus.db" --rebuild-retrieval
```

This performs a local derived-index rebuild and prints index statistics. It does not call an embedding or answer API, re-ingest files, renumber artifacts, change canonical text, or regenerate Wiki pages. A missing index is also built automatically on first use, so a full corpus re-ingestion is unnecessary.

Insert/update/delete triggers keep chunk and document postings in the same transaction as canonical mutations, including writes from another SQLite connection. Initialization and forced rebuild use a savepoint and do not commit a caller's outer transaction. Existing stores reopen without a text scan or index rebuild for each query. `index_stats(store)` reports index version, mutation revision, build count, and indexed chunk/document counts. SQLite must include FTS5; no separate search service or Python model download is required.

The index tests cover initial build, reopen, force rebuild, canonical identity/text/vector preservation, metadata updates, deletion, external-connection mutation, literal query handling, explicit document scopes, interior evidence in a long report, and transaction rollback.

## Diagnostics

Pass a dictionary through `search_corpus(..., diagnostics=trace)` or directly to `retrieve_evidence`. Chat answers attach retrieval details to their existing context diagnostics. The trace records:

- Original and expanded query, query type, aliases, mode, and dense availability.
- Dense, lexical, document, and document-chunk candidate IDs; fused and reranked IDs.
- Candidate metadata/outcome features, suppressed duplicate IDs, and final artifact/document IDs.
- Index/setup, dense, lexical, combined retrieval, reranking, final selection, and total durations.

These diagnostics explain ranking decisions without altering citation data. They include query text and are intended for development/evaluation inspection; the ordinary answer remains focused on evidence. The timings begin inside retrieval, so query-embedding API latency and initial corpus loading are outside that measured interval. Temporal selection has its separate applicability diagnostics.

## Evaluation and ablations

The existing 13-query retrieval benchmark is retained. Its usual keyword/vector path retrieves 20 candidate chunks, collapses them to at most five unique documents, and scores those document IDs; its document-matching cases retain their dedicated route. It is useful for historical regression comparison but does not measure the exact five chunks delivered for synthesis.

`evaluation/retrieval_quality.py` evaluates the final canonical evidence returned at K, normally five. Document metrics are calculated from the unique documents actually present in that final evidence, alongside inspected exact-span hits, quantitative evidence hits, wrong-topic results where explicitly labelled, and duplicate evidence rate. The duplicate audit uses its own normalized-text/shingle comparison rather than copying the production selection predicate. Canonical identity and evidence-span failures are hard failures. Missing partially labelled real positives is reported as a development finding; controlled fixture expectations remain hard invariants.

`evaluation/retrieval_ablation.py` reruns the old questions at the new final-chunk limit and keeps those results separate from the unchanged historical benchmark. It compares `legacy_keyword`, `lexical`, `lexical_rerank`, `dense`, `hybrid`, and `hybrid_rerank` on the same final evidence budget. Run the default offline comparison with:

```powershell
python -m evaluation.retrieval_ablation --corpus "C:\path\to\corpus.db" --output-dir evaluation/reports/retrieval_ablation
```

The runner snapshots the corpus, records dataset/corpus fingerprints and retrieval configuration, and writes per-case JSON plus a Markdown comparison. It records the initial call and repeated warm calls, with warm retrieval/reranking/selection timing. An initial call is not necessarily a fresh-machine cold start: variants share the snapshot and a previous variant may already have built the index. Index time is reported separately. Query-embedding network time and corpus opening are excluded from warm retrieval latency.

Real-corpus labels are partial, inspected development labels rather than exhaustive judgments or an untouched holdout. Absence from the label set does not prove irrelevance. DOC007/DOC008 duplicate one publication and must not inflate claims of independent source coverage. Exact span judgments refer to the stored canonical page convention. A missed positive remains informative even when a different unjudged source might answer the question.

The offline ablation reuses only a query-embedding cache whose query text, model, dimension, and vector values validate. When real query embeddings are unavailable, real dense/hybrid variants are marked skipped; proxy vectors are not substituted. The separate `--live-embeddings` option explicitly opts in to embedding missing benchmark questions. It does not invoke an answer model or external reranker. Synthetic fixture vectors test routing, fusion, and evidence preservation under controlled inputs; their results cannot establish real semantic retrieval accuracy. Retrieval scores alone also do not measure live multi-turn answer quality.

Two conversation fixtures now request `claim_count: 2` from their scripted synthesis provider: the opening carp-method turn in `conversation-E-partial-context-broadens-fish`, and the resumed quantitative-method turn in `conversation-G-resume-old-thread-same-page`. Both valid retrieved methods can therefore appear in fixture context regardless of their changed ranking order. The expected source documents, topic relation, active subject, history use, thread isolation, and forbidden-source assertions remain in place. This adjusts a mechanical fixture assumption about which valid method appears first; it is not a relaxation of context or citation validation and is not a claim of live-model improvement.

Measured before/after conclusions must identify the ranking unit, dataset, vector availability, mode, corpus fingerprint, and latency scope. The old document-level score and the new final-evidence score should not be combined into a single improvement percentage.

## Remaining limits

Dense search is still exact NumPy cosine search over all stored embeddings, with approximately O(Nd) work and O(Nd) resident vector storage for N chunks of dimension d. Scoped dense search currently filters a full ranked search. Metadata tables are small enough to inspect locally, while only a bounded number of candidate bodies enter reranking. This change does not introduce approximate vector indexing or a distributed service.

PDF/text chunking remains approximately 750 words with 100-word overlap by default (`V3_CHUNK_TARGET_WORDS`, `V3_CHUNK_OVERLAP_WORDS`). A PDF chunk may span physical pages and retains its starting page as the canonical location. Existing vectors and chunks are reused; no re-embedding is needed to enable FTS retrieval. Heading-aware chunks, section indexes, and smaller embedding windows could improve long-document representation later, but would require an explicit migration and citation/cache regression checks.

The reranker uses explicit local signals rather than trained semantic entailment. Unseen aliases, heavily paraphrased passages, extraction errors, and tables with distant subject labels can still be missed. Reference-density and measurement cues may also need further evaluation. Changes to candidate limits or ranking signals should be justified by fixed final-evidence benchmarks and latency measurements, while preserving the existing conversation, temporal, Wiki, and provenance tests.

## Measured retrieval release (2026-09-07)

Validation used the bundled 36-document, 964-chunk corpus and 21 distinct real benchmark queries. One explicit batch generated query vectors with the existing `text-embedding-3-small` configuration; all subsequent variants reused the same cached vectors. The four synthetic cases use independently specified fixture vectors. Labels are development judgments, not held-out or exhaustive relevance assessments.

All figures below use final five canonical chunks. Recall/MRR/nDCG operate on unique document IDs in their final-evidence order; exact-span hit is the macro average of inspected page/span coverage. Precision retains K as its denominator. These numbers must not be compared directly to the historical 20-chunk-to-five-unique-documents harness.

### Eight added real-corpus queries

| Variant | Recall@5 | Precision@5 | MRR | nDCG@5 | Exact-span hit | Warm total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy_keyword | 0.7500 | 0.1750 | 0.6875 | 0.6938 | 0.3750 | 62.88 |
| lexical | 0.8125 | 0.1750 | 0.7500 | 0.7522 | 0.5625 | 1.30 |
| dense | 0.6875 | 0.1500 | 0.6875 | 0.6555 | 0.1875 | 0.62 |
| hybrid | 0.8750 | 0.2000 | 0.6875 | 0.7444 | 0.3125 | 4.57 |
| lexical_rerank | 1.0000 | 0.2250 | 0.9167 | 0.9375 | 0.8125 | 52.20 |
| hybrid_rerank | 1.0000 | 0.2250 | 0.7812 | 0.8366 | 0.8750 | 56.04 |

### Thirteen unchanged real query/label definitions

| Variant | Recall@5 | Precision@5 | MRR | nDCG@5 | Exact-span hit | Warm total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy_keyword | 0.8718 | 0.1846 | 0.9231 | 0.9020 | 0.6923 | 64.72 |
| lexical | 0.8718 | 0.1846 | 0.9231 | 0.9020 | 0.6923 | 1.30 |
| dense | 0.9487 | 0.2000 | 0.9615 | 0.9506 | 0.5641 | 0.59 |
| hybrid | 0.9487 | 0.2000 | 0.9615 | 0.9506 | 0.6923 | 4.73 |
| lexical_rerank | 0.8718 | 0.1846 | 0.8846 | 0.8814 | 0.7692 | 76.29 |
| hybrid_rerank | 0.9487 | 0.2000 | 0.9231 | 0.9299 | 0.7692 | 79.79 |

The default hybrid reranker improves inspected evidence coverage, but does not win every document-ranking metric. On the eight added cases, lexical reranking has higher MRR while hybrid reranking finds more labelled spans. On the unchanged thirteen questions, unreranked hybrid has better MRR/nDCG, while reranking raises exact-span coverage. Partial labels also omit valid newly retrieved outcomes, including the DOC016 carp removal finding. No existing relevance judgments were changed to hide this tradeoff.

The two labelled quantitative queries both retrieve their measured outcomes, and all three labelled report lookups rank the target document first. No duplicate final evidence was detected by the independent audit. These denominators are small and do not support a general accuracy claim. The remaining dense-enabled span miss is the semantic paraphrase target DOC001 p139: selected DOC001 p151 and DOC036 p288 may be relevant but are unjudged for that exact span. Offline lexical reranking additionally misses one of the two wetland synthesis spans.

For the added eight queries, warm hybrid retrieval averaged 4.80 ms for candidate retrieval, 48.31 ms for feature computation/reranking, 2.69 ms for selection, and 56.04 ms overall. On the unchanged thirteen, the total averaged 79.79 ms versus 64.72 ms for legacy keyword selection. Raw indexed lexical and dense searches averaged about 1.3 ms and 0.6 ms. These measurements exclude query-embedding network time and corpus loading. A separate disposable-copy FTS build took about 87 ms. Dense storage/scoring remains O(Nd); larger corpora still need capacity measurements.

The original offline document benchmark remains Recall@5 0.8974, MRR 0.9231 and nDCG@5 0.9092 with its original retrieval methods and labels; its wetland recall miss remains visible. It is retained as a historical control, not presented as a measurement of the new pipeline.

All 482 pytest tests passed locally in about 22 seconds. The expanded 80-case offline evaluation passed its declared checks; this does not erase its visible partial-label span warnings. The full suite includes 476 fast and 6 integration tests. No external reranker was added, and cached Wiki pages do not require regeneration.

Local browser checks covered a remembered Lock and Dam 19 report (DOC012),
the 51-glade-site Conservation Health Index finding (DOC036), deliberate 2015
SWAP retrieval, current-guidance preference for DOC036, Wiki version notices,
and refresh clearing chat history. Live carp control and quantitative follow-up
answers cited DOC016's 19-ton removal and 27-percent density decrease. The
zebra-mussel topic switch selected zebra-mussel sources; one generated answer
failed the existing exact-span validator and displayed canonical sources instead.
Retrieval relevance does not guarantee that every model-generated paraphrase
passes provenance validation, and the validator was not relaxed.
