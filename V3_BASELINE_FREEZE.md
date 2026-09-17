# CDIRP V3 Experimental Baseline Freeze

> **Status:** Frozen experimental baseline
> **System:** CDIRP V3
> **Release tag:** `v3.0-experimental-baseline`
> **Freeze date:** September 17, 2026
> **Live deployment:** <https://conservation-document-intelligence-v3.streamlit.app/>

## 1. Purpose and scope

This document defines the reproducible **V3 Experimental Baseline** for the Conservation Document Intelligence and Retrieval Platform (CDIRP V3) before any corpus-scale expansion. It freezes the code path, corpus identity, model selection, retrieval limits, lifecycle policy, provenance boundary, presentation behavior, and offline evaluation gate used for subsequent comparisons.

The Git tag is the authoritative source-code identity. The companion [`v3_baseline_config.json`](v3_baseline_config.json) is the machine-readable parameter contract. A run is a V3 baseline run only when it uses the tagged code, the identified corpus, and the locked configuration below. Any override must be reported as an experimental deviation.

## 2. Release identity

| Field | Frozen value |
| --- | --- |
| System name | CDIRP V3 |
| Release label | V3 Experimental Baseline |
| Git tag | `v3.0-experimental-baseline` |
| Freeze date | September 17, 2026 |
| Implementation parent | `483b7c3f9924ee260d1293675cfc0cab6ce3e7ee` |
| Tag target | The commit containing this document and its machine-readable manifest |
| Python runtime | Python 3.12 |
| Deployment | Streamlit Community Cloud, tracking `main` |

The implementation-parent commit identifies the final functional state before the freeze manifest and K=6 consistency changes were added. The release tag, rather than the parent commit, must be used to reproduce the complete baseline.

## 3. Model configuration

| Component | Frozen value |
| --- | --- |
| Core reasoning engine | **GPT-5.6sol** |
| Runtime model ID | `gpt-5.6-sol` |
| Alternate UI model | `gpt-4.1-mini` |
| Baseline model | `gpt-5.6-sol` only |
| Embedding model | `text-embedding-3-small` |
| Embedding dimension | 1,536 |
| Maximum chatbot output | 2,500 tokens |
| Provider request timeout | 30 seconds |
| Embedding batch size | 100 |

The alternate model remains available for controlled comparison, but a response generated with `gpt-4.1-mini` is not a baseline generation result.

Provider-backed generation is not bit-for-bit deterministic. Reproducibility therefore means configuration, evidence selection, validation rules, and evaluation gates are fixed; it does not claim identical wording across repeated live model calls.

## 4. Corpus identity

The baseline corpus is the curated **36-source Missouri conservation corpus**.

| Property | Frozen value |
| --- | --- |
| Canonical documents | 36 |
| Canonical evidence artifacts | 964 |
| SQLite database | `data/corpus.db` |
| Source catalog | `data/source_catalog.csv` |
| Committed database SHA-256 | `4d5bb333c07db69302fc8bc1617b4ab0de490b89aac62a3ea1016e9334ea6b70` |
| Committed catalog SHA-256 | `71d95c4815d97ed458072a8318d33591dadfef91cbc40a154ae96bef34e916b7` |
| Logical SQLite fingerprint | `sqlite-logical-v1:0d7529068b35d58933d315cf45a41c2d2d77e59df3c9821129288149ccea6b3e` |

The SHA-256 values refer to the Git blobs in the release tag. Runtime-created FTS indexes, caches, SQLite journaling, or local database writes can change the working-file byte hash without changing the reviewed corpus. Use the tagged Git blob and the logical fingerprint for verification.

### Chunking

| Parameter | Value |
| --- | ---: |
| Target words per chunk | 750 |
| Overlap words | 100 |
| Page processing timeout | 60 seconds |

## 5. RCA-refined retrieval architecture

Production retrieval uses `hybrid_rerank` and returns exactly **`top_k = 6`** final evidence chunks when six eligible chunks exist.

### 5.1 Candidate generation and fusion

1. Resolve conversation context and normalize whole-query quote wrappers.
2. Detect explicit document titles and stable document IDs.
3. Generate lexical candidates with SQLite FTS5/BM25.
4. Generate dense candidates with `text-embedding-3-small` when embeddings are available.
5. Recover document-level candidates and representative chunks.
6. Fuse lexical and dense rankings with reciprocal rank fusion (RRF).
7. Apply deterministic query-aware reranking and duplicate suppression.
8. Apply document-level diversity selection for explicitly targeted multi-document comparisons.

The deployed path remains hybrid when dense embeddings are available. If the embedding provider is unavailable, the lexical/document path remains operational, but that fallback must not be reported as dense hybrid retrieval.

### 5.2 Frozen retrieval parameters

| Parameter | Value |
| --- | ---: |
| Final `top_k` | **6** |
| Dense candidates | 48 |
| Lexical candidates | 48 |
| Fused candidates | 96 |
| Rerank candidates | 64 |
| Document candidates | 8 |
| Chunks per document candidate | 6 |
| RRF `k` | 60 |
| Duplicate threshold | 0.85 |
| Diversity decay | 1.0 |

`V3_TOP_K=6` is required for the baseline. Supplying another value through an environment variable, function argument, CLI flag, or test fixture defines a non-baseline experiment.

### 5.3 Document-level diversity penalty

The diversity layer is a simplified document-level MMR mechanism. It activates when a comparison query resolves multiple explicit target documents. It first reserves eligible coverage across unseen targets, then discounts subsequent chunks from a document already represented:

```text
utility(chunk) =
    1 / (60 + fused_rank(chunk))
    / (1 + 1.0 * selected_count(document_id))
```

This policy reduces evidence starvation while preserving canonical chunks and provenance. It is not a semantic confidence score and is not applied indiscriminately to every query.

### 5.4 Title-aware routing

Exact corpus titles and uniquely resolved comparison titles become hard document scopes. Years embedded in recognized titles do not, by themselves, force a `temporal_comparison` route. This prevents year tokens such as `2022` and `2015` from bypassing normal comparison synthesis or allowing unrelated keyword-heavy documents to monopolize evidence.

### 5.5 Lifecycle extraction and currentness

`document_lifecycle.py` extracts planning periods with the frozen rule:

```regex
\b(?P<start>20\d{2})\s*(?:[-\u2013\u2014]|to|through)\s*(?P<end>20\d{2})\b
```

The lifecycle record stores `planning_period.start_year` and `planning_period.end_year` with exact evidence. For current operational guidance, an active planning-period end year can take precedence over an older static publication date. Publication dates, revision dates, effective dates, and planning horizons remain distinct events. A newer date alone does not prove authority, replacement, or supersession.

## 6. Provenance and trace boundary

Every answer claim is validated against opaque evidence handles and exact, contiguous source spans before trusted citations are rendered. Machine-readable claim states are:

- `SUPPORTED`
- `PARTIALLY_SUPPORTED`
- `UNSUPPORTED`
- `INSUFFICIENT_EVIDENCE`

The JSONL provenance record retains claim text, document ID, printed page, PDF page, exact evidence span, and validation status. The user interface never treats retrieval or reranking scores as provenance confidence.

`PipelineTracer` preserves the eight diagnostic junctions:

1. `original_query`
2. `rewritten_query`
3. `initial_retrieval`
4. `reranked_evidence`
5. `llm_context_payload`
6. `draft_answer`
7. `validation_results`
8. `final_output`

## 7. Frozen presentation contract

`output_formatter.py` operates only on claims that passed the existing provenance boundary.

- **Dynamic structural alignment:** headings come from the user's requested dimensions.
- **Duplicate suppression:** each dynamic category heading is emitted once; repeated categories are merged.
- **State isolation:** supported content appears under **Validated Findings**. Missing or rejected facets appear under **Remaining evidence gaps / Unsupported facets**.
- **Actionable sequence extraction:** an explicit request containing `implementation sequence` produces a concise numbered **Implementation Sequence:** immediately before the evidence-gap block.
- **Adjacent citations:** each rendered claim or step retains its own citation.
- **Dual-coordinate locations:** where the two coordinates differ, citations render both, for example `[printed p. 11; PDF p. 27]`.
- **UI hygiene:** raw source excerpts, lifecycle debug strings, catalog conflicts, global machine status, and provenance-failure diagnostics remain in logs rather than user-facing prose.
- **No table completion:** the formatter does not invent content to fill requested categories, table cells, or comparison dimensions.

## 8. Known frozen limitation

### Semantic abstraction / generation completeness

For comparative questions, the generator can prioritize macro-level thematic structures over exact micro-transaction figures. A relevant chunk may successfully survive retrieval and enter the six-chunk context while the generated comparison omits a specific number, such as a **$120,000** funding amount.

This is an accepted baseline behavior. It is a generation-completeness limitation, not necessarily a retrieval failure. Future work must measure and report improvements against this frozen behavior rather than silently changing the baseline.

## 9. Offline evaluation baseline

The reviewed offline baseline is stored at `evaluation/baselines/offline.json`.

| Measure | Frozen result |
| --- | ---: |
| Configuration `top_k` | 6 |
| Seed | 0 |
| Metrics version | 1 |
| Cases | 83 |
| Passed | 83 |
| Failed | 0 |
| Baseline SHA-256 | `c1bd699c1d9f5dc548482d760345f04670bc7bec335a2f8bb878873c292d719e` |

The offline suite blocks provider calls. Its fixture passes establish deterministic pipeline invariants; they do not estimate live-model answer quality. Partial relevance labels also mean precision and nDCG are judged-label proxies rather than exhaustive corpus relevance measures.

## 10. Reproduction procedure

### 10.1 Checkout and environment

```bash
git clone https://github.com/TonyLi0109/conservation-document-intelligence-v3.git
cd conservation-document-intelligence-v3
git checkout v3.0-experimental-baseline
python -m venv .venv
# Activate .venv using the command appropriate for the operating system.
python -m pip install -r requirements.txt
```

Required baseline settings:

```text
V3_LLM_MODEL=gpt-5.6-sol
V3_TOP_K=6
V3_EMBEDDING_MODEL=text-embedding-3-small
V3_EMBEDDING_DIMENSION=1536
```

An OpenAI API credential is required only for live embedding or generation paths. Never store credentials in the freeze manifest.

### 10.2 Verify code and corpus identity

```bash
git describe --tags --exact-match
python -c "import json; print(json.load(open('v3_baseline_config.json', encoding='utf-8'))['baseline_id'])"
```

Expected tag: `v3.0-experimental-baseline`. Expected baseline ID: `CDIRP-V3-EXPERIMENTAL-BASELINE`.

Verify the committed corpus blob without relying on a potentially mutated runtime copy:

```bash
python -c "import hashlib,subprocess; b=subprocess.check_output(['git','show','v3.0-experimental-baseline:data/corpus.db']); print(hashlib.sha256(b).hexdigest())"
```

Expected SHA-256: `4d5bb333c07db69302fc8bc1617b4ab0de490b89aac62a3ea1016e9334ea6b70`.

### 10.3 Run gates

```bash
python -m pytest -q
python -m evaluation.run --top-k 6 --baseline evaluation/baselines/offline.json --fail-on-regression
```

A valid baseline run must complete with no failed tests, no failed evaluation cases, and a compatible baseline comparison.

### 10.4 Launch locally

```bash
streamlit run app.py
```

Confirm at runtime that the selected generator is `gpt-5.6-sol` and that no deployment-level environment variable overrides `V3_TOP_K=6`.

## 11. Change-control rule

The following changes require a new experimental version or an explicitly named ablation:

- corpus membership, source files, parsing, chunking, or embeddings;
- `top_k` or any candidate, fusion, rerank, duplicate, or diversity parameter;
- title/entity routing or lifecycle extraction;
- generation model, prompt, context assembly, or output token budget;
- provenance validation or citation rendering;
- output-template classification, headings, or evidence-gap handling;
- evaluation datasets, fingerprints, metrics, or pass criteria.

Do not move the `v3.0-experimental-baseline` tag. Corrections after release must receive a new tag and a documented relationship to this baseline.
