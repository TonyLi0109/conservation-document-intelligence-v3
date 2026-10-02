# CDIRP V3 Baseline Freeze

> **System:** CDIRP V3
> **Production baseline:** `v3.0-experimental-baseline` (September 17, 2026)
> **Reviewed IR benchmark revision:** `CDIRP-V3-IR-V2-REVIEWED`
> **Benchmark tag:** `v3.1-ir-benchmark-v2-reviewed`
> **Reviewed snapshot:** October 1, 2026
> **Document updated:** October 2, 2026
> **Live deployment:** <https://conservation-document-intelligence-v3.streamlit.app/>

## 1. Scope and version boundary

This file is the current reproducibility contract for CDIRP V3. It freezes two related but distinct layers:

1. **Production RAG baseline:** the deployed answer-generation pipeline, using hybrid retrieval and a final evidence limit of `top_k=6`.
2. **Reviewed IR benchmark:** the page-level retrieval experiment over nine diagnostic questions, using Top-20 candidate runs, a frozen pool of 761 query-page pairs, three independent blind judges, consensus qrels, and 37 human-reviewed evidence-facet core groups.

The original production tag remains immutable. The reviewed IR benchmark receives its own tag rather than moving `v3.0-experimental-baseline`. The companion [`v3_baseline_config.json`](v3_baseline_config.json) records both layers in machine-readable form.

## 2. Current frozen identities

| Layer | Frozen identity |
|---|---|
| System | CDIRP V3 |
| Production tag | `v3.0-experimental-baseline` |
| Production freeze | 2026-09-17 |
| Reviewed benchmark ID | `CDIRP-V3-IR-V2-REVIEWED` |
| Reviewed benchmark tag | `v3.1-ir-benchmark-v2-reviewed` |
| Reviewed snapshot | 2026-10-01 |
| Python runtime | Python 3.12 |
| Diagnostic questions | Q1–Q9 |
| Live deployment | Streamlit Community Cloud, tracking `main` |

Provider-backed calls are not bit-for-bit deterministic. Reproducibility means that inputs, prompts, model identifiers, candidate depth, consensus rules, artifact hashes, and evaluation definitions are fixed and reported.

## 3. Production RAG baseline

### 3.1 Models

| Component | Frozen value |
|---|---|
| Default reasoning engine | `gpt-5.6-sol` |
| Alternate UI model | `gpt-4.1-mini` |
| Embedding model | `text-embedding-3-small` |
| Embedding dimension | 1,536 |
| Maximum chatbot output | 2,500 tokens |

A response generated with the alternate UI model is not a production-baseline generation result.

### 3.2 Retrieval architecture

The production path uses `hybrid_rerank`:

1. Normalize the query and resolve conversation context.
2. Detect exact document titles and stable document IDs.
3. Retrieve lexical SQLite FTS5/BM25 candidates.
4. Retrieve dense candidates when embeddings are available.
5. Add document-level candidates and representative chunks.
6. Fuse rankings using reciprocal rank fusion.
7. Apply query-aware reranking and duplicate suppression.
8. Apply document-level diversity selection for explicitly targeted multi-document comparisons.

| Parameter | Frozen value |
|---|---:|
| Final answer evidence `top_k` | **6** |
| Dense candidates | 48 |
| Lexical candidates | 48 |
| Fused candidates | 96 |
| Rerank candidates | 64 |
| Document candidates | 8 |
| Chunks per document candidate | 6 |
| RRF `k` | 60 |
| Duplicate threshold | 0.85 |
| Diversity decay | 1.0 |

`V3_TOP_K=6` applies to the deployed RAG context. It does **not** define the research benchmark candidate depth.

### 3.3 Routing and lifecycle rules

- Exact titles and uniquely resolved comparison targets become hard document scopes.
- Years embedded in recognized titles do not force a temporal-comparison route.
- Whole-query straight or curly quote wrappers are normalized before routing and retrieval.
- Planning periods are extracted with:

```regex
\b(?P<start>20\d{2})\s*(?:[-\u2013\u2014]|to|through)\s*(?P<end>20\d{2})\b
```

An active planning-period end year can establish a current operational horizon. Publication, revision, effective, catalog, and planning dates remain distinct lifecycle events.

### 3.4 Provenance and presentation

Claim-validation states are:

- `SUPPORTED`
- `PARTIALLY_SUPPORTED`
- `UNSUPPORTED`
- `INSUFFICIENT_EVIDENCE`

`PipelineTracer` records the eight junctions from `original_query` through `final_output`. The UI keeps citations adjacent to claims, separates **Validated Findings** from **Remaining evidence gaps / Unsupported facets**, suppresses internal diagnostic strings, derives requested headings from the question, and emits an implementation sequence when explicitly requested.

## 4. Frozen page corpus for the IR benchmark

| Property | Frozen value |
|---|---|
| Source documents | 36 |
| Page records | 1,884 |
| Canonical page ID | `DOC###_P####` |
| Artifact | `benchmark/corpus_pages.jsonl` |
| SHA-256 | `df1c18ea6c862e577309990d0b537e072210848b17a31078f6bc0f30a21b9e07` |
| Questions | `benchmark/questions.csv`, Q1–Q9 |

The page corpus contains extracted source text and is an authorized-project artifact. It is excluded from the public repository and must be transferred through the approved private reproducibility package.

## 5. Candidate generation and frozen pool

### 5.1 Ten evaluated systems

| System ID | System | Frozen depth |
|---|---|---:|
| `bm25` | BM25 | 20 per query |
| `dense` | `all-MiniLM-L6-v2` dense retrieval | 20 per query |
| `hybrid` | BM25 + dense RRF | 20 per query |
| `llm_a` | GPT-5.6 Sol | exactly 20 per query |
| `llm_b` | Claude Sonnet 5 | exactly 20 per query |
| `llm_c` | Gemini 3.1 Pro Preview | exactly 20 per query |
| `gpt_6_luna` | GPT-6 Luna | exactly 20 per query |
| `claude_haiku` | Claude Haiku 4.5 | exactly 20 per query |
| `gemini_flash` | Gemini 3.7 Flash | exactly 20 per query |
| `copilot` | Sponsor-supplied Microsoft Copilot run | 16 pages total at available depth |

The six API-backed LLM retrievers use the same corpus-only prompt and must return exactly 20 distinct pages per query. Internal hierarchical-reduction calls may use smaller batch targets; the final run depth remains exactly 20.

Copilot is retained as the tenth external system. Its supplied run contains 1–3 pages per query, so it may be evaluated at available depth but must not be described as a Top-20, Top-50, or Top-100 run.

### 5.2 Pool identity

| Property | Frozen value |
|---|---|
| Candidate pool | `benchmark/pools/candidate_pool_v2_761.jsonl` |
| Unique query-page pairs | **761** |
| SHA-256 | `c41e97b6a62144f1aae59d830ec7e425daeed05729e6e6d062e5c69269597216` |
| Deduplication key | `(query_id, page_id)` |

Candidate provenance, retrieval rank, and selection count are retained for retrieval diagnostics but removed before judge prompt construction.

## 6. V2 independent blind judging

| Setting | Frozen value |
|---|---|
| Prompt version | `core-evidence-v2-2026-09-30` |
| Judge A | `openai/gpt-5.6-sol` |
| Judge B | `anthropic/claude-sonnet-5` |
| Judge C | `gemini/gemini-3.1-pro-preview` |
| Input per judgment | Question, page ID, document title, complete page text |
| Hidden from judges | Retriever identity, provenance, rank, selection count, other judge outputs |
| Records per judge | 761 |

Each judge returns:

- `grade` from 0 to 4;
- `core_facet` when the page is graded 4;
- `redundancy_group_hint` when applicable;
- a short `rationale`.

A formal consensus requires all three judgments. A missing grade is an incomplete triplet; it is neither replaced by a two-judge mean nor imputed as zero.

## 7. Consensus qrels

For grades `r1`, `r2`, and `r3`:

```text
MeanRel = (r1 + r2 + r3) / 3
Range = max(r1, r2, r3) - min(r1, r2, r3)
Variance = sum((ri - MeanRel)^2) / 3
```

- **Relevant:** `MeanRel >= 2.5` and at least two judges assign `grade >= 2`.
- **Page-level core:** `MeanRel >= 3.5` and at least two judges assign `grade == 4`.
- **Final grade:** nearest integer using half-up rounding, clamped to 0–4.

| Artifact | Frozen value |
|---|---|
| Consensus qrels | `benchmark/qrels/consensus_qrels_v2.csv` |
| Rows | 761 |
| Relevant pages | 127 |
| Page-level core pages | 62 |
| SHA-256 | `f7c2fc3177bdc0182d5e5dc1523cd07393cf96c693457e144b2ef532978489db` |

## 8. Human-reviewed core groups

Page-level core pages can be redundant. Human review consolidates pages that support the same indispensable evidence facet.

| Property | Frozen value |
|---|---|
| Artifact | `benchmark/qrels/core_groups.json` |
| Reviewed groups | **37** across Q1–Q9 |
| Assigned page-level core pages | 62 of 62 |
| SHA-256 | `eb60a35182df4e44b6c9764111631e02044a4c58f0c4c9905329b27363770187` |

`CoreGroupRecall@20` gives credit when a run retrieves at least one member of a reviewed group. This prevents equivalent pages from inflating the denominator.

## 9. Reviewed ten-system Top-20 evaluation

Metrics are macro-averaged across Q1–Q9. All retrieved pages in the frozen runs have judged coverage `1.0000`.

| System | Relevant Recall@20 | Page CoreRecall@20 | CoreGroupRecall@20 | nDCG@20 | MRR |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.3945 | 0.3632 | 0.3870 | 0.3709 | 0.4027 |
| Dense | 0.4157 | 0.4637 | 0.5315 | 0.4731 | 0.4660 |
| Hybrid RRF | 0.5112 | 0.5576 | 0.5815 | 0.5563 | 0.6671 |
| GPT-5.6 Sol | 0.8277 | 0.9117 | 0.9889 | 0.8981 | 1.0000 |
| Claude Sonnet 5 | 0.5754 | 0.6935 | 0.7519 | 0.7168 | 1.0000 |
| Gemini 3.1 Pro | 0.7438 | 0.7876 | 0.8259 | 0.8549 | 1.0000 |
| Copilot, available depth | 0.0629 | 0.0699 | 0.1037 | 0.0770 | 0.2593 |
| GPT-6 Luna | 0.7726 | 0.8681 | 0.9222 | 0.8979 | 1.0000 |
| Claude Haiku 4.5 | 0.3928 | 0.4807 | 0.5833 | 0.6049 | 0.8611 |
| Gemini 3.7 Flash | 0.8317 | 0.9276 | 0.9667 | 0.9335 | 1.0000 |

Primary artifact: `benchmark/analysis/system_evaluation_v2_reviewed.csv`, SHA-256 `e97323aeb8b7b87685ca6c278fca0c85a3c747cebbd5132122df30feb7eb3fb2`.

Copilot's row reports its frozen available depth. Its lower recall is not a like-for-like Top-20 result.

## 10. Repeat-judging stability

A complete independent repeat judged the same 761 pairs with the same prompt and model configuration.

| Judge | Exact grade agreement | MAE | Quadratic weighted kappa | Grade-4 Jaccard |
|---|---:|---:|---:|---:|
| GPT-5.6 Sol | 0.8844 | 0.1156 | 0.9660 | 0.8105 |
| Claude Sonnet 5 | 0.8896 | 0.1143 | 0.9656 | 0.7922 |
| Gemini 3.1 Pro | 0.8883 | 0.1156 | 0.9619 | 0.8358 |

Consensus membership stability:

- Repeat relevant pages: 128; reviewed baseline: 127.
- Repeat provisional core pages: 60; reviewed baseline: 62.
- Relevant-page Jaccard: 0.9173.
- Core-page Jaccard: 0.8485.

The repeat groups remain provisional and do not replace the 37 human-reviewed baseline groups.

## 11. Depth diagnostics

Independent Top-100 depth runs exist for the nine programmatic retrievers. Each contains exactly 100 unique pages per question. Their Top-20 prefixes can differ from the frozen primary Top-20 runs because LLM retrieval is variable; depth-run prefixes must not silently replace the reviewed Top-20 runs.

The original six-system frozen Top-20 union covers 36 of the 37 reviewed core groups. The remaining group is Q3_G10, covering floodplain/agricultural wetland restoration outcomes and flood-risk management. Later depth runs recover it, including Gemini Flash at rank 10 and Dense at rank 95.

## 12. Public repository and private artifacts

The public GitHub repository contains source code, prompt definitions, tests, questions, configuration contracts, and onboarding documentation. The following artifacts contain full page text or research outputs and remain outside the public repository:

- `benchmark/corpus_pages.jsonl`
- `benchmark/pools/candidate_pool_v2_761.jsonl`
- `benchmark/judgments_v2/`
- `benchmark/qrels/consensus_qrels_v2.csv`
- `benchmark/qrels/core_groups.json`
- `benchmark/candidate_runs/`
- `benchmark/analysis/`

Authorized collaborators receive these through the approved private reproducibility package. API keys, `.env.local`, billing information, and secrets are never included.

## 13. Reproduction workflow

```powershell
git clone https://github.com/TonyLi0109/conservation-document-intelligence-v3.git
cd conservation-document-intelligence-v3
git checkout v3.1-ir-benchmark-v2-reviewed
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env.local
```

After placing the authorized frozen artifacts at their documented paths:

```powershell
python -m pytest -q

python scripts/run_llm_judges.py `
  --pool benchmark/pools/candidate_pool_v2_761.jsonl `
  --questions benchmark/questions.csv `
  --corpus benchmark/corpus_pages.jsonl `
  --catalog data/source_catalog.csv `
  --output-dir benchmark/runs/<run_id>/judgments `
  --resume

python scripts/build_consensus_qrels.py `
  --pool benchmark/pools/candidate_pool_v2_761.jsonl `
  --judgment-dir benchmark/runs/<run_id>/judgments `
  --full-output benchmark/runs/<run_id>/qrels/consensus_qrels_v2_repeat.csv
```

Do not overwrite reviewed baseline artifacts. Every rerun must use a run-specific directory and record the commit, prompt version, model IDs, commands, timestamps, retries, and environment.

## 14. Production offline regression gate

The deterministic production suite remains separate from the IR benchmark:

| Property | Frozen value |
|---|---:|
| Artifact | `evaluation/baselines/offline.json` |
| Production evidence `top_k` | 6 |
| Cases | 83 |
| Passed | 83 |
| Failed | 0 |
| Seed | 0 |
| SHA-256 | `c1bd699c1d9f5dc548482d760345f04670bc7bec335a2f8bb878873c292d719e` |

## 15. Known limitations

- Frontier APIs can vary across repeated calls despite fixed configuration.
- The public repository cannot reproduce judgments without the authorized private corpus and frozen pool.
- Copilot has only 16 sponsor-supplied candidates and is not a full-depth comparator.
- Free-text facet labels vary more than grades or page membership; page, Grade-4, group-member, and co-membership stability are more defensible.
- A comparative generator response can omit a micro-transaction number even when its page survives retrieval; this is a generation-completeness limitation.
- The current runner preserves parsed judgments and checkpoints; complete raw provider response and structured per-attempt audit logging remain a documented improvement area.

## 16. Change control

A new named experiment or version is required for changes to any of the following:

- corpus membership, extraction, canonical IDs, or frozen hashes;
- candidate systems, model IDs, prompt, or candidate depth;
- judging prompt, judge models, schema, or missing-judge policy;
- relevant/core thresholds or final-grade rounding;
- human-reviewed core-group membership;
- run files, metric definitions, or aggregation;
- production retrieval, lifecycle, provenance, generation, or presentation behavior.

Never move either frozen tag. Corrections require a new commit and a new versioned tag with an explicit relationship to these baselines.
