# V3 automated evaluation

V3 has an offline-first, declarative evaluation suite. The current hybrid retrieval
extension is described in [RETRIEVAL.md](RETRIEVAL.md), including real-query-vector
ablations and final-evidence metrics. The latest conversation
behavior is preserved: threads can be resumed within one page; refreshing clears
all page conversations.

## What changed

The former evaluation.py had eight fixed questions, paid semantic/generation calls
by default, Markdown status inference, and a hit-any-positive metric named recall.
Empty relevance labels also counted as retrieval success. There was no ranked
metric suite, baseline comparison, offline runner or dedicated evaluation tests.
Existing 106 tests already protected many Wiki and conversation regressions.

The evaluation package separates case definitions, execution adapters, deterministic
metrics and reporting. The public `from evaluation import run_evaluation` entry is
retained for the UI. It launches a subprocess so scripted provider patches cannot
affect another visitor's concurrent real chatbot request. Normal evaluation does
not load credentials or call providers; an offline provider guard detects mistakes.
The canonical database is copied with SQLite backup and evaluated only in a
throwaway store; caches and metadata in data/corpus.db are not modified.

Useful patterns were reviewed in both local comparison implementations:
- conservation-document-intelligence-main: declarative evaluation specifications,
  distinct development/holdout cases and provider-optional execution.
- conservation-intelligence-master: separate deterministic/hybrid metrics,
  citation integrity and explicit cost/latency reporting.

Those directories were not assumed to identify a particular author. No foreign
runtime code or loose term-overlap-as-groundedness score was copied.

## Files and datasets

- evaluation/dataset.py: validated JSON loaders, canonical-JSON fingerprints,
  immutable-source database snapshots and in-memory fixture stores.
- evaluation/metrics.py: ranked retrieval, citation integrity and claim provenance.
- evaluation/scenarios.py: real V3 Wiki and conversation execution with scripted
  providers only where offline model outputs are required.
- evaluation/run.py: CLI, corpus retrieval and provenance runners, provider guard,
  optional live semantic retrieval/answer generation and report assembly.
- evaluation/reporting.py: JSON/Markdown reports and fingerprint-checked baselines.
- evaluation/judges.py: optional semantic-judge protocol; no default provider.
- evaluation/baselines/offline.json: reviewed deterministic starting baseline.
- .github/workflows/evaluation.yml: fast tests, integration tests and offline
  evaluation/baseline checking on push and pull request, with no provider secrets;
  JSON/Markdown reports are retained as a workflow artifact.

Dataset files under evaluation/datasets:

| File | Coverage |
| --- | --- |
| retrieval.json | 13 development queries over the bundled 36-document corpus (currently 964 canonical chunks); 16 inspected positive judgments with exact source spans and reasons |
| retrieval_quality.json | 8 additional real queries with page/span labels, 4 controlled retrieval fixtures, and 4 explicit reruns of context/lifecycle compatibility cases |
| fixture_corpus.json | 10 explicitly synthetic documents with invented outcomes and hand-authored three-dimensional vectors |
| retrieval_fixtures.json | controlled carp/zebra rankings, a plant distractor and a no-result query |
| provenance.json | valid multi-source numeric answer plus 8 invalid/adversarial examples |
| wiki.json | cached load, local rebuild, scripted regeneration compatibility, invalid refresh, provider failure and missing evidence |
| conversations.json | 11 scenarios: follow-up, independent switch, post-switch pronoun, explicit return, partial scope, isolation, resume, direct threat follow-up, refresh reset, ambiguity and no evidence |
| temporal.json | 22 lifecycle scenarios: 16 synthetic cases and 6 inspected corpus cases; dates, revision relationships, current/historical/comparison selection and follow-up currentness |

Temporal execution is in evaluation/temporal_cases.py. Its deterministic metrics
cover labelled current/historical selection, temporal intent, relation validity,
metadata evidence and unsupported supersession statements. See
[DOCUMENT_LIFECYCLE.md](DOCUMENT_LIFECYCLE.md) for temporal rules, limitations and migration.

The earlier evaluation/cases.json is retained as a legacy reference; its unreviewed
expectations are not silently treated as ground truth by the new runner.

## Commands

Run from new-V3 with dependencies installed from requirements.txt:

```powershell
# Fast deterministic tests (no API calls)
python -m pytest -q -m "not integration"

# Corpus/subprocess integration tests
python -m pytest -q -m integration

# All tests
python -m pytest -q

# Full offline evaluation, JSON + Markdown output
python -m evaluation.run

# Compare with the checked-in baseline; stable regressions return exit code 1
python -m evaluation.run --baseline evaluation/baselines/offline.json --fail-on-regression

# One offline case or a different corpus cutoff
python -m evaluation.run --case corpus_carp_control_effectiveness
python -m evaluation.run --top-k 3 --output-dir evaluation/reports/k3

# Explicit paid opt-in; run only the selected real-corpus case
python -m evaluation.run --live --model gpt-4.1-mini --case corpus_carp_control_effectiveness --output-dir evaluation/reports/live
```

The last command needs the existing OPENAI_API_KEY environment/configuration and
sends the selected benchmark question and retrieved evidence to OpenAI. It adds
live semantic retrieval and grounded generation; it does not convert scripted
conversation/Wiki scenarios into live model evaluations. It was not run for this
release. --live is never enabled by default or in CI. No pricing or monetary total
is invented because token usage/pricing are not measured by this harness.

The website Evaluation tab also defaults to offline, displays modes separately,
and provides JSON/Markdown downloads. Paid evaluation requires selecting its
explicit live checkbox before running.

Outputs are evaluation/reports/latest.json and latest.md, ignored by Git. Exit 0
means all configured invariants passed; exit 1 means an invariant/regression failed;
exit 2 indicates invalid configuration. A failed retrieval produces a failure row
rather than dropping the case or exposing raw provider exception details.

## Measurement definitions

Retrieval is evaluated at document level: request 4K canonical chunks from the
existing retriever, preserve their order, deduplicate document IDs and evaluate the
first K documents. Document lookup uses its existing top-K document path. Corpus
K defaults to 5 and --top-k overrides it; synthetic cases keep their declared cutoff.
This is explicitly a document benchmark, not a score of the UI's first K chunks.

- Recall@K: retrieved positively judged documents / all positively judged documents.
- Precision@K: positively judged documents in top K / K, including short-result lists.
- MRR: reciprocal first relevant rank in the supplied top-K document list.
- nDCG@K: graded gain 2^grade-1 with log2 discount, normalized against judged ideal.
- Hit@K: at least one positive; reported separately from recall.

Duplicate IDs are counted by the metric when supplied. No positive judgments yields
null quality scores, not a vacuous pass. Real-corpus labels are PARTIAL: unjudged
results are not proven irrelevant. Thus precision and nDCG are judged-label proxies,
and recall measures recovery of known positives, not all relevant corpus documents.
The benchmark was curated with corpus access; it is not a held-out research set.

Citations are checked against the actual canonical artifact identity: document,
title, page, text and source URL where supplied. Checks include unknown or malformed
citations, fake page/title, cited-source membership, duplicate citations/sources,
uncited claim-like sections, and sources-only fallbacks. Unknown coverage denominators
are null. Lack of a source URL is reported separately where URLs are optional.

Claim checks validate schema, request-local evidence handles, every span's verbatim
ownership and each source's contribution. Offline fixtures independently check
canonical store membership; live raw claims check the captured retrieval artifacts
and report independent canonical membership as unassessed. Numeric-token
coverage is an additional lexical proxy. Matching numeric tokens and verbatim spans
DO NOT establish claim-level semantic entailment (e.g. increased vs decreased).
The numeric-substitution adversarial case deliberately exposes this distinction:
the evaluator detects the mismatch even though the production exact-span validator
alone does not enforce it. No fabricated accuracy is reported from these fixtures.
Optional live answer rows expose raw numeric/provenance metrics and fail when the
generated payload violates those checks, even if the rendered citations are valid.
Report-lookup cases declare `expect_claims: false` to allow canonical source lists;
their citation coverage remains null instead of claiming complete answer coverage.

The optional `assess_semantic_support(payload, artifacts, judge=callable)` accepts
an explicitly supplied judge returning supported/partially_supported/unsupported.
With judge=None it returns not_assessed and calls nothing. External implementations
must supply their own provider, cost controls and model/version metadata; no LLM
judge result is included in this release's baseline.

Wiki checks require the expected schema, a nontrivial overview, facts,
related entities when fixtures support them, supporting evidence, canonical spans
and no duplicate document/page/exact-span identities. Overview length is a content
presence proxy, not semantic grading. Repeated quotations in DIFFERENT documents
retain distinct provenance and are not blindly removed. Corpus-specific thresholds
live in wiki.json rather than rewarding arbitrary verbosity. All page-open checks
forbid API calls. Regeneration is compared structurally using a scripted provider;
that measures integration compatibility, not literary quality of a live model.

Conversation fixtures execute the actual resolver, retrieval, synthesis validator
and thread state. Scripted queries/vectors/outputs are labelled offline_fixture.
They protect data flow and isolation, not the actual model's semantic classification
accuracy. They include the recent threats-to-methods bug and refresh-reset behavior.
The conventional pytest suite additionally checks a nine-turn conversation and
bounded recent-history selection; this is outside the 11 declarative report cases.

## Baselines, adding cases and reproducibility

To add a case, edit the relevant dataset JSON with a unique case_id and the existing
executor mode; no runner changes are needed. For real retrieval labels, first inspect
canonical text and include document_id, stored chunk start page, exact_span and a
reason. Never mark an uninspected document irrelevant. For a synthetic case, label
it synthetic, use fixture documents and declare expected diagnostics/violations.
The runner verifies real judgment spans against the current corpus before scoring.

Reports include UTC timestamp, model/mode, execution settings, case-level metrics,
pass/fail, failure reasons, dataset and corpus SHA-256 fingerprints, Git revision,
Python version and elapsed time. JSON fingerprinting ignores formatting/line endings.
The corpus fingerprint uses the snapshot's logical schema and typed data, including
row identities and vector blobs, before evaluation mutates its disposable cache.
SQLite file headers, page layout and library-version bookkeeping are excluded so
Windows and Linux can compare the same corpus. Committed WAL changes are included.
Hand-authored vectors contain no
random sampling; seed 0 is recorded. LLM reproducibility is not promised.

Baseline comparison requires matching dataset/corpus/configuration fingerprints.
If they differ, comparison is skipped with an explicit warning; --fail-on-regression
also fails an incompatible comparison so CI cannot silently pass it. Stable ranking
drops, missing cases/measurements and pass-to-fail invariant changes are regressions; live metric differences
are warnings. Current deterministic invariant failures still fail the run.

To intentionally replace a reviewed baseline:

```powershell
python -m evaluation.run --write-baseline evaluation/baselines/offline.json
```

This is explicit and refuses a failing run. Do not refresh a baseline just to hide
regressions. The stored baseline omits transcripts, timestamps and runtime duration.

## Initial evaluation release results and limitations

Initial offline run: 42/42 case checks passed in approximately 2 seconds of runner work
(about 3 seconds including process startup on this workstation). This pass count tests
invariants/label integrity; it does not mean every query retrieves every relevant doc.

The lifecycle extension added 22 cases, bringing that release to 64 cases.
The retrieval extension adds 16 rows, bringing the current suite to 80 cases.
Four new rows are compatibility reruns, not independent relevance questions.
The 42-case measurements below describe the original evaluation release and its
unchanged retrieval/Wiki benchmark; temporal scores are reported separately.

The current suite additionally scores actual final-K evidence in the
`retrieval_quality` category. Its real labels are partial: missing an inspected
positive span is visible even when a case passes structural invariants. The
reviewed baseline gates changes in these metrics. Use
`python -m evaluation.retrieval_ablation` to compare retrieval variants; dense
variants require a matching cached query vector, or explicit
`--live-embeddings` opt-in. Missing vectors are skipped rather than approximated.
See [RETRIEVAL.md](RETRIEVAL.md) for the measured comparison and its limits.

Validation on this workstation: 261 pytest cases passed (258 fast tests and 3
integration tests), up from 106 existing tests. Full pytest took about 8 seconds;
the baseline comparison was compatible with zero regressions. No live API calls
were used. Browser inspection confirmed the offline checkbox default, a 42/42
report download, and the cached Species/Invasive carp page with all 96 expandable
evidence entries. Conversation A-G execution traces were manually inspected using
scripted providers; this does not claim a fresh live-model conversation test.

| Actual corpus metric, K=5 | Value |
| --- | ---: |
| Recall of labelled positives | 0.8974 |
| Judged-label Precision | 0.2000 |
| MRR within top K | 0.9231 |
| Judged-label nDCG | 0.9092 |
| Hit rate | 0.9231 |

- corpus_wetland_loss_quantitative misses judged DOC001 in top 5. Some returned docs
  may also be relevant but are unjudged. This is a visible retrieval weakness.
- The carp-control case recovers only part of its known positives. DOC007/DOC008
  duplicate one publication under separate catalog IDs, not independent corroboration.
- DOC003 and DOC012 have catalog-title/year differences from document text; labels
  are justified by stored spans, not assumed catalog-year correctness.
- Invasive carp Wiki: 274-character overview, 8 facts, 96 evidence entries, 1 related entity,
  valid links and zero duplicate evidence identities. DOC036 lacks an optional URL.
- Citation fixtures: valid multi-source answer passes; all 8 negative fixtures expose
  their intended violations. These are not production citation-accuracy percentages.
- Conversation scenarios: 11/11 offline integration cases pass, including A-G and
  refresh clearing. External model routing/generation quality remains unmeasured.
- Provider-stub Wiki refresh and corpus-cached load are schema/provenance compatible;
  live regeneration quality was not scored. Existing genuine citations are unchanged.

Conventional added tests also cover repeated/updated/interrupted ingestion, malformed
and blank PDFs, long-page extraction/fallback telemetry, missing metadata, duplicate
chunks, no-result/long queries, contrasting sources and invalid vectors. Unsupported
metadata-filter/hybrid APIs are not pretended to exist. No external evaluation
service, new application dependency, corpus mutation or ingestion rewrite is needed.
