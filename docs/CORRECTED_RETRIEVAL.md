# Corrected PDF retrieval benchmark

This branch packages the corrected 2026-10-21 retrieval workflow. It reads all 36 PDFs
from `PDF/`, including the corrected 40-page DOC014 Habitat Matters report. The prior
one-page archived-web DOC014 is rejected before inference. Each page receives its
physical `DOCnnn_Pnnnn` identifier; converted PDFs use their actual PDF page numbers.

## Local inputs and setup

Place these locally (not included in the commit):

- `PDF/`: the approved 36-document, 2,036-page PDF corpus.
- `Revised_9_Questions.docx`: the authoritative nine-question document. The parser uses
  the **REVISED QUESTION** column, including Q3, Q8 and Q9.
- `Proposed_Core_Groups_Page_Level_Reference.docx`: archived for provenance only;
  neither proposed groups nor old qrels enter inference.
- `data/corpus.db`: KnowledgeStore seed. The worker copies it and rebuilds its index
  from the frozen PDF text in an isolated runtime database.

```bash
python -m pip install -r requirements.txt -r scripts/requirements-corrected-retrieval.txt
python -m unittest discover -s tests -p 'test_corrected_pdf*.py'
python scripts/run_pdf_benchmark.py prepare
python scripts/audit_corrected_pdf_sources.py --root benchmark/revised_pdf_corrected_20261021
python scripts/run_pdf_benchmark.py run
python scripts/export_corrected_pdf_handoff.py --root benchmark/revised_pdf_corrected_20261021
```

Run commands from the repository root. The original workspace placed helper scripts
one directory above `new-V3`; this branch adapts paths to a normal single-repository
checkout. Do not resume historical runs against these changed code hashes: prepare
a fresh run directory. `--root` selects another output directory.

The native CDIRP generator uses its existing OpenAI API configuration; configure the
required local credentials without committing them. The three model baselines use
authenticated Codex, Claude and Antigravity CLIs. Exact configured models are
`gpt-6-luna` (low), `claude-haiku-5-5` (low), and `gemini-3.8-flash-low`.
Provider availability depends on the account and CLI installation; unavailable models
fail explicitly. No automatic model or billing-route substitution occurs.

## Candidate definitions and exports

The experiment-root `candidate_pool.json` is the exhaustive **audit union**. It keeps
all recorded retrieval stages, including broad temporal scans and dense rankings.
It is not the recommended QGCG construction pool.

`handoff/candidate_pool.json` and `handoff/pools/Q*/pages.jsonl` are the admitted QGCG
union. V3 admission includes native reranked evidence, generation/final evidence and
fused-ranking candidates. Raw dense/lexical scans and temporal evidence scans remain
in audit outputs. For the LLM arms, every accepted model-selected candidate is kept.

`handoff/top100/<method>/Q*.json` contains up to 100 admitted pages per method/question,
without padding. `top100_candidate_pool.json` and `top100_pools/` contain that union.
V3 ordering prioritizes final evidence followed by fused candidates in observed order;
this is a documented export ordering, not a single native global relevance score.

The ZIP also contains question texts, the source question DOCX, PDF inventory hashes,
original stage occurrences/rankings, answers and a file-hash manifest. Canonical IDs
belong to this PDF snapshot; do not join historical text-derived qrels without remapping.

These four arms are not standalone BM25/Dense/Hybrid-RRF experiments. Model retrieval
scores are not independent relevance judgments. No adjudicated ground truth is produced.

## Resuming and observing

```bash
python scripts/watch_pdf_progress.py
python scripts/resume_pdf_method.py --root benchmark/revised_pdf_corrected_20261021 --method gemini38_flash
python scripts/report_pdf_benchmark.py --root benchmark/revised_pdf_corrected_20261021
```

The desktop monitor is read-only (Tk required). Resume only a stopped arm. Completed
questions and accepted batches are reused; failed attempts are retained and new attempts
are numbered separately. Three consecutive call failures stop that arm. Quota exhaustion
requires waiting for the provider reset. Resuming an arm does not automatically coordinate
with every other worker: export only when all workers are quiescent. Use a single export
process; the optional watcher is intended for the primary runner's completion.

Native CDIRP source-list fallback outputs remain failed answers, although completed
retrievals with saved terminal fallback outputs can contribute to a frozen evidence pool.
The corrected observed run had 30 successful answers and 6 native fallback outputs;
this is an observed outcome, not a promised reproduction result.

## Audit and source provenance

All PDFs, questions, runtime source and outputs are hashed. The source audit compares
the current PDFs with frozen copies and verifies extracted page text. Blank physical
pages retain IDs; no OCR is performed. Luna requests are paced at least 20 seconds apart.
The baseline models scan the complete corpus in bounded batches and answer from their
top 30 pages; CDIRP uses native retrieval and top_k=6. Compute budgets are not matched.

`run_qgcg_pilot.py` and `build_qgcg_groups.py` are included as dependencies for shared
request/response types and schema utilities; this workflow does not execute QGCG
group construction. `database.py` includes the scored vector-search implementation
used by the corrected experiment, including native trace preservation.

No experiment outputs, PDFs, local authentication profiles, `.env` files, or changed
corpus database are part of this code commit.
