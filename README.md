# Conservation Document Intelligence V3

V3 is an evidence-grounded research platform over the 35-document legacy
conservation corpus plus the 2022 Missouri Comprehensive Conservation Strategy.
It borrows useful ideas from both reference implementations, but owns its data
contracts, source registry, storage, validation, and knowledge-compilation flow.

## What is distinct about V3

- Raw `KnowledgeArtifact` chunks are immutable evidence, not generated summaries.
- Claims are rendered only after opaque handles and verbatim spans validate.
- Compiled knowledge is stored separately from raw evidence and retains exact
  artifact links, method, model, and compiler version.
- Wiki results are reusable: generation is cached unless the user requests a
  refresh.
- PDF extraction uses isolated workers plus a PyMuPDF fallback and records page
  coverage and source SHA-256.
- Citations preserve physical PDF pages and, when the source defines standard
  PDF PageLabels, also show the logical printed label.
- Offline evaluation measures ranked retrieval, citation safety, Wiki and conversation regressions; live provider runs are opt-in.
- Document lifecycle evidence distinguishes current guidance, historical versions,
  explicit replacements and partial updates without treating every newer source as authoritative.

## Architecture

```text
source_catalog.csv + 36 local sources
                  |
        PDF/TXT parsing and coverage
                  |
        immutable KnowledgeArtifact rows ---- vector cache
                  |                               |
                  +---------- retrieval ----------+
                                  |
                  structured generation + exact-span validation
                         |                    |
             compiled knowledge          chatbot claims
                         |                    |
                        Wiki          citations + source inspection
```

The `documents` registry represents original-source provenance. The
`knowledge_artifacts` table represents canonical evidence. Tables prefixed with
`compiled_` represent derived, reusable knowledge. Derived rows never replace
source text.

## Setup

```powershell
cd new-V3
python -m pip install -r requirements.txt
```

Create `.env.local` locally:

```text
OPENAI_API_KEY=your-new-key
```

Do not commit this file. If a key has ever been pasted into a chat or log,
revoke it and create another one.

## Build the corpus once

```powershell
python main.py --ingest .\pdfs
```

For an existing index, logical labels can be added without parsing or embedding:

```powershell
python main.py --backfill-page-labels .\pdfs
```

Citations deliberately distinguish the two coordinate systems, for example
`printed p. iv; PDF p. 8`. PDFs without trustworthy PageLabels display only
`PDF p. 8`; V3 does not guess a printed label from arbitrary body text.

Ingestion is atomic. A failed rebuild retains the previous searchable corpus.
After upgrading to the coverage-aware parser, run ingestion once to populate the
coverage/SHA-256 report. Failed pypdf pages are retried with PyMuPDF; unresolved
pages remain visible in the Corpus tab rather than being silently treated as
complete.

## Run the application

```powershell
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

The repository includes the read-ready `data/corpus.db`, so a hosted instance
does not need to download, parse, or embed the corpus during startup. Create an
app from this repository with `app.py` as the entrypoint, then add the following
secret in the Streamlit Community Cloud dashboard:

```toml
OPENAI_API_KEY = "your-api-key-here"
```

Never commit `.env.local` or `.streamlit/secrets.toml`. The hosted filesystem is
ephemeral; evaluation exports and regenerated wiki cache entries may be reset
when the app restarts, while the committed canonical corpus remains available.

The UI contains Corpus, Search, Wiki, Chatbot, and Evaluation tabs. Wiki pages
are pre-generated from exact canonical excerpts and appear immediately when an
entity is selected. “Regenerate from evidence” explicitly requests a richer AI
compilation; if that request times out, the existing page remains available.

After ingestion or a compiler upgrade, prepare the deployment database without
an API call. This creates missing pages and upgrades older deterministic pages;
existing AI-refreshed pages are preserved:

```powershell
python main.py --precompile-wiki
```

The local compiler now builds a source-derived introduction, up to eight distinct
facts, conservative explicit relationships, and multiple excerpts per source.
Both builders share evidence selection, the five-section schema, evidence
aggregation, validation, and SQLite persistence. Defaults select up to 12 chunks
across documents with up to eight spans each. AI refresh adds model-written
synthesis; it remains an explicit action. Existing artifacts load directly from
SQLite. A missing or outdated deterministic artifact is repaired locally, with
no model or embedding request.

Run preparation before starting Streamlit. Restart an already running app after
updating its seed database so that its runtime copy and session cache reload.
See [Wiki quality implementation notes](WIKI_QUALITY.md) for the root cause,
regression coverage, measurements, limitations, and manual verification.

## Configuration

### Multi-turn chatbot

Chatbot follow-ups now resolve recent conversation into a standalone question
**before** embedding and retrieval. The existing session message list retains
the resolved query and active subject alongside each assistant response. Only
the most recent six messages are considered; conversation text is intent
context, never evidence. Explicit new topics override prior context.

Use **New conversation** to create a separate thread within the current page.
Use the **Conversation** selector to return to another thread before refreshing.
**Refreshing the website clears all conversation history and starts an empty chat.**
Old browser archives are removed; messages are no longer stored in localStorage.

If an earlier answer lists conservation threats and the user asks how effective
"these methods" are, the chatbot directly searches for measures addressing those
threats and their measured outcomes. It cites available corpus evidence and
acknowledges missing data instead of asking the user to choose an interpretation.
Recognized independent questions add no contextualizer call; other follow-ups use
one bounded request when needed. Unresolvable references can still require
clarification, and provider failures cannot supply invented evidence.

See [multi-turn implementation and verification](CHAT_CONTEXT.md) for details,
limitations, tests and the optional backend diagnostics interface.

Defaults live in `config.py`. Important overrides include:

- `V3_LLM_MODEL`
- `V3_EMBEDDING_MODEL`
- `V3_EMBEDDING_DIMENSION`
- `V3_TOP_K`
- `V3_CHUNK_TARGET_WORDS`
- `V3_CHUNK_OVERLAP_WORDS`
- `V3_EMBEDDING_BATCH_SIZE`
- `V3_WIKI_TOP_K`
- `V3_WIKI_COMPILER_VERSION`

## Reference implementations and V3 decisions

| Area | Reference idea | V3 decision |
|---|---|---|
| Parsing | CharlesChen130 cleanup and PDF extraction | Page-aware chunks, hard process isolation, fallback extraction, coverage report |
| UI/Wiki | shanged entity-oriented exploration and legacy tab structure | Thin Streamlit UI backed by persistent V3 contracts |
| Retrieval | Semantic and keyword approaches from both prototypes | Persistent exact cosine index plus deterministic keyword ranking |
| Synthesis | Corpus-grounded prompting from both prototypes | Atomic claims, strict JSON Schema, exact-span fail-closed validation |
| Knowledge compilation | Legacy Wiki/entity organization | Separate versioned compiled tables with canonical evidence edges and caching |
| Insufficient evidence | Baseline refusal behaviors | Partial-context synthesis with unsupported facets outside factual claims |

Neither legacy repository is imported at runtime. `data/source_catalog.csv` is
the V3-owned manifest initially curated using their public source metadata.

## Evaluation

For document dates, revision families, temporal queries and metadata-only migration,
see [document lifecycle and current-guidance reasoning](DOCUMENT_LIFECYCLE.md).

Run `python -m evaluation.run` for the offline suite and JSON/Markdown reports.
The Evaluation tab uses the same isolated runner without API calls by default.
See [evaluation datasets, metrics, baseline and commands](EVALUATION.md) for details.
Live semantic/answer evaluation requires explicit `--live` or the UI opt-in checkbox.

For the final research comparison, run the same question set against both legacy
systems and preserve their outputs. Compare Retrieval Recall@5, citation
precision, exact-span validity, answer completeness, abstention accuracy,
multi-document synthesis, and latency. Adapters for automated remote baseline
runs are intentionally not embedded in V3 because the legacy deployments may
change; exported baseline outputs can be scored as immutable fixtures.

## Known limitations

- Existing databases do not contain ingestion coverage until they are rebuilt.
- OCR is not yet used for image-only pages; such pages remain explicitly failed.
- The evaluation suite is an initial research fixture and requires professor or
  domain-expert review of expected documents.
- Compiled relations validate their evidence spans, but broader ontology
  normalization remains future research.
