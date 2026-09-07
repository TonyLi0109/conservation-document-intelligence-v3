# Document versions and current guidance

V3 can distinguish an explicit replacement from a newer related publication,
retrieve earlier guidance deliberately, and explain its source selection with
canonical evidence. Its conclusion is bounded by the documents in the indexed
corpus. It does not establish that a document is the latest external publication
or legally controlling guidance.

Previously, the document registry stored a free-text `year`, title, agency and
source URLs. Retrieval did not model revision relationships, applicability or
draft status. A highly similar older chunk could therefore dominate a question
about current guidance. A catalog year could also describe a reporting period or
download rather than the document's publication date.

## Stored metadata and extraction

[document_lifecycle.py](document_lifecycle.py) derives a lifecycle record for each
stable document ID. The canonical document registry, evidence chunks, vectors and
citation contracts retain their existing roles.

| Field | Meaning |
| --- | --- |
| `publication_date`, `revision_date`, `effective_date` | Separate date signals; an unknown value remains null |
| `version` | Recognized version/edition number or explicitly named version year, stored as a string |
| `status` | `final`, `draft`, `withdrawn` or `unknown` |
| `kind` | A title-based document-type cue, such as guidance, strategy, research, meeting or report |
| `family_id`, `family_source` | Candidate/explicit family membership and its basis |
| `relations` | Directional links to uniquely resolved indexed document IDs |
| `dates`, `status_evidence`, `version_evidence` | Signal origin, confidence and supporting location/span when available |
| `signals`, `warnings` | Retained date alternatives, conflicts and unresolved references |

A date signal records `value`, `precision`, `source`, `confidence` and `evidence`.
Explicit evidence contains the canonical document ID, stored page and exact text.
Origins distinguish `explicit`, `metadata_inferred` and `title_inferred`. Catalog
and title fallbacks do not receive fabricated supporting quotations.

Supported dates include ISO dates, English month names and year-only values.
`2024`, `2024-02` and `2024-02-29` retain year, month and day precision respectively.
Invalid dates and ambiguous numeric formats such as `01/02/2024` are rejected.
The literal catalog values `Current`, `Unknown` and multi-year reporting ranges
do not establish a publication date.

The extractor recognizes labelled publication/revision/effective dates,
self-identifying version statements, final/draft/withdrawn labels, and selected
front-matter patterns: publisher imprints, dated preparation credits, suggested
self-citations and web bylines. It reads version relationships throughout the
stored chunks, since important revision statements can occur beyond the cover.
An unedited advance version is treated conservatively as draft.

Explicit documentary date signals take precedence over catalog/title fallbacks;
compatible dates with finer precision are retained. Conflicting alternatives
remain visible in `signals` and produce warnings. This does not resolve every
conflict: competing explicit dates may still need human review. An issuer-matched
copyright year is a **publication-year proxy**, marked with a distinct basis,
medium confidence and a warning. Copyright text is evidence of that copyright
year, not definitive proof of publication on that date.

The parser requires a statement about the document itself. A report saying that
a separate CWD plan was revised in 2022 must not acquire that revision date.
Negated, hypothetical, scheduled and quoted third-party replacement statements
are rejected. The supported patterns are deliberately conservative; unfamiliar
wording may remain unresolved.

## Families and directional relationships

Candidate families require the same nonempty issuing agency and matching
normalized titles. Normalization removes recognized years, version markers and
revision/draft/final cues while preserving the topic and document type. This is
a heuristic grouping signal, not proof that one item replaces another. Different
issuers, or a research report versus management guidance, remain separate unless
explicit documentary evidence connects them.

An explicit relationship can resolve a target by its document ID, unambiguous
title, or a qualifying year/acronym reference associated with the same issuer.
Ambiguous references and targets absent from the index are reported without
inventing an edge. Explicit edges also join their documents into a family.

| Relationship | Effect |
| --- | --- |
| `supersedes`, `replaces` | Identifies an explicit successor; the predecessor remains available for history |
| `revision_of` | Establishes a revision and permits preference for revised material; does not declare every earlier section obsolete |
| `amendment_of` | A partial update; retain the base and amendment together for current requests |
| `supplement_to` | Additional material; retain the base and supplement together |
| `withdraws` | A dated withdrawal relationship applied at query time, rather than permanently rewriting the target's historical status |

No persistent `is_current` boolean is assigned. Applicability depends on the
question, date, family and available relationship evidence. Conflicting cycles
in replacement links produce uncertainty rather than a claimed final successor.

## Retrieval and answer behavior

[temporal.py](temporal.py) detects obvious temporal intent without a model call.
The existing retrieval ranking continues to handle non-temporal questions.
Temporal requests use keyword/topic candidates, family expansion and canonical
metadata lookup, followed by inspectable precedence rules rather than an
arbitrary recency multiplier.

| Request | Behavior |
| --- | --- |
| “What is the current invasive carp guidance?” | Prefer an applicable explicit successor; distinguish guidance/strategy from newer research |
| “What does the latest research report say?” | Use the latest relevant report/date ordering while retaining status and family uncertainty |
| “What did the 2020 guidance recommend?” | Retrieve the dated historical source deliberately |
| “What was applicable as of 2024-06-01?” | Apply currentness rules at the requested cutoff |
| “How did guidance change from 2020 to 2024?” | Retrieve relevant versions in chronological order |
| “Is that still the current recommendation?” | Reuse the preceding subject and cited family, then check current applicability without retaining an old year as a hard constraint |

For current guidance, effective explicit replacement evidence comes first.
Draft, withdrawn and not-yet-applicable sources are excluded from preference
when applicable alternatives exist. The ordering then considers document role,
known final status, explicit revision relationships and topic relevance. Dates
break ties within candidate families; an unrelated newer report does not gain
formal authority merely through age. A latest-report request permits date-based
ordering without asserting that the newest report replaces guidance.

Effective dates determine when a successor or withdrawal can take effect. A
document published before its future effective date does not displace currently
applicable guidance early. A year/month-only effective date does not activate
replacement during that unresolved interval: its end must have passed. The
uncertainty is disclosed and the predecessor remains eligible. An `as of` cutoff
without a day uses the start of the stated year/month; use a full ISO date for an exact cutoff.
Queries without a cutoff use the runtime date. Unknown dates/status remain
unknown, and selection without conclusive currentness evidence is qualified.

Older sources may still appear as historical context. A base document needed to
interpret an amendment or supplement is retained even if doing so exceeds the
initial result limit. Source-date conflicts, uncertain candidate families,
multiple issuers and unavailable applicability evidence are disclosed.

Recognized temporal answers use the local renderer, canonical excerpts and
verified lifecycle statements. They do not need an embedding or answer-generation
API call. The supported temporal follow-up path also resolves context locally;
the existing resolver still handles other ambiguous follow-ups. Context is used
only to identify the request, never as factual evidence.

Before using an edge, V3 rechecks its target and exact span against the canonical
store. Answers show date/status labels, the reason a document is preferred,
supporting lifecycle quotations and relevant source excerpts with citations.
A claim that an older source was replaced cites the successor's actual
replacement evidence. The regular model-answer path reserves authoritative
lifecycle assertions for this verified renderer; an ordinary model output cannot
supply its own supersession or current-guidance conclusion.

## Wiki and source presentation

Corpus listings, search results and chatbot source cards expose compact
date/status information. Wiki supporting evidence displays the same labels.
If a verified revision, replacement, amendment or supplement concerns a source
used by a cached Wiki page, the page shows a version-context notice and expandable
relationship evidence.

Wiki notices attribute a relationship to the source statement and visibly show
the source's date/status, including a known effective date. A draft or future
update's statement is not presented as an already applicable replacement.

Cached Wiki facts and historical evidence are preserved. This change does not
silently rewrite old claims, automatically regenerate Wiki text, or globally
discard superseded evidence. A Wiki page remains an evidence overview; its
version notice directs the reader to check the applicable source before using
recommendations.

## Index lifecycle and migration

Two small SQLite tables store derived state:

- `document_lifecycles`: document ID and JSON payload.
- `document_lifecycle_state`: canonical revision counter, built revision and
  extractor version.

Insert/update/delete triggers on `documents` and `knowledge_artifacts` increment
the revision counter. `get_lifecycles()` reads the saved payloads when the index
matches that counter and extractor version. Otherwise it rebuilds once from
canonical metadata/text. Rebuild uses a savepoint and replaces only derived
lifecycle records; an extraction failure does not discard the evidence index.

The Streamlit app creates the index automatically in its writable runtime copy
of the bundled database. Successful corpus ingestion also refreshes lifecycle
metadata. Existing deployments do not require text re-extraction or new
embeddings solely for this feature.

For an existing local database, run this metadata-only migration from `new-V3`:

```powershell
python main.py --database C:\path\to\corpus.db --backfill-lifecycle
```

Choose the database you intend to update. The command adds/refreshes lifecycle
tables and triggers while preserving document IDs, canonical text, vectors and
compiled Wiki artifacts. It uses the selected database's registry, rather than
overwriting it with the bundled catalog. Updating an extraction rule requires a
new extractor version or an explicit rebuild, not a paid re-ingestion.

## Evaluation and reproducibility

[evaluation/datasets/temporal.json](evaluation/datasets/temporal.json) defines 22
temporal cases: 16 synthetic scenarios and six real-corpus checks. The complete
standard evaluation currently contains 64 cases across all categories. These are
dataset counts, not an estimate of general policy accuracy.

Release validation: all 410 pytest tests passed (404 fast and 6 integration),
including the existing suite. All 64 offline evaluation cases passed; all 22
temporal cases passed their declared checks. Within labelled synthetic cases,
current/historical/latest selection and intent metrics were 1.0, with zero
unsupported supersession claims and invalid lifecycle evidence. The six corpus
cases validate inspected metadata/relations, not universal current-guidance accuracy.
Original retrieval benchmark scores remained unchanged.

Local browser verification covered 2015 Missouri SWAP retrieval, the follow-up
asking whether it remains current, chronological comparison with DOC036, Wiki
relationship evidence, and refresh clearing all chat history. These temporal
workflows called no providers. Full tests took about 20 seconds; the complete
offline evaluation took about 7 seconds including startup on this workstation.
An observed first lifecycle build took about 3.6 seconds, with subsequent source
selection/answers around 0.1–0.25 seconds on the bundled corpus. These are local
observations, not performance guarantees for larger corpora.

DOC951–DOC965 are explicitly invented test-only sources. Each fixture case builds
its own in-memory store with the stated agency, date, status and relationship
evidence. Currentness is evaluated at `2026-09-07`, except the explicit future
activation case at `2027-02-01`. Corpus cases use a disposable SQLite snapshot
and revalidate their inspected exact spans before scoring.

The temporal runner measures current-source preference, historical retrieval,
latest-report selection, temporal intent, relationship/metadata correctness,
required-source recall, source sequence, family isolation and unsupported
replacement assertions where the case provides corresponding labels. It also
checks citation ownership and canonical lifecycle evidence. Cases without a
known current source do not contribute an invented current-source accuracy.
Generic engine uncertainty notices are excluded from factual claim coverage;
arbitrary uncited assertions are still checked.

Run from `new-V3`:

```powershell
# Fast offline tests
python -m pytest -q -m "not integration"

# All tests, including disposable-corpus integration
python -m pytest -q

# Temporal scenario and evaluator tests
python -m pytest tests/test_temporal_evaluation.py -q

# Complete offline evaluation; writes JSON and Markdown reports
python -m evaluation.run

# A single declared temporal case
python -m evaluation.run --case temporal_historical_to_current_followup

# Compare with the reviewed baseline
python -m evaluation.run --baseline evaluation/baselines/offline.json --fail-on-regression
```

Reports are written to `evaluation/reports/latest.json` and `latest.md` by
default. No paid temporal evaluation mode is needed. The existing optional
`--live` evaluation applies to selected retrieval/generation cases and does not
turn temporal fixtures into an external current-policy check. See
[EVALUATION.md](EVALUATION.md) for report interpretation and baseline handling.

## Real-corpus findings and limits

The corpus contains one useful indexed revision pair: DOC036 states on stored
p22 that the 2020 CCS is the comprehensive revision of the 2015 SWAP (DOC001).
This supports `revision_of`, without a blanket assertion that every original
section was superseded. DOC036's catalog title/year says 2022 while its own
version statements say 2020. The stored publication date remains the catalog's
2022 metadata, while revision/version 2020 retains its separate documentary
evidence; the version year is not invented as a publication date.

Other inspected examples demonstrate why raw catalog recency is insufficient:

- DOC006's cover says June 2020; its catalog year is 2023.
- DOC012's publisher imprint says 2023; its catalog year is 2024.
- DOC007/008 are the same accomplishments publication, whose suggested citation
  says 2026; 2021–2025 is the plan/reporting period, not its publication date.
- DOC016/017 have January 2025/2024 bylines while reviewing FY2024/FY2023.
- DOC027 is a sixth-edition Ramsar manual with a 2013 copyright-year proxy,
  despite a catalog value of `Current`.
- DOC032 identifies itself as an unedited advance version, rather than an
  established final document.

Some mismatches remain outside the supported extraction patterns. For example,
DOC003's catalog says 2024 while its text identifies the 1998 NAWMP update;
DOC018's catalog says FY2021 while the stored article discusses FY2022 and has
a January 2023 byline. These are corpus curation issues, not evidence that a
newer policy exists. No catalog corrections or external documents are invented
by lifecycle extraction.

Family/title rules and document-type classification are heuristic. Acronyms,
unusual layouts, missing metadata, changes in issuing authority and OCR errors
can prevent a real relationship from resolving. References to plans that are
not independently indexed cannot form a complete version chain. Conflicting
jurisdictions or applicability to a particular location are disclosed where
possible, but this is not a legal authority engine or a section-by-section
amendment resolver.

Indexing scans the corpus only after relevant changes. Query time still uses
the existing keyword scan, loads small lifecycle payloads, verifies relationship
spans and reads evidence from selected families; it does not introduce a new
model call. Larger corpora may need a more selective retrieval/index strategy.
Deterministic tests establish these documented invariants, not complete semantic
entailment, scientific accuracy or externally guaranteed current guidance.
