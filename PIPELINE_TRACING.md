# CDIRP V3 pipeline tracing and retrieval RCA

## Files and use

- `pipeline_tracer.py`: standard-library-only, opt-in request-local JSON tracer.
- `scripts/debug_q1.py`: corpus date extraction, persisted/fresh comparison, catalog-year removal experiment, source year ranges, temporal route.
- `scripts/debug_q4.py`: actual temporal route, exact title/year resolution, lexical/dense/hybrid/rerank ablations, hard scope and per-document counterfactuals.
- `tests/test_pipeline_tracer.py`: exact request/raw response capture, unchanged answers, request isolation, snapshot copying, and failure behavior.

From `new-V3`, enable tracing before starting Streamlit:

```powershell
$env:V3_TRACE_DIR = Join-Path $PWD 'traces'
streamlit run app.py
```

Or trace one existing backend call:

```python
from pipeline_tracer import PipelineTracer
from main import ask_chatbot_with_context

with PipelineTracer('traces') as trace:
    answer, preamble, sources = ask_chatbot_with_context(
        question, store, model='gpt-5.6-sol', history=history,
    )
print(trace.path)
```

Tracing does not change the configured model, routing, ranking, validation, or return tuple. The trace is exported before the backend returns to Streamlit. Production provider calls retain their existing behavior. A trace contains full source text and prompts; keep its directory private. ContextVar isolates concurrent requests/tasks; manually spawned worker threads need explicit context propagation. The wrapper targets this backend's synchronous entry point.

## Eight insertion points

| Stage | Existing code boundary |
| --- | --- |
| original_query | `main.ask_chatbot_with_context` decorator, before query resolution |
| rewritten_query | Immediately after `resolve_query`; full routing diagnostics in `routing` |
| initial_retrieval | `retrieval.retrieve_evidence`, after candidate fusion and before reranking; chunk IDs/text and raw search events |
| reranked_evidence | Retriever final selection; overwritten by temporal final selection or normal synthesis selection where applicable |
| llm_context_payload | `api_clients.call_structured_llm`, exact request dictionary immediately before SDK call, including system/user messages and JSON schema |
| draft_answer | Raw completion message content before stripping, envelope edits, or validation |
| validation_results | `validator.validate_render_and_collect_sources`, accepted/rejected claims, unsupported facets and parse errors |
| final_output | Entry-point decorator captures exact answer, preamble, and returned sources |

Raw cosine scores are captured inside `ExactVectorStore.search` from the same score array used to rank. SQLite BM25 scores are selected by `lexical_candidates` in the same query used to choose hits. Cosine is higher-is-better; SQLite FTS5 BM25 is lower-is-better. Document-recovery hits carry their own scoped search events; fallback first-page hits have no fabricated BM25 score. Fusion uses equal reciprocal rank votes, not a weighted sum of these raw scores.

Additional events expose effective hard document scope, alias groups, reranker features, fused rankings, duplicate suppression, temporal eligibility and exact ordering keys, full lifecycle selection, compiled evidence handles, raw HTTP response JSON, lifecycle claim removals, and individual provenance rejection reasons.

`stages` contains the latest snapshot per junction; `events` preserves every invocation in order, including query-resolution LLM calls or repeated searches. Unvisited stages are `not_reached`, with a reason. Local temporal answers, clarification, no-evidence answers, and early returns must not be represented as fabricated LLM calls. Legacy/mock stores without the indexed retriever do not expose complete raw-score instrumentation. Capture/export failures are logged and do not replace the answer or original exception.

## Offline diagnostic commands

```powershell
python scripts/debug_q1.py
python scripts/debug_q4.py
# Optional: use a cached embedding for this exact Q4 query and the index's embedding model.
python scripts/debug_q4.py --embedding-json q4_embedding.json
```

Both scripts accept `--database`, `--output`, `--query`, and `--top-k`. They open the original database read-only and use SQLite backup into a temporary copy, so index refreshes do not alter the production corpus. They make no provider calls. Their traces are diagnostic experiments, not full synthesis traces; they inspect the raw query's temporal route directly. Use the backend wrapper above to capture query rewriting and full generation. Q4's target-title audit is intentionally fixed to the two requested documents even with a query override.

Without a cached embedding, dense is explicitly not run and hybrid operates with lexical votes only. No conclusion about live semantic contribution can be drawn from that run. Exact title/year resolution must be unique before the hard-scope counterfactual runs; unresolved or ambiguous targets are reported.

## Historical RCA findings before corrections

Q1: DOC002 contains the explicit own-plan statement: "This wetland program plan includes the years 2023-2028" (artifact 2175, page 1). Fresh lifecycle extraction has no publication, revision, or effective date for DOC002, and no planning-period field. This is a missing lifecycle representation, not evidence of a catalog-date weighting coefficient. Source date extraction prefers explicit dates over catalog inference. Current-mode ranking uses lifecycle/status/relevance policy, with recency considered within families.

The offline Q1 run selected DOC002 first, followed by DOC036, DOC001, DOC025, DOC017. The reported older-document outranking was not reproduced in this checkout. Compare a trace from the failing deployed instance, including rewritten query, corpus/index state, and temporal ordering keys. A planning horizon end year must not be treated as publication date or automatic proof of authority.

Q4: both exact names resolve: DOC036 (2022 strategy) and DOC001 (2015 plan). The raw query triggers temporal comparison because it contains two years and "from", bypassing the ordinary synthesis path. That route selected DOC001, DOC036, DOC005, DOC006, DOC002 in the offline run.

Normal retrieval does not turn these two unquoted names into a hard document filter. There is a functioning document-ID scope mechanism; titles are otherwise soft reranker features. The unscoped lexical ablation selected DOC036, DOC036, DOC006, DOC008, DOC007. The lexical-only reranked ablation selected DOC036 for all five chunks. Restricting to DOC036 and DOC001 still selected DOC036 for all five chunks. Hard filtering alone therefore does not guarantee two-sided comparison coverage; inspect the per-document counterfactual for DOC001 evidence. The reported Chesapeake output was not reproduced in these top-five final selections; semantic contribution remains untested without the matching embedding.

These changes instrument the failures; they deliberately do not change lifecycle extraction or ranking policy.

## Implemented corrections

See `RCA_FIXES.md` for exact code excerpts, integration details, and the corrected corpus results. The historical findings above describe the pre-fix baseline.
