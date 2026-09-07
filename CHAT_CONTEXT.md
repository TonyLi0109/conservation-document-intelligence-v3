# Contextual answers and refresh reset (v3.6)

## Direct follow-up answers

The chatbot resolves recent intent before embedding, search and synthesis.
It distinguishes FOLLOW_UP, PARTIAL_CONTEXT and NEW_TOPIC. Recognizable explicit
new topics bypass old context, and partial-context comparisons can broaden scope.

For a conservation-threat discussion followed by "provide some data on how effective
these methods are", the target is now **measures addressing those conservation
threats**. The answer proceeds directly through retrieval and citation validation;
it no longer asks the user to choose between threat impacts and intervention
outcomes. The inference names threats already present in the discussion, but does
not invent a method list or quantitative facts. Requested facets such as cost and
comparison are retained. Missing evidence remains an explicit limitation.

A simple threat-only list can resolve locally with no contextualizer call. Mixed
answers use the existing model resolver, preserving actual methods when identified.
If structured output fails or asks for clarification despite the recoverable threat
topic, a local query inference still enters the grounded answer pipeline. Truly
unresolved references outside this recoverable scope can request clarification.

History is intent, never evidence. Only newly retrieved canonical chunks support
factual claims. Existing exact-span validation, source ownership and no-evidence
behavior remain intact. Broad topics prioritize relevant sources without requiring
every excerpt to repeat a phrase such as "conservation threats" verbatim.

## Page-scoped history

The latest user requirement replaces earlier browser-persistence behavior.
Messages, citations, UUID threads, titles and timestamps live only in Streamlit
session state for the open page. New conversation and the Conversation selector
work within that page. **Refreshing creates an empty conversation and discards all
previous page threads and context.** Closing/reopening also starts fresh.

The small component returns a document identity held on the parent window.
It survives Streamlit internal reruns/component remounts, but not browser refresh.
A changed identity resets the session's book, active thread and message alias.
The component never reads or writes chat contents. It removes only the old
`v3-conversation-history-v1` localStorage key; unrelated storage is untouched.
Legacy archive helpers remain available internally, but the running UI does not
restore an archive or migrate old messages into a refreshed session.

## Verification and diagnostics

`python -m pytest -q --tb=short`: **106 passed**.

Tests cover direct contextual retrieval producing canonical numeric intervention
outcomes, reporting missing coverage, empty-corpus behavior, unsafe model output,
malformed context recovery, preserved cost/comparison intent, explicit topic
switches, separate threads and fresh-page reset. AppTest verifies that internal
reruns retain messages while a changed page identity discards all old threads.

Chrome component checks confirm that remount preserves identity, full reload
changes it, the legacy history key is removed and unrelated storage is retained.
These checks and fixture-backed answer tests do not call OpenAI. Live provider
synthesis for the conservation-threat dialogue has not been tested in this release.

Optional backend diagnostics include original_query, standalone_query,
retrieval_query, active_subject, relation, uses_history, selected_context,
history_messages_used, resolution_error and context_version. Invalid-output error
codes contain no raw provider content or private transcript logging.

## Costs and scope

At most six recent messages are sent to the contextualizer, with the existing
per-message caps and 700-token output cap. Locally inferred follow-ups skip that
call but still use the normal embedding/synthesis requests to answer from evidence.
Refresh reset, history selection and titles add no model calls.

Dependency constraints, Wiki pre-generation and corpus data are unchanged.
The requirements release marker is v3.6 for a fresh Streamlit deployment.
