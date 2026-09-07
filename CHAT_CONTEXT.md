# Conversation-aware grounded chatbot

## Root cause

The UI already stored user/assistant messages, source objects and preambles in
`st.session_state.v3_chat_messages`, but passed only the latest question into
`ask_chatbot_with_context`. Despite its name, that function's existing “context”
was the answer preamble and cited sources, not conversation memory. It embedded
the original question directly and used that same question for keyword fallback
and final synthesis. A question about “these methods” therefore lacked an entity
at retrieval time, allowing generic effectiveness evidence about plants to rank.

## Architecture

`chat_context.resolve_query` runs before any direct-answer lookup, document
discovery, embedding, or keyword retrieval. It returns an immutable
`ResolvedQuery` containing the original question, standalone question, active
subject, history-use flag, resolution status and method.

- First turns and recognizable independent/topic-switch questions stay unchanged
  and incur no contextualization API call.
- Other turns use one small structured request through the existing API client
  and UI-selected model. It resolves implicit subjects, method lists, ordinal
  options, prior conclusions and report references. Prior conclusions become
  questions to verify, not facts to assume.
- Input is restricted to six recent messages (normally three turns), each capped
  at 1,800 text characters. Prior resolved queries are capped at 2,400 characters,
  subject output at 160, and cited metadata at five titles/IDs per message.
  Source chunks are not carried into the contextualizer as evidence.
- The output subject must originate in user intent, previous resolved state, or
  trusted cited-report titles. Known entity phrases are normalized when the model
  wraps them in a facet such as “effectiveness of invasive carp control methods.”
- Unresolvable references, invalid structured output or provider failure produce
  a clarification response with no retrieval or fabricated answer.

The standalone question actually replaces the query used by the existing
retrievers. For context-dependent turns, keyword results supplement semantic
results, and a subject guard checks canonical source text/titles before assigning
evidence handles. Alias groups cover invasive/Asian carp, zebra mussels, and
invasive aquatic plants; other grounded subjects use exact phrase matching.
Existing source diversity and top-k behavior are retained.

Final synthesis receives the resolved question and newly retrieved canonical
chunks. Raw prior assistant answers are not appended to its evidence payload.
The prompt explicitly treats context-derived methods/conclusions as search intent
and requires missing quantities or comparative evidence to be acknowledged.
Canonical artifact ownership, exact-span validation, citation rendering and
the fail-closed source-list fallback remain unchanged.

## Session state and reset

No global conversation store or database migration was added. Assistant message
records now include a `context` dictionary with the resolved query, active subject,
resolution flags, context version and actual retrieval-query input. The UI passes
a slice of prior messages before appending the current user message, avoiding
duplicate current turns. **New conversation** clears that same message list,
including all context and cached source references. Independent browser sessions
retain independent Streamlit session state.

Explicit statements such as “Now tell me about invasive aquatic plants” are
processed as new subjects without previous carp constraints. Other topic switches
are resolved by the contextualizer. Existing single-turn callers remain compatible
because `history` and `diagnostics` are optional keyword arguments and the returned
answer/preamble/sources tuple is unchanged.

## Diagnostics

Backend callers may opt into diagnostics without logging a transcript:

```python
diagnostics = {}
answer, preamble, sources = ask_chatbot_with_context(
    question, store, history=recent_messages, diagnostics=diagnostics
)
```

Inspect `original_query`, `standalone_query`, `active_subject`, `uses_history`,
`method`, `needs_clarification`, `context_version` and `retrieval_query` in the
returned dictionary. These are also available in the session's assistant record.
They are not displayed as product controls or automatically logged. Callers should
treat this diagnostic state as private conversation content.

## Tests and actual-app verification

`python -m pytest -q`: **52 passed**, including all 27 previous tests (Wiki,
provenance, professor-feedback fixes and other existing checks) and 25 new
multi-turn test cases in `tests/test_chat_context.py`.

The new tests cover the reported two-turn case, implicit subjects, method and
ordinal references, report references, a three-turn comparison chain, topic
switches, fresh sessions, clear-chat behavior, bounded history, rejected invented
subjects, provider failure, unrelated plants ranked ahead of carp, valid numeric
citations, absent evidence and rejection of fabricated prior-answer quotations.
An integration test executes the real Streamlit application through AppTest,
replacing only storage setup/provider calls with fixtures, and runs the complete
three-turn workflow and reset.

With explicit user authorization, the same Streamlit application was also tested
against the existing OpenAI configuration and actual corpus:

1. “What are the effective methods to control invasive carp?” produced a cited
   answer about behavioral deterrents and developing genetic-control methods.
2. “Provide some data on how effective these methods are.” became a standalone
   query naming invasive carp, carbon dioxide, underwater acoustic deterrents,
   oblique bubble curtains and RNA interference. The cited retrieval stayed on
   carp. One synthesis attempt used the existing source-only fallback after
   validation failure; a targeted recheck produced verified claims with exact
   quotes from DOC012, printed page 6 / PDF page 14, and explicitly listed the
   missing effectiveness data for those four methods. No effectiveness numbers
   were invented or taken from aquatic-plant evidence.
3. “Now tell me about invasive aquatic plants.” used that new question directly,
   with `uses_history=false`, and produced cited aquatic-plant information.
4. **New conversation** emptied all session messages/context.

Observed whole-turn durations in one real three-turn run were approximately
6.4 s, 7.2 s and 10.0 s. These include the provider, retrieval, validation and
AppTest execution; they are not browser latency guarantees. The context request
has a 700-token output cap. Independent questions add no API calls; a dependent
turn adds one small contextualization call plus a local keyword scan. There is
no additional persistence/ingestion work and no change to Wiki pre-display.

## Limits and deployment

The resolver is model-based and can request clarification for ambiguous,
out-of-window or unresolved references. Truncation may omit a distant option.
Exact-subject filtering is conservative and may miss synonyms not in the small
alias map or useful excerpts without the subject in their source title/text.
Complex comparisons spanning several unrelated topics are not represented by a
full entity graph. Model-written claims can still fail the existing verbatim
validator, in which case canonical source links replace unsupported synthesis.

No new dependencies or data migration are required. The release comment in
`requirements.txt` changes to v3.4 so Streamlit Community Cloud performs a fresh
deployment and loads the new module. Dependency constraints remain unchanged.
The existing Wiki compiler, precomputed database and ingestion pipeline are not
modified by this task.

For browser verification after deployment, open Chatbot, use **New conversation**,
and submit the three questions above in sequence. Check that the second response
stays on carp and cites relevant evidence or clearly states missing data, then
that the third response switches to plants. Clear the conversation once more.
