# Selective conversation context and saved threads

## Context routing

`chat_context.resolve_query` runs before retrieval and classifies turns as
`FOLLOW_UP`, `PARTIAL_CONTEXT`, or `NEW_TOPIC`. Recognizable independent questions
(including zebra mussels, hydrilla, Asian longhorned beetle and climate/wetland
questions) bypass old context without requiring a reset command. A new-topic
model decision retains the original question rather than a stale rewritten query.
Subsequent follow-ups use the latest topic segment within the last six messages.
Explicitly returning to carp starts that topic again.

Dependent questions use one bounded structured request to resolve relevant
subjects, methods, comparisons or references. Partial-context questions retain
selected methods while allowing a broader target such as other invasive fish.
Recognized entity guards apply only to same-topic FOLLOW_UP retrieval, never to
PARTIAL_CONTEXT or NEW_TOPIC. Broad subjects such as "conservation threats"
prioritize matching sources without rejecting other relevant evidence simply
because it lacks that exact phrase. Selected context and relation are saved alongside
original/standalone queries, active subject, flags and actual retrieval query.

History is intent, not evidence. Final answers still use newly retrieved canonical
chunks and the existing exact-span/citation checks. Invalid or ambiguous context
requests clarification. Unsupported generated claims still use the existing
validated-source fallback. Wiki generation and the corpus remain unchanged.

## Conversation lifecycle and storage

`conversation_history.py` stores UUID threads with first-question titles, UTC
creation/update timestamps, timestamped messages and per-thread context.
**New conversation** creates and selects a new thread without deleting old ones.
The **Conversation** selector reopens a thread and passes only its messages to
the backend. No model call generates titles.

The small bundled component in `components/conversation_archive/index.html`
persists this browser's archive in localStorage, isolated by app origin and browser
profile. Refreshes and server restarts restore it. There is no global server-side
transcript store, new database schema, account login or cross-device sync.
People sharing a browser profile share its saved history; clearing site data or
private browsing can remove it. Browser storage quotas can prevent saving.

Source references contain document ID, page and a SHA-256 chunk digest, not copied
corpus chunks. `database.resolve_source_reference` restores source objects from
the canonical corpus rather than trusting browser-supplied source metadata.
Missing sources are marked unavailable while messages are retained. Invalid
archives/storage failures preserve the old archive and show a warning. Revision
checks prevent a stale tab overwriting another tab's saved changes; conflicting
new messages remain in that tab until the user preserves them. Existing session
messages migrate when no browser archive exists.

## Verification

`python -m pytest -q`: **99 passed**, including existing Wiki/provenance tests.
Coverage includes explicit topic switches, post-switch follow-ups, rejected stale
rewrites, partial-context retrieval of other-fish evidence, malformed context,
thread isolation, canonical source restoration, legacy migration and Streamlit
new-thread/title-change/switch behavior.

Real local-browser checks using the approved existing OpenAI configuration:
- Carp and zebra-mussel questions produced topic-appropriate cited answers in
  separate threads; some claims used the existing provenance fallback.
- Reopening the saved carp thread and asking "Which of those methods has the
  strongest quantitative evidence?" resolved as FOLLOW_UP to invasive carp and
  its previously discussed deterrents/genetic methods, without zebra context.
- Switching threads, creating a third empty thread, refreshing/restoring all
  threads and opening an isolated browser context passed.

Hydrilla and other explicit-topic cases have offline coverage. The additional
hydrilla API test was not run after automatic approval review rejected that topic.

## Costs and limits

Independent recognized questions add no contextualizer call. Dependent turns add
one request capped at 700 output tokens, with at most six recent messages capped
at 1,800 text characters each, bounded resolved-query metadata and five source
titles/IDs per message. Persistence and title creation incur no model calls.

Ambiguous wording, distant references and unfamiliar aliases can still require
clarification or miss evidence. Same-topic subject matching remains conservative.
The archive is browser storage, not a backup or authenticated transcript record.

No dependency constraints or corpus data changed. The requirements release marker
is v3.5.2 to trigger a fresh Streamlit deployment. Backend callers retain the same
answer/preamble/sources tuple and optional `history`/`diagnostics` arguments.


## v3.5.1: contextual clarification recovery

The reported threat-list -> "how effective these methods are" exchange has a
referent mismatch: the answer listed threats, not interventions. The contextualizer
now selects a clarification category; the application renders fixed wording asking
whether the user means threat impacts or the effectiveness of measures addressing
those threats. Model-written clarification prose is never displayed, preventing a
second path for uncited factual claims.
It does not invent a method list or search until the user resolves the ambiguity.
Provider failures instead request a retry; invalid structured output remains
conservative. No provider exception text is shown or logged by the resolver.

Pending clarification turns retain FOLLOW_UP/PARTIAL_CONTEXT rather than resetting
the topic. History trimming also recognizes the clarification flag on old saved
NEW_TOPIC messages, so a clarification reply can recover the original question,
threat list and request for data. Explicit new topics still bypass this context.
Clarification metadata survives the existing browser archive without migration.

Twelve additional offline regression cases cover the exact reported exchange, both
clarification choices, legacy saved history, independent topics, archive isolation,
malformed clarification categories, injected uncited claims and canonical intervention evidence whose wording
does not literally contain "conservation threats". Existing numerical citation and
unrelated-species tests continue to pass. The earlier unsupported-facets warning
is a separate provenance result and remains unchanged.

The existing contextualization call now includes a clarification category; no extra
API request or model-generated title is added. The 700-token output cap is unchanged.


## v3.5.2: repeated generic fallback

The exact one-line "Please name the subject..." message was still reachable when
a model returned valid JSON with a semantically invalid subject or inconsistent
clarification flags. For example, "measures addressing conservation threats" is
not a verbatim subject from the user's question. Earlier tests supplied an ideal
clarification classification and missed this rejected-output branch.

All invalid-output paths now preserve a topic-aware clarification derived from
recent user questions, rather than returning an empty clarification. Model output
is never displayed through this fallback. Diagnostics include a fixed
`resolution_error` code and `history_messages_used`, without logging raw model
responses or private conversations.

A narrow local rule also recognizes the reported answer when it consists solely
of the listed threat categories. Asking about "these methods" then produces the
specific threat-versus-intervention clarification without any API call. Answers
containing actual methods, extra explanation or unfamiliar wording continue
through normal contextualization. This rule does not manufacture a method list.

Saved historical replies remain unchanged; send the follow-up again to obtain a
new response. Old pending clarification metadata remains supported. The archive
component receives an internal runtime version for deployment verification; no
new product control or provider request is added.

Verification: `python -m pytest -q` passes 99 tests. The exact reported dialogue
is tested with contextualization, embedding, search and synthesis forbidden;
additional invalid-output fixtures exercise the model-owned fallback.


Production verification for v3.5.2: an isolated browser first observed the actual
runtime_version from the archive component, then restored the exact reported
threat question/answer and submitted the same methods follow-up. The displayed
reply was the specific threats-versus-measures clarification. Saved diagnostics
confirmed active_subject="conservation threats", uses_history=true,
history_messages_used=2, method="referent_type_mismatch", and an empty retrieval
query. Reload restored the reply. This branch makes no OpenAI calls; arbitrary
model-response behavior is covered by offline invalid-output fixtures.
