"""Clarifications retain the real antecedent without inventing control methods."""

import json

import pytest

import chat_context as context
import conversation_history as threads
import main
from data_models import KnowledgeArtifact
from database import KnowledgeStore
from test_chat_context import resolved


THREAT_QUESTION = "What are the main conservation threats mentioned across the documents?"
THREAT_ANSWER = (
    "- The main conservation threats include direct drivers such as land and sea use change, "
    "direct exploitation of organisms, climate change, pollution, and invasive alien species. "
    "[DOC032 — DocumentCloud Environment Project Search, PDF p. 12]\n\n"
    "**Unsupported facets**\n\n"
    "- One or more generated claims failed provenance validation."
)
METHOD_FOLLOWUP = "provide some data on how effective these methods are"
CLARIFICATION = (
    "The previous answer listed conservation threats rather than control methods. "
    "Do you mean data on the impacts of those threats, or the effectiveness of measures addressing them?"
)


def threat_history():
    return [
        {"role": "user", "content": THREAT_QUESTION},
        {"role": "assistant", "content": THREAT_ANSWER, "context": {
            "standalone_query": THREAT_QUESTION,
            "active_subject": "",
            "relation": "NEW_TOPIC",
            "needs_clarification": False,
        }},
    ]


def model_owned_threat_history():
    """An explanatory sentence makes intent classification require the model."""
    turns = threat_history()
    turns[-1]["content"] = THREAT_ANSWER.replace(
        "**Unsupported facets**",
        "The documents also discuss the effects of these drivers on biodiversity.\n\n**Unsupported facets**",
    )
    return turns


def ambiguous_payload():
    return json.dumps({
        "standalone_query": "",
        "active_subject": "",
        "uses_history": True,
        "needs_clarification": True,
        "relation": "FOLLOW_UP",
        "selected_context": "The previous answer listed conservation threats, not management methods.",
        "clarification_kind": "THREATS_NOT_METHODS",
    })


def pending_history(*, legacy=False):
    turns = threat_history()
    diagnostic = {
        "original_query": METHOD_FOLLOWUP,
        "standalone_query": "",
        "active_subject": "",
        "uses_history": not legacy,
        "needs_clarification": True,
        "method": "ambiguous",
        "relation": "NEW_TOPIC" if legacy else "FOLLOW_UP",
    }
    if not legacy:
        diagnostic["clarification_question"] = CLARIFICATION
    turns.extend([
        {"role": "user", "content": METHOD_FOLLOWUP},
        {"role": "assistant", "content": CLARIFICATION if not legacy else
         "Please name the subject, method, option, or report you mean so I can search the right evidence.",
         "context": diagnostic},
    ])
    return turns


def test_reported_threats_to_methods_mismatch_gets_contextual_clarification(monkeypatch):
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: pytest.fail("Exact threat-only list called contextualizer"))
    monkeypatch.setattr(main, "generate_embedding", lambda *a: pytest.fail("Ambiguous methods were embedded"))
    monkeypatch.setattr(main, "call_llm", lambda *a, **kw: pytest.fail("Ambiguous methods reached synthesis"))
    with KnowledgeStore(":memory:") as store:
        monkeypatch.setattr(store, "retrieve", lambda *a, **kw: pytest.fail("Ambiguous methods reached retrieval"))
        monkeypatch.setattr(store, "retrieve_document_matches", lambda *a, **kw: pytest.fail("Ambiguous methods reached document search"))
        diagnostics = {}
        answer, _, sources = main.ask_chatbot_with_context(
            METHOD_FOLLOWUP, store, history=threat_history(), diagnostics=diagnostics,
        )
    assert CLARIFICATION in answer
    assert "Please name the subject" not in answer
    assert sources == []
    assert diagnostics["needs_clarification"] is True
    assert diagnostics["uses_history"] is True
    assert diagnostics["relation"] == "FOLLOW_UP"
    assert diagnostics["retrieval_query"] == ""
    assert diagnostics["clarification_question"] == CLARIFICATION
    assert diagnostics["method"] == "referent_type_mismatch"
    assert diagnostics["history_messages_used"] == 2


def test_less_obvious_threat_answer_uses_model_classification(monkeypatch):
    turns = model_owned_threat_history()
    calls = []

    def clarify(system, prompt, schema, **kwargs):
        data = json.loads(prompt)
        calls.append(data)
        assert data["current_question"] == METHOD_FOLLOWUP
        assert data["recent_messages"][0]["content"] == THREAT_QUESTION
        assert data["recent_messages"][1]["content"] == turns[-1]["content"]
        assert "clarification_kind" in schema["schema"]["required"]
        assert "clarification_question" not in schema["schema"]["properties"]
        return ambiguous_payload()

    monkeypatch.setattr(context, "call_structured_llm", clarify)
    actual = context.resolve_query(METHOD_FOLLOWUP, turns)
    assert len(calls) == 1
    assert actual.method == "ambiguous"
    assert actual.clarification_question == CLARIFICATION


@pytest.mark.parametrize("answer", [
    THREAT_ANSWER.replace(
        "**Unsupported facets**",
        "Measures addressing these threats include riparian restoration and invasive-species removal.\n\n**Unsupported facets**",
    ),
    "The main conservation threats include climate change and pollution; methods include riparian restoration.",
    "The main conservation threats are climate change and pollution (managed through riparian restoration).",
    "The main conservation threats include climate change, pollution, and invasive alien species.\n"
    "- Riparian restoration is one management approach.",
])
def test_threat_answer_with_actual_methods_bypasses_local_mismatch_guard(monkeypatch, answer):
    turns = threat_history()
    turns[-1]["content"] = answer
    calls = []
    query = "Provide quantitative effectiveness data on riparian restoration and invasive-species removal addressing conservation threats."

    def rewrite(*args, **kwargs):
        calls.append(True)
        return resolved(query, "conservation threats")

    monkeypatch.setattr(context, "call_structured_llm", rewrite)
    actual = context.resolve_query(METHOD_FOLLOWUP, turns)
    assert calls == [True]
    assert not actual.needs_clarification
    assert actual.standalone_query == query
    assert actual.method == "contextualized"


def test_unknown_list_item_is_not_assumed_to_be_a_threat(monkeypatch):
    turns = threat_history()
    turns[-1]["content"] = (
        "The main conservation threats include climate change, pollution, "
        "and an unidentified management practice."
    )
    calls = []

    def clarify(*args, **kwargs):
        calls.append(True)
        return ambiguous_payload()

    monkeypatch.setattr(context, "call_structured_llm", clarify)
    actual = context.resolve_query(METHOD_FOLLOWUP, turns)
    assert calls == [True]
    assert actual.method == "ambiguous"


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("reply,query", [
    (
        "I mean the effectiveness of measures to address those threats.",
        "Provide quantitative data on the effectiveness of measures addressing the conservation threats "
        "previously listed: land and sea use change, direct exploitation, climate change, pollution, "
        "and invasive alien species.",
    ),
    (
        "I mean their impacts.",
        "Provide quantitative data on the impacts of the conservation threats previously listed: "
        "land and sea use change, direct exploitation, climate change, pollution, and invasive alien species.",
    ),
])
def test_reply_to_clarification_keeps_original_threats_and_request(monkeypatch, legacy, reply, query):
    turns = pending_history(legacy=legacy)
    before = json.dumps(turns)

    def rewrite(system, prompt, schema, **kwargs):
        data = json.loads(prompt)
        recent = data["recent_messages"]
        assert len(recent) == 4
        assert recent[0]["content"] == THREAT_QUESTION
        assert recent[1]["content"] == THREAT_ANSWER
        assert recent[2]["content"] == METHOD_FOLLOWUP
        assert recent[-1]["resolved_context"]["needs_clarification"] is True
        if not legacy:
            assert recent[-1]["resolved_context"]["clarification_question"] == CLARIFICATION
        return resolved(query, "conservation threats")

    monkeypatch.setattr(context, "call_structured_llm", rewrite)
    actual = context.resolve_query(reply, turns)
    assert not actual.needs_clarification
    assert actual.uses_history
    assert actual.standalone_query == query
    assert actual.active_subject == "conservation threats"
    assert json.dumps(turns) == before


def test_explicit_new_topic_discards_pending_clarification(monkeypatch):
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: pytest.fail("Independent topic called contextualizer"))
    question = "Tell me about zebra mussels."
    actual = context.resolve_query(question, pending_history())
    assert actual.standalone_query == question
    assert actual.active_subject == "zebra mussels"
    assert actual.relation == "NEW_TOPIC"
    assert not actual.uses_history and not actual.needs_clarification
    assert actual.clarification_question == ""


def test_broad_threat_followup_keeps_grounded_intervention_evidence(monkeypatch):
    evidence = "Restoration of riparian forest reduced sediment loads by 35 percent in the study watershed."
    artifact = KnowledgeArtifact("DOC090", "Land-use restoration outcomes", "7", evidence)
    assert "conservation threats" not in artifact.title + artifact.original_text_chunk
    query = "Provide quantitative data on the effectiveness of measures addressing conservation threats such as land use change."
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: resolved(query, "conservation threats"))
    embedded = []
    def embed(question):
        embedded.append(question)
        return [1., 0.]
    monkeypatch.setattr(main, "generate_embedding", embed)

    def answer(system, prompt, artifacts, **kwargs):
        assert len(artifacts) == 1 and artifacts["K1"].document_id == "DOC090"
        assert THREAT_ANSWER not in prompt
        return json.dumps({
            "preamble": "", "status": "answered",
            "claims": [{"text": evidence, "evidence_ids": ["K1"], "supporting_spans": [evidence]}],
            "unsupported_facets": [],
        })
    monkeypatch.setattr(main, "call_llm", answer)
    with KnowledgeStore(":memory:") as store:
        store.ingest_chunk(artifact, [1., 0.])
        diagnostics = {}
        result, _, sources = main.ask_chatbot_with_context(
            "I mean the effectiveness of measures to address those threats.", store,
            history=pending_history(), diagnostics=diagnostics,
        )
    assert embedded == [query]
    assert "35 percent" in result and "DOC090" in result
    assert [source.document_id for source in sources] == ["DOC090"]
    assert diagnostics["active_subject"] == "conservation threats"
    assert diagnostics["relation"] == "FOLLOW_UP"


def test_pending_clarification_survives_archive_without_leaking_to_new_thread(monkeypatch):
    book = threads.empty_book()
    original_id = book["active_id"]
    for message in pending_history():
        threads.append_message(book, original_id, message)
    fresh_id = threads.new_conversation(book)
    with KnowledgeStore(":memory:") as store:
        restored = threads.restore_book(json.loads(json.dumps(threads.export_book(book))), store)
    original = restored["conversations"][original_id]["messages"]
    assert original[-1]["context"]["clarification_question"] == CLARIFICATION
    assert original[-1]["context"]["needs_clarification"] is True
    assert restored["active_id"] == fresh_id
    assert restored["conversations"][fresh_id]["messages"] == []

    calls = []
    def rewrite(system, prompt, schema, **kwargs):
        calls.append(json.loads(prompt))
        return resolved("Provide quantitative data on the impacts of conservation threats.", "conservation threats")
    monkeypatch.setattr(context, "call_structured_llm", rewrite)
    resumed = context.resolve_query("I mean their impacts.", original)
    fresh = context.resolve_query("I mean their impacts.", restored["conversations"][fresh_id]["messages"])
    assert resumed.uses_history and not resumed.needs_clarification
    assert calls[0]["recent_messages"][0]["content"] == THREAT_QUESTION
    assert len(calls) == 1
    assert not fresh.uses_history and fresh.active_subject == ""


@pytest.mark.parametrize("bad_kind", [17, "INVENTED_CLARIFICATION"])
def test_invalid_clarification_kind_does_not_reach_search(monkeypatch, bad_kind):
    payload = json.loads(ambiguous_payload())
    payload["clarification_kind"] = bad_kind
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: json.dumps(payload))
    actual = context.resolve_query(METHOD_FOLLOWUP, model_owned_threat_history())
    assert actual.needs_clarification
    assert actual.method == "invalid_context"
    assert actual.standalone_query == ""


def test_model_cannot_add_a_free_text_answer_through_clarification(monkeypatch):
    fabricated = "The intervention reduced pollution by 90% [DOC999 - Invented Study, PDF p. 1]. Do you mean that method?"
    payload = json.loads(ambiguous_payload())
    payload["clarification_question"] = fabricated
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: json.dumps(payload))
    with KnowledgeStore(":memory:") as store:
        diagnostics = {}
        answer, preamble, sources = main.ask_chatbot_with_context(
            METHOD_FOLLOWUP, store, history=model_owned_threat_history(), diagnostics=diagnostics,
        )
    assert diagnostics["method"] == "invalid_context"
    assert "90%" not in answer + preamble and "DOC999" not in answer + preamble
    assert sources == []


def test_selected_context_cannot_be_rendered_as_an_unvalidated_answer(monkeypatch):
    fabricated = "The intervention reduced pollution by 90% [DOC999 - Invented Study, PDF p. 1]."
    payload = json.loads(ambiguous_payload())
    payload["selected_context"] = fabricated
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: json.dumps(payload))
    monkeypatch.setattr(main, "generate_embedding", lambda *a: pytest.fail("Clarification reached embedding"))
    monkeypatch.setattr(main, "call_llm", lambda *a, **kw: pytest.fail("Clarification reached synthesis"))
    with KnowledgeStore(":memory:") as store:
        answer, preamble, sources = main.ask_chatbot_with_context(
            METHOD_FOLLOWUP, store, history=model_owned_threat_history(),
        )
    assert answer == CLARIFICATION
    assert "90%" not in answer + preamble and "DOC999" not in answer + preamble
    assert sources == []
