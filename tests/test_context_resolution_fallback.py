"""Invalid context output must preserve the user's topic without inventing evidence."""

import json
import re

import pytest

import chat_context as context
import main
from database import KnowledgeStore
from test_chat_context import resolved
from test_contextual_clarification import (
    METHOD_FOLLOWUP,
    THREAT_QUESTION,
    ambiguous_payload,
    assert_inferred_threat_query,
    model_owned_threat_history,
    threat_history,
)


TARGETED_FALLBACK = (
    "We were discussing conservation threats. Do you mean quantitative data on measures "
    "addressing those threats, or data on their impacts?"
)
INJECTED_CLAIM = "Invented treatment reduced pollution by 99% [DOC999 - Fake Report, PDF p. 1]."


def failed_payloads():
    payload = json.loads(resolved(
        "Provide quantitative effectiveness data for measures addressing conservation threats.",
        "conservation threats",
    ))
    return [
        pytest.param("not JSON " + INJECTED_CLAIM, id="malformed-json"),
        pytest.param(json.dumps({**payload, "active_subject": "conservation methods"}),
                     id="ungrounded-subject"),
        pytest.param(json.dumps({**payload, "active_subject": "land and sea use change"}),
                     id="assistant-only-subject"),
        pytest.param(json.dumps({**payload, "active_subject": ""}), id="missing-subject"),
        pytest.param(json.dumps({**payload, "clarification_kind": "THREATS_NOT_METHODS"}),
                     id="inconsistent-clarification-kind"),
        pytest.param(json.dumps({**json.loads(ambiguous_payload()), "clarification_kind": "NONE"}),
                     id="missing-clarification-kind"),
        pytest.param(json.dumps({**payload, "unexpected_answer": INJECTED_CLAIM}),
                     id="unexpected-schema-field"),
        pytest.param(json.dumps({
            **payload,
            "active_subject": "invented treatment",
            "standalone_query": INJECTED_CLAIM,
            "selected_context": INJECTED_CLAIM,
        }), id="model-prose-never-rendered"),
    ]


@pytest.mark.parametrize("raw", failed_payloads())
def test_invalid_output_searches_known_threat_target_without_using_model_prose(monkeypatch, raw):
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: raw)
    queries = []
    monkeypatch.setattr(main, "generate_embedding", lambda query: queries.append(query) or [1., 0.])
    monkeypatch.setattr(main, "call_llm", lambda *a, **kw: pytest.fail("Empty corpus reached synthesis"))
    with KnowledgeStore(":memory:") as store:
        diagnostics = {}
        answer, preamble, sources = main.ask_chatbot_with_context(
            METHOD_FOLLOWUP, store, history=model_owned_threat_history(), diagnostics=diagnostics,
        )
    assert "No relevant evidence" in answer + preamble
    assert "Do you mean" not in answer
    assert "Please name the subject" not in answer
    assert "99%" not in answer + preamble and "DOC999" not in answer + preamble
    assert sources == []
    assert diagnostics["uses_history"] is True
    assert diagnostics["needs_clarification"] is False
    assert diagnostics["relation"] == "FOLLOW_UP"
    assert diagnostics["method"] == "context_inferred"
    assert queries == [diagnostics["retrieval_query"]]
    assert "measures addressing conservation threats" in queries[0]
    assert "effective" in queries[0]
    assert "99%" not in queries[0] and "DOC999" not in queries[0]
    assert "invented treatment" not in queries[0].lower()
    assert diagnostics["clarification_question"] == ""
    assert diagnostics["history_messages_used"] == 2
    assert not diagnostics["resolution_error"] or re.fullmatch(r"[a-z_]+", diagnostics["resolution_error"])
    assert len(diagnostics["resolution_error"]) <= 80


def test_model_threat_method_ambiguity_uses_direct_inferred_search(monkeypatch):
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: ambiguous_payload())
    actual = context.resolve_query(METHOD_FOLLOWUP, model_owned_threat_history())
    assert_inferred_threat_query(actual)
    assert actual.diagnostics()["resolution_error"] == ""


def test_malformed_rewrite_still_searches_for_evidence_without_claiming_methods_were_absent(monkeypatch):
    turns = threat_history()
    turns[-1]["content"] = (
        "The reports discuss conservation threats and methods to address them: "
        "riparian restoration and invasive-species removal."
    )
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: "not JSON")
    actual = context.resolve_query(METHOD_FOLLOWUP, turns)
    assert_inferred_threat_query(actual)
    assert "rather than control methods" not in actual.standalone_query
    assert "did not identify" not in actual.standalone_query


def test_inferred_query_retains_topic_for_subsequent_followup(monkeypatch):
    turns = model_owned_threat_history()
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: "not JSON")
    inferred = context.resolve_query(METHOD_FOLLOWUP, turns)
    turns.extend([
        {"role": "user", "content": METHOD_FOLLOWUP},
        {"role": "assistant", "content": "The retrieved report provides sediment reduction data for riparian restoration.",
         "context": inferred.diagnostics()},
    ])
    query = "Provide quantitative data on the effectiveness of measures addressing conservation threats."

    def rewrite(system, prompt, schema, **kwargs):
        recent = json.loads(prompt)["recent_messages"]
        assert len(recent) == 4
        assert recent[0]["content"] == THREAT_QUESTION
        assert recent[2]["content"] == METHOD_FOLLOWUP
        assert recent[-1]["resolved_context"]["needs_clarification"] is False
        assert recent[-1]["resolved_context"]["clarification_question"] == ""
        assert recent[-1]["resolved_context"]["active_subject"] == "conservation threats"
        return resolved(query, "conservation threats")

    monkeypatch.setattr(context, "call_structured_llm", rewrite)
    actual = context.resolve_query("I mean their effectiveness.", turns)
    assert actual.uses_history and not actual.needs_clarification
    assert actual.standalone_query == query
    assert actual.active_subject == "conservation threats"
    assert actual.diagnostics()["history_messages_used"] == 4


def test_provider_failure_still_uses_known_threat_search(monkeypatch):
    def failed_provider(*args, **kwargs):
        raise RuntimeError("provider failed " + INJECTED_CLAIM)

    monkeypatch.setattr(context, "call_structured_llm", failed_provider)
    actual = context.resolve_query(METHOD_FOLLOWUP, model_owned_threat_history())
    query = assert_inferred_threat_query(actual)
    assert "99%" not in query and "DOC999" not in query
    assert actual.resolution_error == "provider_failure"


@pytest.mark.parametrize("question,facet", [
    ("What is the cost of those methods?", "cost"),
    ("Compare those approaches and their effectiveness.", "Compare"),
])
def test_inferred_target_preserves_the_requested_facet(monkeypatch, question, facet):
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: pytest.fail("Exact threat-only list called contextualizer"))
    actual = context.resolve_query(question, threat_history())
    assert actual.method == "context_inferred"
    assert not actual.needs_clarification
    assert facet in actual.standalone_query
    assert "measures addressing conservation threats" in actual.standalone_query


def test_explicit_new_topic_bypasses_old_fallback(monkeypatch):
    turns = threat_history()
    turns.extend([
        {"role": "user", "content": METHOD_FOLLOWUP},
        {"role": "assistant", "content": TARGETED_FALLBACK, "context": {
            "needs_clarification": True, "relation": "FOLLOW_UP",
            "clarification_question": TARGETED_FALLBACK,
        }},
    ])
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: pytest.fail("New topic called contextualizer"))
    question = "Tell me about zebra mussels."
    actual = context.resolve_query(question, turns)
    assert actual.standalone_query == question
    assert actual.relation == "NEW_TOPIC"
    assert not actual.uses_history and not actual.needs_clarification
    assert actual.clarification_question == ""
    assert actual.diagnostics()["history_messages_used"] == 0


def test_generic_fallback_uses_escaped_prior_user_topic_not_model_prose(monkeypatch):
    turns = [
        {"role": "user", "content": "How can **floodplain** connectivity be restored?"},
        {"role": "assistant", "content": "Several approaches are discussed in the reports."},
    ]
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: "not JSON " + INJECTED_CLAIM)
    actual = context.resolve_query("Which of those approaches is effective?", turns)
    question = actual.clarification_question
    assert "floodplain" in question and "connectivity" in question
    assert "**floodplain**" not in question
    assert "previous" in question.lower() or "earlier" in question.lower()
    assert "conservation threats" not in question
    assert "99%" not in question and "DOC999" not in question
    assert actual.needs_clarification and actual.standalone_query == ""


def test_fallback_does_not_reuse_threat_topic_after_independent_switch(monkeypatch):
    turns = threat_history() + [
        {"role": "user", "content": "Tell me about zebra mussels."},
        {"role": "assistant", "content": "Zebra mussel control approaches vary by setting.", "context": {
            "standalone_query": "Tell me about zebra mussels.", "active_subject": "zebra mussels",
            "relation": "NEW_TOPIC", "needs_clarification": False,
        }},
    ]
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: "not JSON")
    actual = context.resolve_query(METHOD_FOLLOWUP, turns)
    assert "zebra mussels" in actual.clarification_question
    assert "conservation threats" not in actual.clarification_question
    assert actual.diagnostics()["history_messages_used"] == 2
