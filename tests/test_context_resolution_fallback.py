"""Invalid context output must preserve the user's topic without inventing evidence."""

import json
import re

import pytest

import chat_context as context
import main
from database import KnowledgeStore
from test_chat_context import resolved
from test_contextual_clarification import (
    CLARIFICATION,
    METHOD_FOLLOWUP,
    THREAT_QUESTION,
    ambiguous_payload,
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
def test_invalid_output_preserves_reported_topic_without_search(monkeypatch, raw):
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: raw)
    monkeypatch.setattr(main, "generate_embedding", lambda *a: pytest.fail("Unresolved methods were embedded"))
    monkeypatch.setattr(main, "call_llm", lambda *a, **kw: pytest.fail("Unresolved methods reached synthesis"))
    with KnowledgeStore(":memory:") as store:
        monkeypatch.setattr(store, "retrieve", lambda *a, **kw: pytest.fail("Unresolved methods reached retrieval"))
        monkeypatch.setattr(store, "retrieve_document_matches", lambda *a, **kw: pytest.fail("Unresolved methods reached document search"))
        diagnostics = {}
        answer, preamble, sources = main.ask_chatbot_with_context(
            METHOD_FOLLOWUP, store, history=model_owned_threat_history(), diagnostics=diagnostics,
        )
    assert answer == TARGETED_FALLBACK
    assert "Please name the subject" not in answer
    assert "99%" not in answer + preamble and "DOC999" not in answer + preamble
    assert sources == []
    assert diagnostics["uses_history"] is True
    assert diagnostics["needs_clarification"] is True
    assert diagnostics["relation"] == "FOLLOW_UP"
    assert diagnostics["method"] == "invalid_context"
    assert diagnostics["retrieval_query"] == ""
    assert diagnostics["clarification_question"] == TARGETED_FALLBACK
    assert diagnostics["history_messages_used"] == 2
    assert re.fullmatch(r"[a-z_]+", diagnostics["resolution_error"])
    assert len(diagnostics["resolution_error"]) <= 80


def test_valid_specific_ambiguity_keeps_existing_clarification(monkeypatch):
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: ambiguous_payload())
    actual = context.resolve_query(METHOD_FOLLOWUP, model_owned_threat_history())
    assert actual.method == "ambiguous"
    assert actual.clarification_question == CLARIFICATION
    assert actual.diagnostics()["resolution_error"] == ""


def test_fallback_does_not_assert_real_methods_were_absent(monkeypatch):
    turns = threat_history()
    turns[-1]["content"] = (
        "The reports discuss conservation threats and methods to address them: "
        "riparian restoration and invasive-species removal."
    )
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: "not JSON")
    actual = context.resolve_query(METHOD_FOLLOWUP, turns)
    assert actual.clarification_question == TARGETED_FALLBACK
    assert "rather than control methods" not in actual.clarification_question
    assert "did not identify" not in actual.clarification_question


def test_invalid_output_fallback_retains_topic_for_next_reply(monkeypatch):
    turns = model_owned_threat_history()
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: "not JSON")
    failed = context.resolve_query(METHOD_FOLLOWUP, turns)
    turns.extend([
        {"role": "user", "content": METHOD_FOLLOWUP},
        {"role": "assistant", "content": failed.clarification_question,
         "context": failed.diagnostics()},
    ])
    query = "Provide quantitative data on the effectiveness of measures addressing conservation threats."

    def rewrite(system, prompt, schema, **kwargs):
        recent = json.loads(prompt)["recent_messages"]
        assert len(recent) == 4
        assert recent[0]["content"] == THREAT_QUESTION
        assert recent[2]["content"] == METHOD_FOLLOWUP
        assert recent[-1]["resolved_context"]["needs_clarification"] is True
        assert recent[-1]["resolved_context"]["clarification_question"] == TARGETED_FALLBACK
        return resolved(query, "conservation threats")

    monkeypatch.setattr(context, "call_structured_llm", rewrite)
    actual = context.resolve_query("I mean their effectiveness.", turns)
    assert actual.uses_history and not actual.needs_clarification
    assert actual.standalone_query == query
    assert actual.active_subject == "conservation threats"
    assert actual.diagnostics()["history_messages_used"] == 4


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
