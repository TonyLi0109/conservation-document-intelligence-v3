"""Multi-turn intent, retrieval-topic and unchanged citation-boundary regressions."""

import ast
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import chat_context as context
import main
from data_models import KnowledgeArtifact
from database import KnowledgeStore


FIRST_QUESTION = "What are the effective methods to control invasive carp?"
FIRST_ANSWER = (
    "- Targeted harvest removes invasive carp.\n"
    "- Acoustic barriers deter invasive carp movement."
)
CARP_DATA = "Targeted harvest removed 43,000 pounds of invasive carp from the Lamine River."
PLANT_DATA = "Mechanical harvesting reduced invasive aquatic plants by 80 percent in the study."


def history(answer=FIRST_ANSWER):
    return [{"role": "user", "content": FIRST_QUESTION},
            {"role": "assistant", "content": answer}]


def resolved(query, subject="invasive carp", uses_history=True):
    return json.dumps({"standalone_query": query, "active_subject": subject,
                       "uses_history": uses_history, "needs_clarification": False,
                       "relation": "FOLLOW_UP" if uses_history else "NEW_TOPIC",
                       "selected_context": "Prior control methods" if uses_history else "",
                       "clarification_kind": "NONE"})


@pytest.fixture
def store():
    with KnowledgeStore(":memory:") as result:
        result.ingest_chunk(KnowledgeArtifact("DOC001", "Invasive aquatic plant control", "8", PLANT_DATA), [1., 0.])
        result.ingest_chunk(KnowledgeArtifact("DOC002", "Invasive carp field work", "12", CARP_DATA), [0., 1.])
        yield result


def test_reported_followup_changes_actual_retrieval_and_grounded_answer(store, monkeypatch):
    query = "Provide quantitative evidence on the effectiveness of invasive carp control by targeted harvest and acoustic barriers."
    captured = {}
    def rewrite(system, prompt, schema, **kwargs):
        captured["context"] = json.loads(prompt)
        assert schema["name"] == "v3_chat_context"
        return resolved(query)
    def embed(text):
        captured["embedding_query"] = text
        # Deliberately rank the unrelated plants first; the topic guard must fix it.
        return [1., 0.]
    def synthesize(system, prompt, artifacts, **kwargs):
        captured["synthesis"] = prompt
        assert {a.document_id for a in artifacts.values()} == {"DOC002"}
        return json.dumps({"preamble": "The corpus provides removal totals, but not a comparative effectiveness rate.",
                           "status": "partially_answered", "claims": [{
                               "text": "Targeted harvest removed 43,000 pounds of invasive carp from the Lamine River.",
                               "evidence_ids": ["K1"], "supporting_spans": [CARP_DATA],
                           }], "unsupported_facets": ["Comparative effectiveness rate for harvest and barriers"]})
    monkeypatch.setattr(context, "call_structured_llm", rewrite)
    monkeypatch.setattr(main, "generate_embedding", embed)
    monkeypatch.setattr(main, "call_llm", synthesize)
    diagnostics = {}
    answer, _, sources = main.ask_chatbot_with_context(
        "Provide some data on how effective these methods are.", store,
        history=history(), diagnostics=diagnostics,
    )
    assert captured["embedding_query"] == query == diagnostics["retrieval_query"]
    assert FIRST_ANSWER == captured["context"]["recent_messages"][-1]["content"]
    assert FIRST_ANSWER not in captured["synthesis"]
    assert "not factual evidence" in captured["synthesis"]
    assert "43,000" in answer and "DOC002" in answer
    assert "80 percent" not in answer and "aquatic plants" not in answer
    assert "Unsupported facets" in answer
    assert [a.document_id for a in sources] == ["DOC002"]


@pytest.mark.parametrize("question,standalone", [
    ("What are the biggest management challenges?", "What are the biggest invasive carp management challenges?"),
    ("Which of these appears most effective?", "Compare targeted harvest and acoustic barriers for invasive carp effectiveness."),
    ("What about cost?", "What are the costs of targeted harvest and acoustic barriers for invasive carp?"),
    ("How effective are they?", "How effective are targeted harvest and acoustic barriers for invasive carp?"),
    ("Explain the second one.", "Explain acoustic barriers for invasive carp."),
    ("Is there evidence for that?", "Verify the evidence for invasive carp control by targeted harvest and acoustic barriers."),
])
def test_implicit_and_explicit_references_are_resolved(question, standalone, monkeypatch):
    def rewrite(system, prompt, schema, **kwargs):
        data = json.loads(prompt)
        assert data["current_question"] == question
        assert "Acoustic barriers" in data["recent_messages"][-1]["content"]
        return resolved(standalone)
    monkeypatch.setattr(context, "call_structured_llm", rewrite)
    actual = context.resolve_query(question, history())
    assert actual.standalone_query == standalone
    assert actual.uses_history and actual.active_subject == "invasive carp"


def test_three_turn_followup_keeps_subject_and_previous_comparison(monkeypatch):
    turns = history()
    queries = iter([
        "Which invasive carp control method has the strongest evidence: targeted harvest or acoustic barriers?",
        "Verify the conclusion that targeted harvest has stronger evidence for invasive carp control; provide supporting data.",
    ])
    inputs = []
    def rewrite(system, prompt, schema, **kwargs):
        inputs.append(json.loads(prompt))
        return resolved(next(queries))
    monkeypatch.setattr(context, "call_structured_llm", rewrite)
    second = context.resolve_query("Which one has the strongest evidence?", turns)
    turns.extend([
        {"role": "user", "content": "Which one has the strongest evidence?"},
        {"role": "assistant", "content": "Targeted harvest has the strongest evidence.", "context": second.diagnostics()},
    ])
    third = context.resolve_query("Give me the supporting data.", turns)
    assert third.active_subject == "invasive carp"
    assert "targeted harvest" in third.standalone_query
    assert inputs[-1]["recent_messages"][-1]["resolved_context"]["standalone_query"] == second.standalone_query


@pytest.mark.parametrize("question", [
    "Now tell me about invasive aquatic plants.", "Now tell me about zebra mussels.",
    "Tell me about zebra mussels.", "Now tell me about bird migration.",
])
def test_explicit_topic_switch_does_not_inherit_carp(question, monkeypatch):
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: pytest.fail("Independent turn called contextualizer"))
    actual = context.resolve_query(question, history())
    assert actual.standalone_query == question
    assert not actual.uses_history
    assert "carp" not in actual.active_subject


def test_topic_switch_actually_retrieves_new_topic(store, monkeypatch):
    monkeypatch.setattr(main, "generate_embedding", lambda q: [1., 0.])
    def answer(system, prompt, artifacts, **kwargs):
        assert "Now tell me about invasive aquatic plants." in prompt
        assert "Targeted harvest removes" not in prompt
        assert any(a.document_id == "DOC001" for a in artifacts.values())
        return json.dumps({"preamble": "", "status": "answered", "claims": [{
            "text": PLANT_DATA, "evidence_ids": ["K1"], "supporting_spans": [PLANT_DATA],
        }], "unsupported_facets": []})
    monkeypatch.setattr(main, "call_llm", answer)
    result, _, sources = main.ask_chatbot_with_context("Now tell me about invasive aquatic plants.", store, history=history())
    assert "80 percent" in result
    assert [a.document_id for a in sources] == ["DOC001"]


def test_fresh_conversation_has_no_shared_state(monkeypatch):
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: resolved("What are invasive carp control costs?"))
    context.resolve_query("What about cost?", history())
    fresh = context.resolve_query("What about cost?", [])
    assert fresh.standalone_query == "What about cost?"
    assert fresh.active_subject == "" and not fresh.uses_history


def test_insufficient_subject_evidence_abstains_without_answer_call(monkeypatch):
    with KnowledgeStore(":memory:") as store:
        store.ingest_chunk(KnowledgeArtifact("DOC001", "Aquatic plant control", "1", PLANT_DATA), [1., 0.])
        monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: resolved("Provide quantitative invasive carp control effectiveness data."))
        monkeypatch.setattr(main, "generate_embedding", lambda q: [1., 0.])
        monkeypatch.setattr(main, "call_llm", lambda *a, **kw: pytest.fail("Unrelated evidence reached synthesis"))
        answer, _, sources = main.ask_chatbot_with_context("Give me the supporting data.", store, history=history())
        assert "does not provide enough evidence" in answer
        assert sources == [] and "80 percent" not in answer


def test_previous_answer_cannot_supply_a_fabricated_quote(store, monkeypatch):
    invented = "Harvest removes 99 percent of invasive carp each year."
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: resolved("Verify the effectiveness of invasive carp harvest."))
    monkeypatch.setattr(main, "generate_embedding", lambda q: [1., 0.])
    monkeypatch.setattr(main, "call_llm", lambda *a, **kw: json.dumps({
        "preamble": "", "status": "answered", "claims": [{
            "text": invented, "evidence_ids": ["K1"], "supporting_spans": [invented],
        }], "unsupported_facets": [],
    }))
    answer, preamble, _ = main.ask_chatbot_with_context("Is there evidence for that?", store, history=history(invented))
    assert "99 percent" not in answer
    assert "could not be validated" in preamble


@pytest.mark.parametrize("raw", ["not JSON", resolved("Data on penguins", "penguins"), json.dumps({
    "standalone_query": "", "active_subject": "", "uses_history": True, "needs_clarification": True,
})])
def test_unresolved_or_invented_subject_never_causes_generic_search(store, monkeypatch, raw):
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: raw)
    monkeypatch.setattr(store, "retrieve", lambda *a, **kw: pytest.fail("Unresolved reference reached search"))
    answer, _, sources = main.ask_chatbot_with_context("Which one?", store, history=history())
    assert "Please name" in answer and sources == []


def test_history_and_metadata_are_bounded_and_not_modified(monkeypatch):
    turns = [{"role": "user", "content": "x" * 10000} for _ in range(100)] + history()
    before = json.dumps(turns)
    def rewrite(system, prompt, schema, **kwargs):
        data = json.loads(prompt)
        assert len(data["recent_messages"]) == 2  # Latest explicit subject discards stale turns.
        assert all(len(m["content"]) <= context.MAX_MESSAGE_CHARACTERS for m in data["recent_messages"])
        return resolved("What are the costs of invasive carp control?")
    monkeypatch.setattr(context, "call_structured_llm", rewrite)
    context.resolve_query("What about cost?", turns)
    assert json.dumps(turns) == before


def test_ui_passes_thread_history_and_new_conversation_preserves_it(store):
    import conversation_history as threads
    source = Path(main.__file__).with_name("app.py").read_text(encoding="utf-8")
    renderer = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "render_chatbot_tab")
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    st = MagicMock()
    st.session_state = State(v3_chat_messages=history())
    st.button.return_value = False
    st.chat_input.return_value = "What about cost?"
    book = threads.empty_book()
    original_id = book["active_id"]
    book["conversations"][original_id]["messages"] = history()
    def prepare(store):
        if st.button.return_value:
            threads.new_conversation(book)
        st.session_state.v3_chat_messages = book["conversations"][book["active_id"]]["messages"]
        return book
    calls = []
    def ask(question, store, *, history, diagnostics, **kwargs):
        calls.append(history)
        diagnostics.update({"standalone_query": "Costs of invasive carp control", "active_subject": "invasive carp"})
        return "No cost data available.", "", []
    namespace = {"st": st, "os": __import__("os"), "KnowledgeStore": KnowledgeStore,
                 "MAX_HISTORY_MESSAGES": context.MAX_HISTORY_MESSAGES,
                 "prepare_conversations": prepare, "chat_history": threads,
                 "ask_chatbot_with_context": ask}
    exec(compile(ast.Module(body=[renderer], type_ignores=[]), "app.py", "exec"), namespace)
    namespace["render_chatbot_tab"](store, "gpt-4.1-mini")
    assert calls[0] == history()
    assert st.session_state.v3_chat_messages[-1]["context"]["active_subject"] == "invasive carp"
    st.button.return_value = True
    st.chat_input.return_value = "Tell me about zebra mussels."
    namespace["render_chatbot_tab"](store, "gpt-4.1-mini")
    assert calls[1] == []
    assert len(st.session_state.v3_chat_messages) == 2
    assert len(book["conversations"][original_id]["messages"]) == 4


def test_previous_report_is_identified_from_trusted_cited_metadata(monkeypatch):
    title = "Invasive Carp Strategic Science Plan"
    turns = history("The report describes deterrence research.")
    turns[-1]["sources"] = [KnowledgeArtifact("DOC012", title, "11", CARP_DATA)]
    def rewrite(system, prompt, schema, **kwargs):
        data = json.loads(prompt)
        assert data["recent_messages"][-1]["cited_documents"] == [{"document_id": "DOC012", "title": title}]
        assert CARP_DATA not in prompt
        return resolved(f"Summarize {title} (DOC012).", title)
    monkeypatch.setattr(context, "call_structured_llm", rewrite)
    result = context.resolve_query("Summarize that report.", turns)
    assert result.active_subject == title and "DOC012" in result.standalone_query


def test_contextualizer_failure_does_not_use_generic_query(store, monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("Provider unavailable")
    monkeypatch.setattr(context, "call_structured_llm", timeout)
    monkeypatch.setattr(main, "generate_embedding", lambda *a: pytest.fail("Unresolved query was embedded"))
    diagnostics = {}
    answer, _, sources = main.ask_chatbot_with_context("Which one is better?", store, history=history(), diagnostics=diagnostics)
    assert "Please retry" in answer and not sources
    assert diagnostics["method"] == "resolution_failed"
    assert diagnostics["retrieval_query"] == ""


def test_live_model_facet_phrase_is_normalized_to_grounded_entity(monkeypatch):
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: resolved(
        "Provide effectiveness data for invasive carp control by acoustic deterrents.",
        "effectiveness of invasive carp control methods",
    ))
    actual = context.resolve_query("Provide some data on how effective these methods are.", history())
    assert actual.active_subject == "invasive carp"
    assert actual.uses_history and not actual.needs_clarification


def test_streamlit_app_runs_three_turn_workflow_and_reset_offline(store, monkeypatch):
    """Exercise the complete Streamlit page, with only provider calls replaced."""
    import database
    import conversation_history as threads
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    monkeypatch.setattr(threads, "archive_component", lambda **kwargs: {"status": "ready", "page_id": "test-page"})
    monkeypatch.setattr(database, "KnowledgeStore", lambda *args, **kwargs: store)
    monkeypatch.setattr(database, "prepare_runtime_database", lambda *args: Path("unused.db"))
    monkeypatch.setattr(store, "upsert_document_sources", lambda *args: None)
    monkeypatch.setattr(main, "generate_embedding", lambda q: [1., 0.])
    def rewrite(system, prompt, schema, **kwargs):
        data = json.loads(prompt)
        assert data["recent_messages"][0]["content"] == FIRST_QUESTION
        return resolved("Provide quantitative evidence for invasive carp control by targeted harvest.")
    monkeypatch.setattr(context, "call_structured_llm", rewrite)
    def synthesize(system, prompt, artifacts, **kwargs):
        question = prompt.split("QUESTION:\n", 1)[1].split("\n\n", 1)[0]
        target = "DOC001" if "aquatic plants" in question else "DOC002"
        handle, artifact = next((h, a) for h, a in artifacts.items() if a.document_id == target)
        return json.dumps({"preamble": "", "status": "answered", "claims": [{
            "text": artifact.original_text_chunk, "evidence_ids": [handle],
            "supporting_spans": [artifact.original_text_chunk],
        }], "unsupported_facets": []})
    monkeypatch.setattr(main, "call_llm", synthesize)
    try:
        app = AppTest.from_file(str(Path(main.__file__).with_name("app.py")), default_timeout=15).run()
        assert not app.exception
        for question in (FIRST_QUESTION, "Provide some data on how effective these methods are.",
                         "Now tell me about invasive aquatic plants."):
            app.chat_input[0].set_value(question).run()
            assert not app.exception
        messages = app.session_state["v3_chat_messages"]
        original_id = app.session_state["v3_chat_book"]["active_id"]
        assert len(messages) == 6
        assert "43,000" in messages[3]["content"] and "80 percent" not in messages[3]["content"]
        assert messages[3]["context"]["uses_history"]
        assert "80 percent" in messages[5]["content"]
        assert not messages[5]["context"]["uses_history"]
        app.button(key="v3_new_conversation").click().run()
        assert app.session_state["v3_chat_messages"] == []
        assert len(app.session_state["v3_chat_book"]["conversations"][original_id]["messages"]) == 6
        second_id = app.session_state["v3_chat_book"]["active_id"]
        app.chat_input[0].set_value("Tell me about invasive aquatic plants.").run()
        assert app.session_state["v3_chat_book"]["active_id"] == second_id
        assert len(app.session_state["v3_chat_messages"]) == 2
        app.selectbox(key="v3_active_conversation").set_value(original_id).run()
        assert len(app.session_state["v3_chat_messages"]) == 6
        assert not app.exception
    finally:
        st.cache_resource.clear()
