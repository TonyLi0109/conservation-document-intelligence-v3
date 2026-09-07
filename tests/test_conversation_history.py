import json

import pytest

import chat_context as context
import conversation_history as threads
import main
from data_models import KnowledgeArtifact
from database import KnowledgeStore
from test_chat_context import history, resolved, store


@pytest.mark.parametrize("question", [
    "What are the major impacts of zebra mussels?",
    "Tell me about hydrilla.",
    "What reports discuss Asian longhorned beetle?",
    "How does climate change affect wetland restoration?",
    "What is hydrilla?",
    "Tell me about oak forests.",
    "Going back to invasive carp, which control method has the strongest evidence?",
])
def test_explicit_question_overrides_long_history_without_model(question, monkeypatch):
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: pytest.fail("Self-contained query called model"))
    result = context.resolve_query(question, history() * 20)
    assert result.relation == "NEW_TOPIC" and not result.uses_history
    assert result.standalone_query == question and result.selected_context == ""


def test_followup_after_new_subject_sees_only_relevant_segment(monkeypatch):
    turns = history() + [
        {"role": "user", "content": "Tell me about zebra mussels."},
        {"role": "assistant", "content": "Zebra mussels can be controlled through decontamination."},
    ]
    def rewrite(system, prompt, schema, **kwargs):
        assert "invasive carp" not in prompt
        return resolved("What management approaches are effective for zebra mussels?", "zebra mussels")
    monkeypatch.setattr(context, "call_structured_llm", rewrite)
    result = context.resolve_query("What management approaches are effective for them?", turns)
    assert result.active_subject == "zebra mussels"
    assert result.relation == "FOLLOW_UP"


def test_new_topic_model_decision_discards_stale_rewrite(monkeypatch):
    payload = json.loads(resolved("How do prescribed burns work, compared to invasive carp controls?", "prescribed burns", False))
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: json.dumps(payload))
    question = "How do prescribed burns work?"
    result = context.resolve_query(question, history())
    assert result.standalone_query == question
    assert result.relation == "NEW_TOPIC" and result.selected_context == ""


def test_partial_context_retrieves_other_fish_instead_of_filtering_them_out(monkeypatch):
    evidence = "Acoustic barriers are also used to deter invasive northern snakehead movement."
    with KnowledgeStore(":memory:") as store:
        store.ingest_chunk(KnowledgeArtifact("DOC010", "Northern snakehead management", "2", evidence), [1., 0.])
        payload = json.loads(resolved("Which of targeted harvest and acoustic barriers are also used for other invasive fish?", "other invasive fish"))
        payload.update(relation="PARTIAL_CONTEXT", selected_context="Targeted harvest and acoustic barriers from the previous answer")
        monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: json.dumps(payload))
        queries = []
        def embed(query):
            queries.append(query)
            return [1., 0.]
        monkeypatch.setattr(main, "generate_embedding", embed)
        def answer(system, prompt, artifacts, **kwargs):
            assert artifacts["K1"].document_id == "DOC010"
            return json.dumps({"preamble": "", "status": "answered", "claims": [{
                "text": evidence, "evidence_ids": ["K1"], "supporting_spans": [evidence],
            }], "unsupported_facets": []})
        monkeypatch.setattr(main, "call_llm", answer)
        diagnostics = {}
        result, _, sources = main.ask_chatbot_with_context(
            "Which of those methods are also used for other invasive fish?", store,
            history=history(), diagnostics=diagnostics,
        )
        assert "snakehead" in result and [s.document_id for s in sources] == ["DOC010"]
        assert diagnostics["relation"] == "PARTIAL_CONTEXT"
        assert diagnostics["selected_context"] and diagnostics["retrieval_query"] == queries[0]
        assert "other invasive fish" in queries[0]


def test_explicit_broader_scope_corrects_overly_narrow_model_classification(monkeypatch):
    monkeypatch.setattr(context, "call_structured_llm", lambda *a, **kw: resolved(
        "Which acoustic barriers are used for other invasive fish?", "invasive carp"))
    result = context.resolve_query("Which of those methods are used for other invasive fish?", history())
    assert result.relation == "PARTIAL_CONTEXT"
    assert "Subject: invasive carp" not in result.standalone_query


def test_two_thread_roundtrip_preserves_messages_and_isolated_context(store, monkeypatch):
    book = threads.empty_book()
    a = book["active_id"]
    for message in history():
        threads.append_message(book, a, message)
    b = threads.new_conversation(book)
    assert book["conversations"][b]["messages"] == []
    threads.append_message(book, b, {"role": "user", "content": "What are the impacts of zebra mussels?"})
    restored = threads.restore_book(json.loads(json.dumps(threads.export_book(book))), store)
    assert set(restored["conversations"]) == {a, b}
    assert restored["active_id"] == b
    assert restored["conversations"][a]["title"].startswith("What are the effective methods")
    assert all(m["timestamp"] for m in restored["conversations"][a]["messages"])
    def rewrite(system, prompt, schema, **kwargs):
        assert "zebra mussels" not in prompt
        assert "Acoustic barriers" in prompt
        return resolved("Which invasive carp method has the strongest quantitative evidence?")
    monkeypatch.setattr(context, "call_structured_llm", rewrite)
    actual = context.resolve_query("Which of those methods has the strongest quantitative evidence?",
                                   restored["conversations"][a]["messages"])
    assert actual.active_subject == "invasive carp"
    assert threads.empty_book()["conversations"] != restored["conversations"]


def test_archive_rehydrates_only_canonical_source_references(store):
    artifact = store.retrieve([1., 0.], 1)[0]
    book = threads.empty_book()
    threads.append_message(book, book["active_id"], {
        "role": "assistant", "content": "Prior cited answer", "sources": [artifact],
    })
    serialized = threads.export_book(book)
    assert artifact.original_text_chunk not in json.dumps(serialized)
    restored = threads.restore_book(serialized, store)
    sources = restored["conversations"][book["active_id"]]["messages"][0]["sources"]
    assert sources == [artifact]
    serialized["conversations"][book["active_id"]]["messages"][0]["source_refs"][0]["sha256"] = "invented"
    restored = threads.restore_book(serialized, store)
    message = restored["conversations"][book["active_id"]]["messages"][0]
    assert not message["sources"] and message["unavailable_source_refs"]
    assert message["content"] == "Prior cited answer"


def test_malformed_archive_fails_without_overwriting_input(store):
    payload = {"version": 99, "conversations": {}}
    with pytest.raises(ValueError):
        threads.restore_book(payload, store)
    assert payload == {"version": 99, "conversations": {}}


def test_existing_session_history_is_migrated_without_erasing_messages(store, monkeypatch):
    from pathlib import Path
    import database
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_resource.clear()
    monkeypatch.setattr(database, "KnowledgeStore", lambda *a, **kw: store)
    monkeypatch.setattr(database, "prepare_runtime_database", lambda *a: Path("unused.db"))
    monkeypatch.setattr(store, "upsert_document_sources", lambda *a: None)
    monkeypatch.setattr(threads, "archive_component", lambda **kw: {"status": "loaded", "archive": None, "revision": ""})
    try:
        app = AppTest.from_file(str(Path(main.__file__).with_name("app.py")), default_timeout=15)
        app.session_state["v3_chat_messages"] = history()
        app.run()
        assert not app.exception
        book = app.session_state["v3_chat_book"]
        restored = book["conversations"][book["active_id"]]
        assert [m["content"] for m in restored["messages"]] == [m["content"] for m in history()]
        assert restored["title"].startswith("What are the effective methods")
    finally:
        st.cache_resource.clear()
