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


def test_same_page_preserves_threads_but_refresh_discards_all_history():
    state = {}
    book = threads.synchronize_page_session(state, "first-page")
    first_id = book["active_id"]
    for message in history():
        threads.append_message(book, first_id, message)
    second_id = threads.new_conversation(book)
    threads.append_message(book, second_id, {"role": "user", "content": "Tell me about zebra mussels."})
    assert threads.synchronize_page_session(state, "first-page") is book
    assert len(book["conversations"]) == 2
    assert len(book["conversations"][first_id]["messages"]) == 2

    fresh = threads.synchronize_page_session(state, "reloaded-page")
    assert fresh is not book
    assert len(fresh["conversations"]) == 1
    assert first_id not in fresh["conversations"] and second_id not in fresh["conversations"]
    assert state["v3_active_conversation"] == fresh["active_id"]
    assert state["v3_chat_messages"] is fresh["conversations"][fresh["active_id"]]["messages"]
    assert not state["v3_chat_messages"]


def test_separate_page_sessions_do_not_share_messages():
    first_state, second_state = {}, {}
    first = threads.synchronize_page_session(first_state, "first-page")
    threads.append_message(first, first["active_id"], history()[0])
    second = threads.synchronize_page_session(second_state, "second-page")
    assert not second["conversations"][second["active_id"]]["messages"]
    assert first_state["v3_chat_messages"] == first["conversations"][first["active_id"]]["messages"]


def test_app_discards_legacy_history_and_resets_when_page_identity_changes(store, monkeypatch):
    from pathlib import Path
    import database
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_resource.clear()
    monkeypatch.setattr(database, "KnowledgeStore", lambda *a, **kw: store)
    monkeypatch.setattr(database, "prepare_runtime_database", lambda *a: Path("unused.db"))
    monkeypatch.setattr(store, "upsert_document_sources", lambda *a: None)
    page = {"status": "ready", "page_id": "first-page"}
    calls = []
    def component(**kwargs):
        calls.append(kwargs)
        return page
    monkeypatch.setattr(threads, "archive_component", component)
    monkeypatch.setattr(threads, "restore_book", lambda *a: pytest.fail("Browser history must not be restored"))
    monkeypatch.setattr(threads, "export_book", lambda *a: pytest.fail("Chat content must not be sent to browser storage"))
    monkeypatch.setattr(main, "ask_chatbot_with_context", lambda *a, **kw: ("Answer in this page session.", "", []))
    try:
        app = AppTest.from_file(str(Path(main.__file__).with_name("app.py")), default_timeout=15)
        app.session_state["v3_chat_messages"] = history()
        app.session_state["v3_archive_revision"] = "legacy-revision"
        app.session_state["v3_archive_disabled"] = True
        app.run()
        assert not app.exception
        book = app.session_state["v3_chat_book"]
        assert not book["conversations"][book["active_id"]]["messages"]
        assert "v3_archive_revision" not in app.session_state
        assert "v3_archive_disabled" not in app.session_state
        app.chat_input[0].set_value(history()[0]["content"]).run()
        assert not app.exception
        assert len(app.chat_message) == 2
        app.button(key="v3_new_conversation").click().run()
        assert not app.exception
        assert len(app.session_state["v3_chat_book"]["conversations"]) == 2
        page["page_id"] = "reloaded-page"
        app.run()
        assert not app.exception
        fresh = app.session_state["v3_chat_book"]
        assert len(fresh["conversations"]) == 1 and fresh["active_id"] not in book["conversations"]
        assert not app.chat_message
        assert not app.session_state["v3_chat_messages"]
        assert all("snapshot" not in call and "expected_revision" not in call for call in calls)
        assert all(call["runtime_version"] for call in calls)
    finally:
        st.cache_resource.clear()
