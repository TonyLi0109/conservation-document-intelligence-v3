"""Temporal follow-ups reuse intent without turning old answers into evidence."""

import json

import pytest

import chat_context as context
from data_models import KnowledgeArtifact


@pytest.fixture(autouse=True)
def no_context_provider(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("An obvious temporal follow-up called the context provider")
    monkeypatch.setattr(context, "call_structured_llm", unexpected)


def historical_history(subject="invasive carp", year="2020"):
    source = KnowledgeArtifact("DOC901", f"{subject} Strategy {year}", "1",
                               f"The {year} strategy discusses management options.")
    return [
        {"role": "user", "content": f"What did the {year} {subject} strategy recommend?"},
        {"role": "assistant", "content": "The earlier strategy recommended targeted harvest.",
         "sources": [source], "context": {
             "active_subject": subject, "relation": "NEW_TOPIC",
             "standalone_query": f"What did the {year} {subject} strategy recommend?",
         }},
    ]


@pytest.mark.parametrize("question", [
    "Is that still the current recommendation?",
    "Is that still current?",
    "What is the current recommendation?",
    "What does the latest report recommend?",
    "What did earlier guidance recommend?",
    "Has this recommendation changed?",
    "How has the guidance changed over time?",
])
def test_obvious_temporal_followups_keep_topic_without_old_year(question):
    resolved = context.resolve_query(question, historical_history())
    assert resolved.uses_history and not resolved.needs_clarification
    assert resolved.method == "temporal_context"
    assert resolved.active_subject == "invasive carp"
    assert resolved.relation == "FOLLOW_UP"
    assert "invasive carp" in resolved.standalone_query
    assert "2020" not in resolved.standalone_query
    assert "DOC901" not in resolved.standalone_query
    assert resolved.history_messages_used == 2


def test_explicit_historical_year_in_current_question_is_preserved():
    question = "What did the 2018 guidance recommend?"
    resolved = context.resolve_query(question, historical_history())
    assert resolved.method == "temporal_context"
    assert "2018" in resolved.standalone_query
    assert "2020" not in resolved.standalone_query


def test_comparison_keeps_user_requested_years_without_old_constraint():
    resolved = context.resolve_query("How did the recommendations change from 2018 to 2024?", historical_history())
    assert resolved.method == "temporal_context"
    assert "2018" in resolved.standalone_query and "2024" in resolved.standalone_query
    assert "2020" not in resolved.standalone_query


def test_saved_report_subject_loses_its_previous_year_for_current_lookup():
    history = historical_history(subject="River Habitat Management", year="2019")
    history[-1]["context"]["active_subject"] = "River Habitat Management Strategy 2019"
    resolved = context.resolve_query("Is that still current?", history)
    assert resolved.active_subject == "River Habitat Management Strategy"
    assert "2019" not in resolved.standalone_query


def test_unambiguous_trusted_report_title_can_anchor_without_saved_context():
    history = historical_history(subject="River Habitat Management", year="2019")
    del history[-1]["context"]
    resolved = context.resolve_query("What does the latest report recommend?", history)
    assert resolved.method == "temporal_context"
    assert "River Habitat Management Strategy" in resolved.standalone_query
    assert "2019" not in resolved.standalone_query


def test_generic_but_trusted_report_title_is_a_valid_family_anchor():
    source = KnowledgeArtifact("DOC901", "Guidance 2020", "1", "Original guidance.")
    history = [{"role": "user", "content": "What did the 2020 guidance recommend?"},
               {"role": "assistant", "content": "It recommended monitoring.", "sources": [source]}]
    resolved = context.resolve_query("Is that still current?", history)
    assert resolved.active_subject == "Guidance"
    assert "2020" not in resolved.standalone_query


def test_latest_independent_topic_wins_over_old_historical_context():
    history = historical_history() + historical_history(subject="zebra mussels", year="2023")
    resolved = context.resolve_query("Has this recommendation changed?", history)
    assert resolved.active_subject == "zebra mussels"
    assert "carp" not in resolved.standalone_query
    assert "2020" not in resolved.standalone_query
    assert resolved.history_messages_used == 2


@pytest.mark.parametrize("question", [
    "What is the current guidance for zebra mussels?",
    "What does the latest hydrilla report recommend?",
    "Now tell me about climate change.",
])
def test_explicit_new_topic_does_not_inherit_historical_carp(question):
    resolved = context.resolve_query(question, historical_history())
    assert resolved.standalone_query == question
    assert not resolved.uses_history
    assert resolved.relation == "NEW_TOPIC"
    assert "carp" not in resolved.active_subject


def test_temporal_followup_after_reset_has_no_recoverable_history():
    context.resolve_query("Is that still current?", historical_history())
    resolved = context.resolve_query("Is that still current?", [])
    assert not resolved.uses_history
    assert resolved.active_subject == ""
    assert resolved.standalone_query == "Is that still current?"


def test_prior_claims_do_not_become_temporal_facts_or_search_constraints():
    history = historical_history()
    history[-1]["content"] = "A 2035 revision supersedes the old strategy and requires invented treatment."
    resolved = context.resolve_query("Is that still current?", history)
    assert "2035" not in resolved.standalone_query
    assert "supersedes" not in resolved.standalone_query
    assert "invented treatment" not in resolved.standalone_query


def test_context_remains_bounded_for_long_temporal_threads():
    history = historical_history()
    for _ in range(6):
        history.extend([
            {"role": "user", "content": "Is that still current?"},
            {"role": "assistant", "content": "Only the original source was found.",
             "context": {"active_subject": "invasive carp", "relation": "FOLLOW_UP"}},
        ])
    resolved = context.resolve_query("What did earlier guidance recommend?", history)
    assert resolved.history_messages_used == context.MAX_HISTORY_MESSAGES
    assert resolved.active_subject == "invasive carp"


def test_temporal_words_do_not_override_broadened_species_scope(monkeypatch):
    query = "Are those previous methods also used for other invasive fish?"
    response = {"standalone_query": "Investigate harvest methods for other invasive fish.",
                "active_subject": "other invasive fish", "uses_history": True,
                "needs_clarification": False, "relation": "PARTIAL_CONTEXT",
                "selected_context": "Previous methods", "clarification_kind": "NONE"}
    calls = []
    def model(*args, **kwargs):
        calls.append(True)
        return json.dumps(response)
    monkeypatch.setattr(context, "call_structured_llm", model)
    resolved = context.resolve_query(query, historical_history())
    assert calls
    assert resolved.relation == "PARTIAL_CONTEXT"
    assert resolved.active_subject == "other invasive fish"
    assert "carp" not in resolved.standalone_query


def test_direct_threat_followup_stays_direct():
    history = [
        {"role": "user", "content": "What are the main conservation threats mentioned across the documents?"},
        {"role": "assistant", "content": "- The main conservation threats include direct drivers such as land and sea use change, direct exploitation of organisms, climate change, pollution, and invasive alien species."},
    ]
    resolved = context.resolve_query("provide some data on how effective these methods are", history)
    assert resolved.method == "context_inferred"
    assert not resolved.needs_clarification
    assert "measures addressing conservation threats" in resolved.standalone_query
