"""Scenario assertions protect real regression signals, not model output wording."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evaluation.dataset import fixture_store
from evaluation.scenarios import (
    _canonical_wiki_links,
    evaluate_wiki,
    run_conversation_cases,
    run_wiki_cases,
)
import wiki_compiler as wiki


DATASETS = Path(__file__).parents[1] / "evaluation" / "datasets"
CHAT_CASES = json.loads((DATASETS / "conversations.json").read_text(encoding="utf-8"))["cases"]
WIKI_CASES = json.loads((DATASETS / "wiki.json").read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CHAT_CASES, ids=lambda case: case["case_id"])
def test_declarative_conversations_exercise_real_pipeline_offline(case):
    row = run_conversation_cases([case])[0]
    assert row["passed"], row["failure_reasons"]
    assert row["mode"] == "offline_fixture"
    assert row["details"]["semantic_quality_measured"] is False
    # Reports must remain machine readable without serializing mutable stores,
    # canonical dataclasses or production user messages.
    json.dumps(row)


@pytest.mark.parametrize("case", WIKI_CASES, ids=lambda case: case["case_id"])
def test_declarative_wiki_paths_validate_provenance_offline(case):
    with fixture_store() as store:
        row = run_wiki_cases(store, [case])[0]
    assert row["passed"], row["failure_reasons"]
    assert row["details"]["semantic_quality_measured"] is False
    json.dumps(row)


def test_wrong_topic_resolver_cannot_make_scenario_pass_by_using_its_own_expected_output():
    case = deepcopy(CHAT_CASES[0])
    case["turns"][1]["resolver"].update(
        active_subject="invasive aquatic plants",
        standalone_query="Provide invasive aquatic plants herbicide efficacy data.",
    )
    result = run_conversation_cases([case])[0]
    assert not result["passed"]
    assert any("missing cited fixture documents" in reason or "needs_clarification" in reason
               for reason in result["failure_reasons"])


def test_changed_expected_citation_is_reported_as_a_failure():
    case = deepcopy(CHAT_CASES[1])
    case["turns"][1]["expected"]["source_document_ids"] = ["DOC901"]
    result = run_conversation_cases([case])[0]
    assert not result["passed"]
    assert any("missing cited fixture documents ['DOC901']" in reason for reason in result["failure_reasons"])


def test_long_conversation_remains_bounded_without_losing_subject():
    case = deepcopy(CHAT_CASES[0])
    followup = case["turns"][1]
    case["turns"] = [case["turns"][0]] + [deepcopy(followup) for _ in range(8)]
    case["turns"][-1]["expected"].update(history_messages_used=6, max_history_messages=6)
    case["expected_thread_message_counts"] = {"A": 18}
    row = run_conversation_cases([case])[0]
    assert row["passed"], row["failure_reasons"]
    assert row["details"]["turns"][-1]["active_subject"] == "invasive carp"


def test_conversation_no_evidence_does_not_request_synthesis():
    case = next(item for item in CHAT_CASES if item["case_id"] == "conversation-no-evidence")
    row = run_conversation_cases([case])[0]
    assert row["passed"], row["failure_reasons"]
    assert row["metrics"]["scripted_synthesis_calls"] == 0


def test_wiki_metric_distinguishes_duplicate_links_from_distinct_sources():
    with fixture_store() as store:
        page = wiki.generate_extractive_wiki_concept("Invasive carp", store)
        duplicate = deepcopy(page)
        duplicate["concept"]["supporting_evidence"].append(deepcopy(duplicate["concept"]["supporting_evidence"][0]))
        result = evaluate_wiki(duplicate)
        assert not result["passed"]
        assert result["metrics"]["duplicate_evidence_count"] == 1
        assert result["metrics"]["invalid_citation_count"] == 0


def test_wiki_metric_rejects_unknown_evidence_and_fake_quote():
    with fixture_store() as store:
        page = wiki.generate_extractive_wiki_concept("Invasive carp", store)
        page["concept"]["supporting_evidence"][:2] = [
            {"evidence_id": "K999", "exact_span": "Invented source handle"},
            {"evidence_id": "K1", "exact_span": "A quotation absent from every fixture document."},
        ]
        result = evaluate_wiki(page)
        assert not result["passed"]
        assert result["metrics"]["invalid_citation_count"] == 2
        assert result["metrics"]["citation_validity_rate"] < 1.0


def test_wiki_metric_checks_nested_schema_and_missing_relationship_evidence():
    with fixture_store() as store:
        page = wiki.generate_extractive_wiki_concept("Invasive carp", store)
        relation = page["concept"]["related_entities"][0]
        page["concept"]["supporting_evidence"] = [
            item for item in page["concept"]["supporting_evidence"]
            if item["exact_span"] != relation["exact_span"]
        ]
        result = evaluate_wiki(page)
        assert not result["passed"]
        assert result["metrics"]["missing_relationship_evidence_count"] >= 1
        page["concept"]["related_entities"][0]["unrecognized_field"] = "bad schema"
        assert evaluate_wiki(page)["metrics"]["schema_valid"] is False


def test_wiki_repeating_one_fact_does_not_satisfy_three_fact_threshold():
    with fixture_store() as store:
        page = wiki.generate_extractive_wiki_concept("Invasive carp", store)
        page["concept"]["important_facts"] = [page["concept"]["important_facts"][0]] * 5
        result = evaluate_wiki(page, {"min_important_facts": 3})
        assert not result["passed"]
        assert result["metrics"]["important_fact_count"] == 1
        assert result["metrics"]["duplicate_fact_count"] == 4


def test_missing_optional_url_is_reported_without_invalidating_canonical_provenance():
    with fixture_store() as store:
        page = wiki.generate_extractive_wiki_concept("Invasive carp", store)
        page["artifacts"] = {handle: replace(artifact, source_url=None)
                             for handle, artifact in page["artifacts"].items()}
        result = evaluate_wiki(page)
        assert result["passed"], result["failure_reasons"]
        assert result["metrics"]["missing_source_url_count"] > 0
        assert result["metrics"]["source_url_coverage"] == 0.0
        assert result["metrics"]["citation_validity_rate"] == 1.0


@pytest.mark.parametrize("change", [{"page_number": "9999"}, {"document_id": "DOC9999"}])
def test_wiki_canonical_lookup_rejects_invalid_document_or_page(change):
    with fixture_store() as store:
        page = wiki.generate_extractive_wiki_concept("Invasive carp", store)
        page["artifacts"]["K1"] = replace(page["artifacts"]["K1"], **change)
        assert _canonical_wiki_links(page, store)


def test_cache_hit_performs_neither_retrieval_nor_provider_call():
    with fixture_store() as store:
        wiki.generate_extractive_wiki_concept("Invasive carp", store)
        with patch.object(store, "retrieve", side_effect=AssertionError("Cached page retrieved again")):
            row = run_wiki_cases(store, [WIKI_CASES[0]])[0]
        assert row["passed"], row["failure_reasons"]
        assert row["metrics"]["initial_cache_available"] is True
        assert row["metrics"]["provider_calls"] == 0
