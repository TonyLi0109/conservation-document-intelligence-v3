"""Content-labelled Wiki checks must fail for absent or misattributed evidence."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest

from data_models import KnowledgeArtifact
from evaluation.dataset import fixture_store
from evaluation.scenarios import evaluate_wiki_expectations, run_wiki_cases
import wiki_compiler as wiki


@pytest.fixture
def labelled_page():
    span = "The Agency monitors wetlands and provides habitat information."
    return {
        "concept": {
            "summary": span,
            "important_facts": [span],
            "supporting_evidence": [{"evidence_id": "K1", "exact_span": span}],
            "related_entities": [{"entity_name": "Wetland", "relationship_type": "monitors",
                                  "evidence_id": "K1", "exact_span": span}],
        },
        "artifacts": {"K1": KnowledgeArtifact("DOC999", "Agency report", "3", span)},
    }


@pytest.fixture
def expectation():
    return {
        "expected_evidence": [{"document_id": "DOC999", "contains": ["monitors wetlands", "habitat information"]}],
        "expected_related_entities": ["Wetland"],
    }


def test_inspected_quote_and_grounded_relationship_satisfy_content_labels(labelled_page, expectation):
    expectation["expected_evidence"][0]["contains"][0] = "MONITORS   WETLANDS"
    result = evaluate_wiki_expectations(labelled_page, expectation)
    assert result["passed"]
    assert result["metrics"] == {
        "expected_evidence_count": 1, "matched_expected_evidence_count": 1,
        "expected_related_entity_count": 1, "matched_expected_related_entity_count": 1,
    }


@pytest.mark.parametrize("change", ["wrong_document", "summary_only", "fabricated_quote", "split_quotes"])
def test_content_label_requires_all_phrases_in_one_valid_quote(labelled_page, expectation, change):
    if change == "wrong_document":
        expectation["expected_evidence"][0]["document_id"] = "DOC123"
    elif change == "summary_only":
        labelled_page["concept"]["supporting_evidence"] = []
    elif change == "fabricated_quote":
        labelled_page["artifacts"]["K1"] = replace(
            labelled_page["artifacts"]["K1"], original_text_chunk="Unrelated source text."
        )
    elif change == "split_quotes":
        labelled_page["concept"]["supporting_evidence"] = [
            {"evidence_id": "K1", "exact_span": "The Agency monitors wetlands"},
            {"evidence_id": "K1", "exact_span": "provides habitat information."},
        ]
    result = evaluate_wiki_expectations(labelled_page, expectation)
    assert not result["passed"]
    assert result["metrics"]["matched_expected_evidence_count"] == 0
    assert any("Expected Wiki evidence missing" in reason for reason in result["failure_reasons"])


@pytest.mark.parametrize("change", ["absent", "unbacked"])
def test_expected_relationship_must_have_linked_supporting_quote(labelled_page, expectation, change):
    if change == "absent":
        labelled_page["concept"]["related_entities"] = []
    else:
        labelled_page["concept"]["related_entities"][0]["exact_span"] = "Invented relationship."
    result = evaluate_wiki_expectations(labelled_page, expectation)
    assert not result["passed"]
    assert result["metrics"]["matched_expected_related_entity_count"] == 0
    assert any("Expected supported Wiki relationship missing" in reason for reason in result["failure_reasons"])


def test_runner_propagates_missing_content_labels_into_case_failure():
    with fixture_store() as store:
        case = {
            "case_id": "absent-inspected-evidence", "topic": "Invasive carp", "operation": "local_recompile",
            "expected_evidence": [{"document_id": "DOC999", "contains": ["nonexistent finding"]}],
            "expected_related_entities": ["Absent evaluated entity"],
        }
        row = run_wiki_cases(store, [case])[0]
    assert not row["passed"]
    assert row["metrics"]["matched_expected_evidence_count"] == 0
    assert row["metrics"]["matched_expected_related_entity_count"] == 0
    assert row["details"]["expected_evidence"] == case["expected_evidence"]


def test_matching_content_does_not_bypass_runner_canonical_document_validation():
    with fixture_store() as store:
        page = deepcopy(wiki.generate_extractive_wiki_concept("Invasive carp", store))
        evidence = page["concept"]["supporting_evidence"][0]
        handle = evidence["evidence_id"]
        page["artifacts"][handle] = replace(page["artifacts"][handle], document_id="DOC999")
        case = {
            "case_id": "misattributed-inspected-evidence", "topic": "Invasive carp", "operation": "local_recompile",
            "expected_evidence": [{"document_id": "DOC999", "contains": [evidence["exact_span"]]}],
        }
        with patch.object(wiki, "generate_extractive_wiki_concept", return_value=page):
            row = run_wiki_cases(store, [case])[0]
    assert row["metrics"]["matched_expected_evidence_count"] == 1
    assert not row["passed"]
    assert any("not the canonical source" in reason for reason in row["failure_reasons"])
