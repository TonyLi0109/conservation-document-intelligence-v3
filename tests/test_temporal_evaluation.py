"""Labelled temporal regressions and failure-detection tests for the evaluator."""

from copy import deepcopy
from pathlib import Path
import sqlite3

import pytest

from evaluation.dataset import DATASET_DIR, corpus_copy, load_dataset
from evaluation.temporal_cases import (
    evaluate_temporal_answer,
    run_temporal_cases,
    temporal_fixture_store,
)


DATASET = load_dataset(DATASET_DIR / "temporal.json")
FIXTURE_CASES = [case for case in DATASET["cases"] if case["mode"] == "fixture"]


@pytest.mark.parametrize("case", FIXTURE_CASES, ids=lambda case: case["case_id"])
def test_declared_temporal_fixture_behaviors(case):
    rows = run_temporal_cases(None, {**DATASET, "cases": [case]})
    assert len(rows) == 1
    assert rows[0]["mode"] == "offline_fixture"
    assert rows[0]["passed"], rows[0]["failure_reasons"]


@pytest.mark.integration
def test_real_corpus_temporal_labels_and_extraction():
    selected = [case for case in DATASET["cases"] if case["mode"] == "corpus"]
    with corpus_copy(Path(__file__).resolve().parents[1] / "data/corpus.db") as store:
        rows = run_temporal_cases(store, {**DATASET, "cases": selected})
    assert len(rows) == len(selected)
    failures = {row["case_id"]: row["failure_reasons"] for row in rows if not row["passed"]}
    assert not failures, failures


@pytest.mark.integration
def test_real_temporal_judgments_resolve_without_opening_corpus_for_writes():
    path = Path(__file__).resolve().parents[1] / "data/corpus.db"
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for case in DATASET["cases"]:
            for judgment in case.get("judgments", []):
                chunks = connection.execute(
                    "SELECT original_text_chunk FROM knowledge_artifacts WHERE document_id=? AND page_number=?",
                    (judgment["document_id"], judgment["page_number"]),
                ).fetchall()
                assert any(judgment["exact_span"] in row[0] for row in chunks), case["case_id"]
    finally:
        connection.close()


def test_evaluator_fails_when_superseded_document_is_preferred(monkeypatch):
    import temporal

    original = temporal.select_temporal_evidence

    def stale_preference(*args, **kwargs):
        selected = original(*args, **kwargs)
        selected["selected_document_ids"] = ["DOC951", "DOC952"]
        return selected

    monkeypatch.setattr(temporal, "select_temporal_evidence", stale_preference)
    case = next(case for case in DATASET["cases"] if case["case_id"] == "temporal_explicit_supersession")
    row = run_temporal_cases(None, {**DATASET, "cases": [case]})[0]
    assert not row["passed"]
    assert row["metrics"]["correct_current_document_rate"] == 0
    assert row["metrics"]["inapplicable_document_preference_errors"] == 1


def test_temporal_answer_audit_detects_fabricated_direction_despite_valid_citation():
    import document_lifecycle
    import temporal
    from validator import _citation

    with temporal_fixture_store(DATASET, ["DOC951", "DOC952"]) as store:
        selection = temporal.select_temporal_evidence(
            "What is the current invasive carp control guidance?", store, as_of=DATASET["as_of"],
        )
        answer, _, sources = temporal.render_temporal_answer(selection, store)
        index = document_lifecycle.get_lifecycles(store)
        valid = evaluate_temporal_answer(answer, sources, selection, store, index)
        assert valid["unsupported_supersession_claim_count"] == 0
        assert not valid["failures"]
        forged = answer + "\n- DOC951 explicitly supersedes DOC952. " + _citation(sources[0])

        invalid = evaluate_temporal_answer(forged, sources, selection, store, index)

        assert invalid["invalid_citation_count"] == 0
        assert invalid["unsupported_supersession_claim_count"] == 1
        assert invalid["failures"]


def test_temporal_answer_audit_rejects_relationship_with_wrong_canonical_page():
    import document_lifecycle
    import temporal

    with temporal_fixture_store(DATASET, ["DOC951", "DOC952"]) as store:
        selection = temporal.select_temporal_evidence(
            "What is the current invasive carp control guidance?", store, as_of=DATASET["as_of"],
        )
        answer, _, sources = temporal.render_temporal_answer(selection, store)
        forged_index = deepcopy(document_lifecycle.get_lifecycles(store))
        forged_index["DOC952"]["relations"][0]["evidence"]["page_number"] = "99"

        checks = evaluate_temporal_answer(answer, sources, selection, store, forged_index)

        assert checks["invalid_relationship_evidence_count"] == 1
        assert checks["unsupported_supersession_claim_count"] == 1


def test_temporal_metrics_do_not_claim_current_accuracy_for_unlabelled_relationship():
    case = next(case for case in DATASET["cases"] if case["case_id"] == "temporal_unknown_relationship_is_uncertain")
    row = run_temporal_cases(None, {**DATASET, "cases": [case]})[0]
    assert "correct_current_document_rate" not in row["metrics"]
    assert row["metrics"]["version_relationship_accuracy"] == 1
    assert row["metrics"]["temporal_intent_accuracy"] == 1


def test_fixture_currentness_uses_fixed_as_of_not_machine_clock():
    cases = [case for case in DATASET["cases"] if case["case_id"] in {
        "temporal_future_effective_not_current", "temporal_revision_becomes_effective",
    }]
    rows = run_temporal_cases(None, {**DATASET, "cases": cases})
    assert [row["details"]["as_of"] for row in rows] == ["2026-09-07", "2027-02-01"]
    assert [row["details"]["selected_document_ids"][0] for row in rows] == ["DOC952", "DOC954"]


def test_independent_fixture_cases_do_not_share_lifecycle_state():
    cases = [case for case in DATASET["cases"] if case["case_id"] in {
        "temporal_explicit_supersession", "temporal_final_beats_draft",
    }]
    rows = run_temporal_cases(None, {**DATASET, "cases": cases})
    first_ids, second_ids = (set(row["details"]["selected_document_ids"]) for row in rows)
    assert first_ids <= {"DOC951", "DOC952"}
    assert second_ids <= {"DOC958", "DOC959"}
    assert not first_ids.intersection(second_ids)
