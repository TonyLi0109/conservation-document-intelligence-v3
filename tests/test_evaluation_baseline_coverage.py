"""Regression gates must detect lost cases and lost measurements, not just drops."""

import copy

import pytest

from evaluation.reporting import RANK_METRICS, baseline_from, compare_baseline, render_summary


def report_with(*modes):
    return {
        "configuration": {"top_k": 5},
        "fingerprints": {"dataset": "fixed", "corpus": "fixed"},
        "cases": [
            {"case_id": "shared-id", "category": "retrieval", "mode": mode,
             "passed": True, "metrics": {metric: 1.0 for metric in RANK_METRICS}}
            for mode in modes
        ],
    }


@pytest.mark.parametrize("mode,destination", [
    ("offline_fixture", "regressions"),
    ("offline_corpus", "regressions"),
    ("live_semantic", "warnings"),
    ("live_generation", "warnings"),
])
def test_missing_baseline_cases_cannot_silently_pass(mode, destination):
    previous = report_with(mode)
    current = copy.deepcopy(previous)
    current["cases"] = []
    result = compare_baseline(current, baseline_from(previous))
    assert result["compatible"]
    assert len(result[destination]) == 1
    assert f"retrieval/{mode}/shared-id" in result[destination][0]
    assert "missing from current run" in result[destination][0]
    if destination == "warnings":
        assert not result["regressions"]


@pytest.mark.parametrize("metric", RANK_METRICS)
@pytest.mark.parametrize("replacement", ["missing", None, False, "unavailable"])
def test_lost_offline_ranking_measurement_is_a_regression(metric, replacement):
    previous = report_with("offline_fixture")
    current = copy.deepcopy(previous)
    if replacement == "missing":
        del current["cases"][0]["metrics"][metric]
    else:
        current["cases"][0]["metrics"][metric] = replacement
    result = compare_baseline(current, baseline_from(previous))
    assert len(result["regressions"]) == 1
    assert metric in result["regressions"][0]
    assert "retrieval/offline_fixture/shared-id" in result["regressions"][0]
    assert all(delta["metric"] != metric for delta in result["metric_deltas"])


@pytest.mark.parametrize("replacement", ["missing", None])
def test_lost_live_measurements_warn_without_hard_failure(replacement):
    previous = report_with("live_semantic")
    current = copy.deepcopy(previous)
    if replacement == "missing":
        del current["cases"][0]["metrics"]["mrr"]
    else:
        current["cases"][0]["metrics"]["mrr"] = replacement
    result = compare_baseline(current, baseline_from(previous))
    assert not result["regressions"]
    assert len(result["warnings"]) == 1
    assert "retrieval/live_semantic/shared-id" in result["warnings"][0]


def test_same_case_id_in_another_mode_does_not_hide_missing_case():
    previous = report_with("offline_corpus", "live_semantic")
    current = copy.deepcopy(previous)
    current["cases"].pop(0)
    result = compare_baseline(current, baseline_from(previous))
    assert len(result["regressions"]) == 1
    assert "retrieval/offline_corpus/shared-id" in result["regressions"][0]
    assert not result["warnings"]


def test_same_case_id_in_another_category_does_not_hide_missing_case():
    previous = report_with("offline_fixture")
    current = copy.deepcopy(previous)
    current["cases"][0]["category"] = "conversation"
    result = compare_baseline(current, baseline_from(previous))
    assert "retrieval/offline_fixture/shared-id" in result["regressions"][0]
    assert "conversation/offline_fixture/shared-id" in result["warnings"][0]


def test_undefined_before_and_after_does_not_create_regression():
    previous = report_with("offline_fixture")
    previous["cases"][0]["metrics"] = {"mrr": None}
    current = copy.deepcopy(previous)
    current["cases"][0]["metrics"] = {}
    result = compare_baseline(current, baseline_from(previous))
    assert not result["regressions"]
    assert not result["warnings"]
    assert not result["metric_deltas"]


def test_changed_execution_settings_still_skip_incompatible_coverage_comparison():
    previous = report_with("offline_fixture")
    current = copy.deepcopy(previous)
    current["configuration"]["top_k"] = 3
    current["cases"] = []
    result = compare_baseline(current, baseline_from(previous))
    assert not result["compatible"]
    assert not result["regressions"]
    assert len(result["warnings"]) == 1
    assert "comparison skipped" in result["warnings"][0]


def test_metric_deltas_identify_category_and_execution_mode():
    previous = report_with("offline_corpus", "live_semantic")
    current = copy.deepcopy(previous)
    current["cases"][0]["metrics"]["mrr"] = 0.5
    result = compare_baseline(current, baseline_from(previous))
    delta = next(item for item in result["metric_deltas"] if item["delta"] < 0)
    assert delta == {"case_id": "shared-id", "category": "retrieval", "mode": "offline_corpus", "metric": "mrr", "delta": -0.5}


def test_live_raw_claim_failures_are_visible_in_human_summary():
    report = {
        "generated_at": "2026-01-01T00:00:00+00:00", "case_count": 1,
        "passed_count": 0, "failed_count": 1, "cases": [],
        "summary": {"grounded_answer/live_generation": {
            "case_count": 1, "passed": 0,
            "metrics_mean": {"raw_numeric_support_rate": 0.0,
                             "raw_provenance_validity_rate": 1.0,
                             "raw_grounded_proxy_rate": 0.0},
        }},
    }
    summary = render_summary(report)
    assert "raw_numeric_support_rate: 0.0000" in summary
    assert "raw_provenance_validity_rate: 1.0000" in summary
    assert "raw_grounded_proxy_rate: 0.0000" in summary
