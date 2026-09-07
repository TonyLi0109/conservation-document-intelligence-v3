import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from config import SETTINGS, V3_ROOT
from evaluation.dataset import corpus_copy, fingerprint, fixture_store, load_dataset
from evaluation.judges import assess_semantic_support
from evaluation.reporting import baseline_from, compare_baseline, render_summary, write_reports
from evaluation.run import run_retrieval_cases, run_suite


def report_fixture():
    return {"report_version": 1, "generated_at": "2026-01-01T00:00:00+00:00",
            "configuration": {"live": False}, "fingerprints": {"dataset": "fixed"},
            "case_count": 1, "passed_count": 1, "failed_count": 0,
            "cases": [{"case_id": "one", "category": "retrieval", "mode": "offline_fixture",
                       "passed": True, "metrics": {"recall_at_k": 1.0}, "failure_reasons": []}],
            "summary": {"retrieval/offline_fixture": {"case_count": 1, "passed": 1,
                                                     "metrics_mean": {"recall_at_k": 1.0}}}}


def test_baseline_detects_deterministic_drift_and_refuses_changed_dataset():
    previous = report_fixture()
    baseline = baseline_from(previous)
    current = copy.deepcopy(previous)
    current["cases"][0]["metrics"]["recall_at_k"] = 0.5
    result = compare_baseline(current, baseline)
    assert result["compatible"] and result["regressions"]
    current["fingerprints"]["dataset"] = "changed"
    result = compare_baseline(current, baseline)
    assert not result["compatible"] and not result["regressions"] and result["warnings"]


def test_live_differences_are_warnings_not_hard_regressions():
    previous = report_fixture()
    previous["cases"][0]["mode"] = "live_semantic"
    current = copy.deepcopy(previous)
    current["cases"][0].update(passed=False, metrics={"recall_at_k": 0.0})
    result = compare_baseline(current, baseline_from(previous))
    assert result["warnings"] and not result["regressions"]


def test_reports_keep_machine_metrics_and_human_failures(tmp_path):
    report = report_fixture()
    report["cases"][0].update(passed=False, failure_reasons=["Known wrong-topic result"])
    write_reports(report, tmp_path)
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8")) == report
    text = (tmp_path / "latest.md").read_text(encoding="utf-8")
    assert "Known wrong-topic result" in text and "partial" in text


def test_dataset_fingerprint_is_stable_across_formatting_and_line_endings(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_bytes(b'{"cases": [], "version": 1}\n')
    second.write_bytes(b'{\r\n "version": 1,\r\n "cases": []\r\n}')
    assert fingerprint(first) == fingerprint(second)


@pytest.mark.parametrize("live", [False, True])
def test_ui_evaluation_isolates_patches_and_requires_explicit_live_flag(tmp_path, monkeypatch, live):
    import evaluation
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "latest.json").write_text(json.dumps(report_fixture()), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    monkeypatch.setattr(evaluation.subprocess, "run", execute)
    class Store:
        database_path = SETTINGS.storage.database_path
    evaluation.run_evaluation(Store(), include_generation=live, output_dir=tmp_path)
    assert commands[0][:5] == [sys.executable, "-X", "utf8", "-m", "evaluation.run"]
    assert ("--live" in commands[0]) == live


@pytest.mark.parametrize("cases", [[], [{"case_id": "same"}, {"case_id": "same"}], [{"question": "missing ID"}]])
def test_malformed_case_sets_are_rejected(tmp_path, cases):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"dataset_id": "bad", "version": 1, "cases": cases}))
    with pytest.raises(ValueError):
        load_dataset(path)


def test_wrong_topic_vector_is_visible_as_retrieval_failure():
    case = {"case_id": "drift", "question": "invasive carp control", "retrieval_method": "semantic",
            "query_vector": [0., 0., 1.], "k": 1, "relevance": {"DOC901": 1},
            "minimum_recall": 1.0, "forbidden_document_ids": ["DOC903"]}
    with fixture_store() as store:
        result = run_retrieval_cases(store, {"synthetic": True, "cases": [case]})[0]
    assert not result["passed"]
    assert result["metrics"]["recall_at_k"] == 0.0
    assert "DOC903" in result["details"]["retrieved_document_ids"]


def test_invented_relevance_justification_is_not_accepted():
    case = {"case_id": "bad-label", "question": "carp", "relevance": {"DOC901": 1},
            "judgments": [{"document_id": "DOC901", "page_number": "999", "exact_span": "Invented label"}]}
    with fixture_store() as store:
        result = run_retrieval_cases(store, {"synthetic": True, "cases": [case]})[0]
    assert not result["passed"]
    assert any("justification" in reason for reason in result["failure_reasons"])


def test_provider_retrieval_failure_is_reported_without_leaking_error_content(monkeypatch):
    import main
    def unavailable(*args, **kwargs):
        raise RuntimeError("sensitive provider error detail")
    monkeypatch.setattr(main, "search_corpus", unavailable)
    with fixture_store() as store:
        result = run_retrieval_cases(store, {"cases": [{"case_id": "live-error", "question": "carp",
                                                       "relevance": {"DOC901": 1}}]}, live=True)[0]
    assert not result["passed"] and "RuntimeError" in result["failure_reasons"][0]
    assert "sensitive" not in json.dumps(result)


def test_corpus_top_k_option_overrides_dataset_default():
    with fixture_store() as store:
        result = run_retrieval_cases(store, {"cases": [{"case_id": "k", "question": "carp", "k": 5,
                                                       "relevance": {"DOC901": 1}}]}, top_k=2)[0]
    assert result["metrics"]["k"] == 2


def test_semantic_judge_is_never_implicit_and_checks_outputs():
    assert assess_semantic_support({"claims": []}, {})["status"] == "not_assessed"
    with fixture_store() as store:
        artifact = store.retrieve([1., 0., 0.], 1)[0]
    payload = {"claims": [{"text": "claim", "evidence_ids": ["K1"]}]}
    calls = []
    def judge(text, evidence):
        calls.append((text, evidence))
        return "partially_supported"
    assert assess_semantic_support(payload, {"K1": artifact}, judge=judge)["claims"] == ["partially_supported"]
    assert calls == [("claim", [artifact])]
    with pytest.raises(ValueError):
        assess_semantic_support(payload, {"K1": artifact}, judge=lambda *args: "invented status")


@pytest.mark.integration
def test_offline_suite_produces_repeatable_metrics_without_changing_corpus(tmp_path):
    original = fingerprint(SETTINGS.storage.database_path)
    report = run_suite()
    assert report["failed_count"] == 0, [(row["case_id"], row["failure_reasons"]) for row in report["cases"] if not row["passed"]]
    assert fingerprint(SETTINGS.storage.database_path) == original
    assert report["configuration"]["live"] is False and report["configuration"]["model"] is None
    assert {row["category"] for row in report["cases"]} >= {"wiki", "conversation", "provenance", "retrieval"}
    assert not compare_baseline(report, baseline_from(report))["regressions"]
    write_reports(report, tmp_path)
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))["case_count"] == len(report["cases"])


@pytest.mark.integration
def test_cli_runs_selected_offline_case_and_writes_both_reports(tmp_path):
    result = subprocess.run([sys.executable, "-m", "evaluation.run", "--case", "FIX-RET-CARP",
                             "--output-dir", str(tmp_path)], cwd=V3_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert report["case_count"] == 1 and report["cases"][0]["case_id"] == "FIX-RET-CARP"
    assert (tmp_path / "latest.md").is_file()


@pytest.mark.integration
def test_copy_changes_are_disposable_and_source_stays_identical():
    original = fingerprint(SETTINGS.storage.database_path)
    with corpus_copy(SETTINGS.storage.database_path) as store:
        copy_path = Path(store.database_path)
        store.connection.execute("DELETE FROM compiled_knowledge")
        store.connection.commit()
    assert not copy_path.exists()
    assert fingerprint(SETTINGS.storage.database_path) == original
