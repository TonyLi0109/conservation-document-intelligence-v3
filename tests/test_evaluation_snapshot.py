"""Protect evaluated snapshot identity and per-request report freshness offline."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest

from data_models import KnowledgeArtifact
from database import KnowledgeStore
import evaluation
from evaluation.dataset import corpus_copy, fingerprint


def _source_database(path):
    with KnowledgeStore(path) as store:
        store.ingest_chunk(KnowledgeArtifact("DOC991", "Snapshot fixture", "1", "Initial canonical evidence."), [1., 0.])


def _set_text(connection, text):
    connection.execute("UPDATE knowledge_artifacts SET original_text_chunk=? WHERE document_id='DOC991'", (text,))
    connection.commit()


def _text(store):
    return store.connection.execute("SELECT original_text_chunk FROM knowledge_artifacts WHERE document_id='DOC991'").fetchone()[0]


def test_snapshot_fingerprint_includes_committed_wal_and_ignores_later_source_writes(tmp_path):
    source = tmp_path / "source.db"
    _source_database(source)
    with closing(sqlite3.connect(source)) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        # Keep the writer open so committed changes remain in the WAL while the
        # ordinary main-file hash does not reflect them.
        main_file_before = fingerprint(source)
        _set_text(writer, "Committed evidence in the WAL.")
        assert fingerprint(source) == main_file_before
        with corpus_copy(source) as snapshot:
            captured = snapshot.evaluation_snapshot_fingerprint
            assert _text(snapshot) == "Committed evidence in the WAL."
            assert captured == fingerprint(snapshot.database_path)
            assert captured != main_file_before
            _set_text(writer, "New evidence committed after evaluation started.")
            assert _text(snapshot) == "Committed evidence in the WAL."
            assert snapshot.evaluation_snapshot_fingerprint == captured
            with corpus_copy(source) as later:
                assert _text(later) == "New evidence committed after evaluation started."
                assert later.evaluation_snapshot_fingerprint != captured


def test_snapshot_identity_is_stable_across_copies_and_before_evaluation_mutations(tmp_path):
    source = tmp_path / "source.db"
    _source_database(source)
    source_fingerprint = fingerprint(source)
    with corpus_copy(source) as first:
        captured = first.evaluation_snapshot_fingerprint
        path = Path(first.database_path)
        _set_text(first.connection, "Disposable evaluation mutation.")
        assert first.evaluation_snapshot_fingerprint == captured
        assert fingerprint(path) != captured
    assert not path.exists()
    assert fingerprint(source) == source_fingerprint
    with corpus_copy(source) as second:
        assert second.evaluation_snapshot_fingerprint == captured
        assert _text(second) == "Initial canonical evidence."


def _report(case_id):
    return {"report_version": 1, "generated_at": "2026-01-01T00:00:00+00:00",
            "case_count": 1, "passed_count": 1, "failed_count": 0,
            "cases": [{"case_id": case_id, "category": "retrieval", "mode": "offline_fixture",
                       "passed": True, "metrics": {}, "failure_reasons": []}],
            "summary": {}}


def test_explicit_report_directory_cannot_supply_stale_result_when_subprocess_writes_nothing(tmp_path, monkeypatch):
    old = _report("previous-request")
    report_path = tmp_path / "latest.json"
    report_path.write_text(json.dumps(old), encoding="utf-8")
    used_directories = []

    def stopped(command, **kwargs):
        directory = Path(command[command.index("--output-dir") + 1])
        used_directories.append(directory)
        assert directory != tmp_path and not (directory / "latest.json").exists()
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(evaluation.subprocess, "run", stopped)
    with pytest.raises(RuntimeError, match="Evaluation process failed"):
        evaluation.run_evaluation(SimpleNamespace(database_path="unused.db"), output_dir=tmp_path)
    assert json.loads(report_path.read_text(encoding="utf-8")) == old
    assert not used_directories[0].exists()


def test_explicit_report_directory_receives_only_fresh_validated_artifacts(tmp_path, monkeypatch):
    (tmp_path / "latest.json").write_text(json.dumps(_report("stale")), encoding="utf-8")
    (tmp_path / "latest.md").write_text("stale markdown", encoding="utf-8")
    fresh = _report("this-request")

    def completed(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        assert output != tmp_path
        (output / "latest.json").write_text(json.dumps(fresh), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(evaluation.subprocess, "run", completed)
    actual = evaluation.run_evaluation(SimpleNamespace(database_path="unused.db"), output_dir=tmp_path)
    assert actual == fresh
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8")) == fresh
    assert "Cases: 1 | Passed: 1 | Failed: 0" in (tmp_path / "latest.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("return_code,payload", [(2, _report("failed-process")),
                                                 (0, {**_report("incomplete"), "case_count": 7})])
def test_invalid_or_failed_subprocess_does_not_publish_over_old_reports(tmp_path, monkeypatch, return_code, payload):
    old = _report("previous-request")
    (tmp_path / "latest.json").write_text(json.dumps(old), encoding="utf-8")

    def completed(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        (output / "latest.json").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, return_code)

    monkeypatch.setattr(evaluation.subprocess, "run", completed)
    with pytest.raises(RuntimeError, match="Evaluation process"):
        evaluation.run_evaluation(SimpleNamespace(database_path="unused.db"), output_dir=tmp_path)
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8")) == old
