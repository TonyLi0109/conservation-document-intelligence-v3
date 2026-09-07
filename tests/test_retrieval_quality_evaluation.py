"""The evidence benchmark must expose misses, corruption and unavailable vectors."""

from copy import deepcopy
from dataclasses import replace
import json
import sqlite3
import subprocess
import sys

import pytest

from config import SETTINGS, V3_ROOT
from evaluation.dataset import DATASET_DIR, fixture_store, load_dataset
from evaluation.retrieval_ablation import (
    load_query_vectors, query_cache_key, run_variant, summarize_variants,
)
from evaluation.retrieval_quality import (
    canonical_errors, evidence_metrics, run_retrieval_quality_cases, validate_judgments,
)
from evaluation.reporting import baseline_from, compare_baseline


DATASET = load_dataset(DATASET_DIR / "retrieval_quality.json")
FIXTURES = [case for case in DATASET["cases"] if case["kind"] == "fixture"]


@pytest.fixture
def artifacts():
    with fixture_store() as store:
        rows = store.connection.execute("SELECT artifact_id FROM knowledge_artifacts ORDER BY artifact_id").fetchall()
        yield store, {item.document_id: item for item in store._artifacts_by_ranked_ids([row[0] for row in rows])}


def test_only_supplied_final_evidence_counts_toward_recall(artifacts):
    _, docs = artifacts
    case = {"k": 2, "relevance": {"DOC901": 2, "DOC908": 2}}
    actual = evidence_metrics([docs["DOC901"], docs["DOC901"]], case)
    assert actual["recall_at_k"] == .5
    assert actual["unique_document_count"] == 1
    assert actual["final_evidence_count"] == 2
    assert actual["duplicate_evidence_rate"] == .5


def test_quantitative_match_requires_exact_owner_page_and_span(artifacts):
    _, docs = artifacts
    carp = docs["DOC901"]
    case = {"k": 2, "relevance": {"DOC901": 2}, "judgments": [{
        "document_id": carp.document_id, "page_number": carp.page_number,
        "exact_span": carp.original_text_chunk, "quantitative": True,
    }]}
    for unrelated in [docs["DOC903"], replace(carp, page_number="999"),
                      replace(carp, original_text_chunk="A different study reports 60%.")]:
        result = evidence_metrics([unrelated], case)
        assert result["exact_span_hit_rate"] == 0
        assert result["quantitative_evidence_hit_rate"] == 0
    assert evidence_metrics([carp], case)["quantitative_evidence_hit_rate"] == 1


def test_unlabelled_wrong_topic_and_numeric_metrics_are_not_invented(artifacts):
    _, docs = artifacts
    metrics = evidence_metrics([docs["DOC903"]], {"k": 1, "relevance": {"DOC901": 2}})
    assert "wrong_topic_retrieval_rate" not in metrics
    assert "quantitative_evidence_hit_rate" not in metrics
    assert "exact_span_hit_rate" not in metrics


def test_duplicate_audit_keeps_changed_measurements_and_negation(artifacts):
    _, docs = artifacts
    shared = " ".join("word" + str(i) for i in range(120))
    texts = [shared + " The outcome improved by 60 percent.",
             shared + " The outcome improved by 20 percent.",
             shared + " The outcome did not improve by 60 percent."]
    selected = [replace(docs["DOC901"], original_text_chunk=text) for text in texts]
    assert evidence_metrics(selected, {"k": 3})["duplicate_evidence_rate"] == 0


def test_known_wrong_topic_and_corrupt_provenance_are_detected(artifacts):
    store, docs = artifacts
    metrics = evidence_metrics([docs["DOC901"], docs["DOC903"]], {
        "k": 2, "relevance": {"DOC901": 2}, "forbidden_document_ids": ["DOC903"],
    })
    assert metrics["wrong_topic_retrieval_rate"] == .5
    assert not canonical_errors(store, [docs["DOC901"]])
    assert canonical_errors(store, [replace(docs["DOC901"], source_url="https://example.invalid/wrong")])
    assert validate_judgments(store, {"judgments": [{"document_id": "DOC901", "page_number": "999", "exact_span": "Invented"}]})


@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case["case_id"])
def test_controlled_final_evidence_requirements(case):
    row = run_retrieval_quality_cases(None, {"cases": [case]})[0]
    assert row["mode"] == "offline_fixture"
    assert row["passed"], row["failure_reasons"]
    assert row["metrics"]["final_evidence_count"] <= case["k"]


def test_evaluator_fails_controlled_drift_but_records_partial_corpus_miss(monkeypatch, artifacts):
    import retrieval
    store, docs = artifacts
    monkeypatch.setattr(retrieval, "retrieve_evidence", lambda *args, **kwargs: [docs["DOC903"]])
    case = {"case_id": "adversarial-drift", "kind": "fixture", "category": "topic_drift",
            "question": "carp", "k": 1, "relevance": {"DOC901": 2}, "minimum_recall": 1,
            "forbidden_document_ids": ["DOC903"]}
    failed = run_retrieval_quality_cases(store, {"cases": [case]})[0]
    assert not failed["passed"] and failed["metrics"]["wrong_topic_retrieval_rate"] == 1
    partial = run_retrieval_quality_cases(store, {"cases": [{**case, "kind": "corpus"}]})[0]
    assert partial["passed"] and partial["warnings"]
    assert partial["metrics"]["recall_at_k"] == 0


@pytest.mark.integration
def test_inspected_real_labels_still_resolve_in_read_only_corpus():
    connection = sqlite3.connect(SETTINGS.storage.database_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for case in DATASET["cases"]:
            for judgment in case.get("judgments", []):
                rows = connection.execute(
                    "SELECT original_text_chunk FROM knowledge_artifacts WHERE document_id=? AND page_number=?",
                    (judgment["document_id"], judgment["page_number"]),
                ).fetchall()
                assert any(judgment["exact_span"] in row[0] for row in rows), case["case_id"]
    finally:
        connection.close()


def test_default_cache_miss_never_calls_embedding_provider(tmp_path, monkeypatch):
    import api_clients
    def forbidden(*args, **kwargs):
        pytest.fail("Offline ablation must not request embeddings")
    monkeypatch.setattr(api_clients, "generate_embeddings", forbidden)
    vectors, usage = load_query_vectors(["carp", "carp"], tmp_path / "absent.json")
    assert vectors == {} and usage["unavailable_query_count"] == 1
    assert usage["requested_query_count"] == 0
    assert not (tmp_path / "absent.json").exists()


def test_vector_cache_identity_separates_query_model_and_dimension():
    keys = {query_cache_key(*args) for args in [
        ("carp", "model-a", 3), ("Carp", "model-a", 3),
        ("carp", "model-b", 3), ("carp", "model-a", 4),
    ]}
    assert len(keys) == 4


def test_live_cache_flag_batches_unique_questions_and_reuses_results(tmp_path, monkeypatch):
    import api_clients
    calls = []
    vector = [1.0] + [0.0] * (SETTINGS.models.embedding_dimension - 1)
    def synthetic_provider(texts):
        calls.append(texts)
        return [vector for _ in texts]
    monkeypatch.setattr(api_clients, "generate_embeddings", synthetic_provider)
    path = tmp_path / "vectors.json"
    first, usage = load_query_vectors(["carp", "carp", "zebra"], path, allow_live=True)
    second, cached = load_query_vectors(["carp", "zebra"], path)
    assert calls == [["carp", "zebra"]]
    assert first == second == {"carp": vector, "zebra": vector}
    assert usage["requested_query_count"] == 2
    assert cached["cached_query_count"] == 2 and cached["requested_query_count"] == 0


@pytest.mark.parametrize("bad", ["malformed", [float("nan")], [0.0], [1.0]])
def test_invalid_or_mismatched_cache_entry_is_rejected(tmp_path, bad):
    model, dimension = SETTINGS.models.embedding_model, SETTINGS.models.embedding_dimension
    item = bad if isinstance(bad, str) else {"query": "carp", "model": model, "dimension": dimension, "embedding": bad}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 1, "queries": {query_cache_key("carp", model, dimension): item}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Cached query embedding"):
        load_query_vectors(["carp"], path)


@pytest.mark.parametrize("variant", ["dense", "hybrid", "hybrid_rerank"])
def test_real_vector_ablations_are_skipped_when_embeddings_are_unavailable(variant):
    result = run_variant(None, {"case_id": "missing-vector"}, variant)
    assert result["skipped"] and "metrics" not in result


def test_lexical_rerank_is_measured_without_claiming_dense_quality(artifacts):
    store, _ = artifacts
    result = run_variant(store, FIXTURES[-1], "lexical_rerank", repeats=1)
    assert not result["skipped"] and not result["dense_available"]
    assert not result["execution_errors"]
    assert result["warm_latency_ms"]["total_ms"] > 0
    assert result["metrics"]["required_document_recall"] == 1
    summary = summarize_variants([
        {**result, "dataset": "controlled"},
        {"dataset": "controlled", "variant": "dense", "skipped": True},
    ])
    assert summary["controlled/dense"]["metrics"] == {}
    assert summary["controlled/dense"]["measured_cases"] == 0
    assert summary["controlled/lexical_rerank"]["measured_cases"] == 1


@pytest.mark.parametrize("metric,before,after", [
    ("exact_span_hit_rate", 1, 0), ("quantitative_evidence_hit_rate", 1, 0),
    ("wrong_topic_retrieval_rate", 0, .5), ("duplicate_evidence_rate", 0, .5),
])
def test_quality_baseline_gates_use_correct_metric_direction(metric, before, after):
    previous = {"configuration": {}, "fingerprints": {}, "cases": [{
        "case_id": "quality", "category": "retrieval_quality", "mode": "offline_fixture",
        "passed": True, "metrics": {metric: before},
    }]}
    current = deepcopy(previous)
    current["cases"][0]["metrics"][metric] = after
    result = compare_baseline(current, baseline_from(previous))
    assert len(result["regressions"]) == 1 and metric in result["regressions"][0]


@pytest.mark.integration
def test_ablation_cli_writes_inspectable_reports_without_live_flag(tmp_path):
    process = subprocess.run([
        sys.executable, "-X", "utf8", "-m", "evaluation.retrieval_ablation",
        "--case", "retrieval_quality_exact_catalog_title", "--repeat", "1",
        "--embedding-cache", str(tmp_path / "nonexistent-cache.json"), "--output-dir", str(tmp_path),
    ], cwd=V3_ROOT, capture_output=True, text=True, encoding="utf-8")
    assert process.returncode == 0, process.stderr
    report = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert not report["configuration"]["live_embeddings_enabled"]
    assert report["configuration"]["query_embeddings"]["requested_query_count"] == 0
    assert all(row["skipped"] for row in report["cases"] if row["variant"] in {"dense", "hybrid", "hybrid_rerank"})
    assert "no dense vector" in (tmp_path / "latest.md").read_text(encoding="utf-8")
