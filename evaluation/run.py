"""Run offline V3 evaluation: python -m evaluation.run (API calls require --live)."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import subprocess
import time
from unittest.mock import patch

from config import SETTINGS, V3_ROOT
from evaluation.dataset import DATASET_DIR, corpus_copy, fingerprint, fixture_store, load_dataset
from evaluation.metrics import evaluate_answer, evaluate_claims, retrieval_metrics
from evaluation.reporting import (REPORT_VERSION, baseline_from, compare_baseline,
                                  summarize, write_reports, render_summary)
from validator import validate_render_and_collect_sources, VALIDATION_FAILED_MESSAGE


@contextmanager
def offline_providers():
    attempts = []
    def blocked(*args, **kwargs):
        attempts.append("provider_request")
        raise RuntimeError("Offline evaluation forbids provider requests")
    with patch("api_clients._client", blocked):
        yield attempts


def row(case, category, mode, metrics, failures, details=None, warnings=None):
    return {"case_id": case["case_id"], "category": category, "mode": mode,
            "metrics": metrics, "passed": not failures, "failure_reasons": failures,
            "details": details or {}, "warnings": warnings or []}


def run_retrieval_cases(store, dataset, *, top_k=5, live=False):
    import main
    rows = []
    for case in dataset["cases"]:
        failures, warnings = [], []
        k = case.get("k", top_k) if dataset.get("synthetic") else top_k
        method = "semantic" if live else case.get("retrieval_method", "keyword")
        for judgment in case.get("judgments", []):
            candidates = store.connection.execute(
                "SELECT original_text_chunk FROM knowledge_artifacts WHERE document_id=? AND page_number=?",
                (judgment["document_id"], judgment["page_number"]),
            ).fetchall()
            if not judgment.get("exact_span") or not any(judgment["exact_span"] in item[0] for item in candidates):
                failures.append(f"Relevance justification no longer resolves: {judgment['document_id']} p{judgment['page_number']}")
        started = time.perf_counter()
        try:
            if method == "document":
                artifacts = store.retrieve_document_matches(case["question"], top_k=k)
            elif method == "semantic":
                artifacts = main.search_corpus(case["question"], store, top_k=k * 4) if live else store.retrieve(case["query_vector"], k * 4)
            else:
                artifacts = store.retrieve(None, k * 4, method="keyword", query_text=case["question"])
        except Exception as error:
            artifacts = []
            failures.append(f"Retrieval execution failed: {type(error).__name__}")
        ids = list(dict.fromkeys(a.document_id for a in artifacts))[:k]
        metrics = retrieval_metrics(ids, case.get("relevance", {}), k)
        if set(ids) & set(case.get("forbidden_document_ids", [])):
            failures.append("Known wrong-topic source entered the evaluated top K")
        if case.get("expect_empty") and ids:
            failures.append("Expected no matching evidence")
        minimum = case.get("minimum_recall")
        if minimum is not None and (metrics["recall_at_k"] is None or metrics["recall_at_k"] < minimum):
            failures.append("Labelled recall below this fixture's explicit threshold")
        if metrics["hit_at_k"] == 0:
            warnings.append("No labelled relevant document in top K; inspect retrieval drift.")
        rows.append(row(case, "retrieval", "live_semantic" if live else
                        ("offline_fixture" if dataset.get("synthetic") else "offline_corpus"), metrics, failures,
                        {"question": case["question"], "retrieval_method": method, "k": k,
                         "candidate_k": k * 4, "retrieved_document_ids": ids,
                         "relevance": case.get("relevance", {}), "label_scope": dataset.get("label_scope"),
                         "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}, warnings))
    return rows


def run_provenance_cases(store, cases):
    rows = []
    canonical = store.retrieve([1., 0., 0.], store.artifact_count)
    by_id = {item.document_id: item for item in canonical}
    for case in cases:
        artifacts = {handle: by_id[doc] for handle, doc in case["artifacts"].items()}
        for handle, changes in case.get("artifact_overrides", {}).items():
            artifacts[handle] = replace(artifacts[handle], **changes)
        payload = case["payload"]
        claim_metrics = evaluate_claims(payload, artifacts, canonical_artifacts=canonical)
        answer, sources = validate_render_and_collect_sources(json.dumps(payload), artifacts)
        if "answer_override" in case:
            answer = case["answer_override"]
        answer_metrics = evaluate_answer(answer, list(artifacts.values()), sources,
                                         expect_claims=case.get("expect_claims", True))
        # Adversarial fixtures pass when the named violation is detected; their
        # expected-invalid rates must not be presented as production accuracy.
        metrics = {**{f"claims_{key}": value for key, value in claim_metrics.items()
                      if type(value) in (int, float) or value is None},
                   **{key: value for key, value in answer_metrics.items()
                      if type(value) in (int, float) or value is None}}
        failures = []
        for metric, expected in case.get("expect_metrics", {}).items():
            if metrics.get(metric) != expected:
                failures.append(f"{metric}: expected {expected}, observed {metrics.get(metric)}")
        if case.get("expect_validator_rejection") and answer != VALIDATION_FAILED_MESSAGE:
            failures.append("Production validator did not reject the invalid evidence envelope")
        rows.append(row(case, "provenance", "offline_adversarial" if case.get("adversarial") else "offline_fixture",
                        metrics, failures, {"answer": answer, "claim_checks": claim_metrics,
                                            "citation_checks": answer_metrics,
                                            "production_validator_rejected": answer == VALIDATION_FAILED_MESSAGE,
                                            "semantic_support": "not_assessed"}))
    return rows


def run_live_answers(store, cases, model, top_k):
    """Optional provider generation; expensive/nondeterministic results stay separate."""
    import main
    original = main.call_llm
    rows = []
    for case in cases:
        captured = {}
        def record(system, prompt, artifacts, **kwargs):
            captured["artifacts"] = artifacts
            text = original(system, prompt, artifacts, **kwargs)
            captured["payload"] = json.loads(text)
            return text
        try:
            with patch.object(main, "call_llm", record):
                answer, preamble, sources = main.ask_chatbot_with_context(case["question"], store, model=model, top_k=top_k)
            retrieved = list(captured.get("artifacts", {}).values())
            # Direct canonical answer paths have no synthesis request to capture.
            checks = evaluate_answer(answer, retrieved or sources, sources,
                                     expect_claims=case.get("expect_claims", True))
            raw_checks = evaluate_claims(captured["payload"], captured["artifacts"]) if "payload" in captured else None
            failures = [str(item) for item in checks.get("failures", [])]
            metrics = {key: value for key, value in checks.items() if type(value) in (int, float) or value is None}
            if raw_checks is not None:
                metrics.update({f"raw_{key}": value for key, value in raw_checks.items()
                                if type(value) in (int, float) or value is None})
                failures.extend(f"Generated claim: {item}" for item in raw_checks.get("failures", []))
            rows.append(row(case, "grounded_answer", "live_generation", metrics, failures,
                            {"answer": answer, "preamble": preamble, "citation_checks": checks,
                             "raw_claim_checks": raw_checks, "retrieval_capture": "synthesis" if retrieved else "direct_canonical"}))
        except Exception as error:
            rows.append(row(case, "grounded_answer", "live_generation", {},
                            [f"Provider execution failed: {type(error).__name__}"], {"question": case["question"]}))
    return rows


def run_suite(*, corpus=SETTINGS.storage.database_path, retrieval_cases=DATASET_DIR / "retrieval.json",
              top_k=5, live=False, model=None, case_ids=None):
    from evaluation.scenarios import run_conversation_cases, run_wiki_cases
    from evaluation.temporal_cases import run_temporal_cases
    from evaluation.retrieval_quality import run_retrieval_quality_cases
    paths = {"retrieval": Path(retrieval_cases), **{name: DATASET_DIR / (name + ".json")
             for name in ("retrieval_fixtures", "provenance", "conversations", "wiki", "temporal", "retrieval_quality")}}
    datasets = {name: load_dataset(path) for name, path in paths.items()}
    selected = set(case_ids or [])
    if selected:
        known = {case["case_id"] for dataset in datasets.values() for case in dataset["cases"]}
        if selected - known:
            raise ValueError("Unknown selected case IDs: " + ", ".join(sorted(selected - known)))
        for dataset in datasets.values():
            dataset["cases"] = [case for case in dataset["cases"] if case["case_id"] in selected]
    started = time.perf_counter()
    rows = []
    with corpus_copy(corpus) as store:
        corpus_fingerprint = store.evaluation_snapshot_fingerprint
        with offline_providers() as attempts:
            rows += run_retrieval_cases(store, datasets["retrieval"], top_k=top_k)
            with fixture_store() as fixtures:
                rows += run_retrieval_cases(fixtures, datasets["retrieval_fixtures"], top_k=top_k)
                rows += run_provenance_cases(fixtures, datasets["provenance"]["cases"])
            rows += run_wiki_cases(store, datasets["wiki"]["cases"])
            rows += run_conversation_cases(datasets["conversations"]["cases"])
            rows += run_temporal_cases(store, datasets["temporal"])
            rows += run_retrieval_quality_cases(store, datasets["retrieval_quality"])
        if attempts:
            rows.append(row({"case_id": "OFFLINE-NETWORK"}, "infrastructure", "offline_fixture", {},
                            [f"Offline suite attempted {len(attempts)} provider calls"]))
        if live:
            rows += run_retrieval_cases(store, datasets["retrieval"], top_k=top_k, live=True)
            rows += run_live_answers(store, datasets["retrieval"]["cases"], model, top_k)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=V3_ROOT, capture_output=True, text=True)
    report = {"report_version": REPORT_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
              "configuration": {"top_k": top_k, "live": live, "model": (model or SETTINGS.models.llm_model) if live else None,
                                "seed": 0, "case_ids": sorted(selected), "metrics_version": 1,
                                "retrieval": {key: getattr(SETTINGS.retrieval, key)
                                              for key in SETTINGS.retrieval.__dataclass_fields__}},
              "environment": {"python": platform.python_version(), "git_commit": git.stdout.strip(),
                              "elapsed_seconds": round(time.perf_counter() - started, 3)},
              "fingerprints": {**{name: fingerprint(path) for name, path in paths.items()},
                               "fixture_corpus": fingerprint(DATASET_DIR / "fixture_corpus.json"), "corpus": corpus_fingerprint},
              "case_count": len(rows), "passed_count": sum(item["passed"] for item in rows),
              "failed_count": sum(not item["passed"] for item in rows), "cases": rows, "summary": summarize(rows)}
    return report


def main():
    logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=SETTINGS.storage.database_path)
    parser.add_argument("--retrieval-cases", type=Path, default=DATASET_DIR / "retrieval.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "reports")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--live", action="store_true", help="Opt in to paid embedding/answer calls for selected corpus cases")
    parser.add_argument("--model", default=SETTINGS.models.llm_model)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--write-baseline", type=Path, help="Explicitly store this reviewed run as a baseline")
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    try:
        report = run_suite(corpus=args.corpus, retrieval_cases=args.retrieval_cases, top_k=args.top_k,
                           live=args.live, model=args.model, case_ids=args.case_ids)
        if args.baseline:
            report["baseline_comparison"] = compare_baseline(report, json.loads(args.baseline.read_text(encoding="utf-8")))
        write_reports(report, args.output_dir)
        if args.write_baseline:
            if report["failed_count"]:
                raise ValueError("Refusing to store a failing run as baseline")
            args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
            args.write_baseline.write_text(json.dumps(baseline_from(report), indent=2, allow_nan=False), encoding="utf-8")
        print(render_summary(report))
        comparison = report.get("baseline_comparison", {})
        regression = args.fail_on_regression and (comparison.get("regressions") or comparison.get("compatible") is False)
        return 1 if report["failed_count"] or regression else 0
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
