"""Final-chunk retrieval evaluation with inspected, partial evidence labels."""

from contextlib import contextmanager
import re
import time

from evaluation.dataset import DATASET_DIR, fixture_store, load_dataset
from evaluation.metrics import retrieval_metrics


def evidence_metrics(artifacts, case):
    """Measure supplied final evidence, not a larger hidden candidate pool.

    Exact span metrics use only inspected positives. Document ranking metrics
    deduplicate document IDs within these final chunks, with K as the budget.
    The duplicate audit uses normalized text and five-token shingle Jaccard >=.9,
    preserving changed numbers/negation. It imposes no document diversity quota.
    """
    k = case.get("k", 5)
    ids = list(dict.fromkeys(item.document_id for item in artifacts))
    metrics = retrieval_metrics(ids, case.get("relevance", {}), k)
    signatures = []
    duplicates = 0
    for artifact in artifacts:
        words = re.findall(r"\w+", artifact.original_text_chunk.casefold())
        normalized = " ".join(words)
        shingles = {tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))}
        distinctions = (tuple(re.findall(r"\d+(?:[.,]\d+)*", artifact.original_text_chunk)),
                        tuple(word for word in words if word in {"no", "not", "never", "without"}))
        duplicate = any(normalized == text or (shingles and previous and
                        distinctions == prior_distinctions and
                        len(shingles & previous) / len(shingles | previous) >= .9)
                        for text, previous, prior_distinctions in signatures)
        duplicates += bool(duplicate)
        signatures.append((normalized, shingles, distinctions))
    metrics.update({"final_evidence_count": len(artifacts), "unique_document_count": len(ids),
                    "duplicate_evidence_rate": duplicates / len(artifacts) if artifacts else 0.0})
    judgments = case.get("judgments", [])
    matched = [any(item.document_id == judgment["document_id"]
                   and item.page_number == judgment["page_number"]
                   and judgment["exact_span"] in item.original_text_chunk for item in artifacts)
               for judgment in judgments]
    if judgments:
        metrics["exact_span_hit_rate"] = sum(matched) / len(matched)
        quantitative = [hit for judgment, hit in zip(judgments, matched) if judgment.get("quantitative")]
        if quantitative:
            metrics["quantitative_evidence_hit_rate"] = int(any(quantitative))
    if "expected_first_document_id" in case:
        metrics["exact_document_hit_rate"] = int(bool(artifacts) and artifacts[0].document_id == case["expected_first_document_id"])
    if "expected_document_ids" in case:
        required = set(case["expected_document_ids"])
        metrics["required_document_recall"] = len(required & set(ids)) / len(required) if required else None
    if "forbidden_document_ids" in case:
        wrong = sum(item.document_id in case["forbidden_document_ids"] for item in artifacts)
        metrics["wrong_topic_retrieval_rate"] = wrong / len(artifacts) if artifacts else 0.0
    return metrics


def validate_judgments(store, case):
    errors = []
    for item in case.get("judgments", []):
        rows = store.connection.execute(
            "SELECT original_text_chunk FROM knowledge_artifacts WHERE document_id=? AND page_number=?",
            (item["document_id"], item["page_number"]),
        ).fetchall()
        if not item.get("exact_span") or not any(item["exact_span"] in row[0] for row in rows):
            errors.append("Inspected relevance span no longer resolves: " + item["document_id"] + " p" + item["page_number"])
    return errors


def canonical_errors(store, artifacts):
    errors = []
    for item in artifacts:
        rows = store.connection.execute(
            "SELECT artifact_id FROM knowledge_artifacts WHERE document_id=? AND page_number=? AND original_text_chunk=?",
            (item.document_id, item.page_number, item.original_text_chunk),
        ).fetchall()
        if not rows or item not in store._artifacts_by_ranked_ids([row[0] for row in rows]):
            errors.append("Retrieved evidence differs from canonical text/provenance: " + item.document_id)
    return errors


@contextmanager
def quality_fixture_store(case):
    with fixture_store() as store:
        for doc_id in case.get("duplicate_document_ids", []):
            rows = store.connection.execute(
                "SELECT artifact_id FROM knowledge_artifacts WHERE document_id=? ORDER BY artifact_id LIMIT 1", (doc_id,),
            ).fetchall()
            if not rows:
                raise ValueError("Unknown duplicate fixture document: " + doc_id)
            store.ingest_chunk(store._artifacts_by_ranked_ids([rows[0][0]])[0], [1.0, 0.0, 0.0])
        yield store


def _run_case(store, case):
    from retrieval import retrieve_evidence
    failures = validate_judgments(store, case)
    trace = {}
    started = time.perf_counter()
    artifacts = retrieve_evidence(store, case["question"], query_embedding=case.get("query_vector"),
                                  top_k=case.get("k", 5), mode="hybrid_rerank", diagnostics=trace)
    elapsed = (time.perf_counter() - started) * 1000
    metrics = evidence_metrics(artifacts, case)
    failures += canonical_errors(store, artifacts)
    if len(artifacts) > case.get("k", 5):
        failures.append("Final evidence exceeds the requested K")
    warnings = []
    findings = []
    if metrics.get("exact_document_hit_rate") == 0:
        findings.append("Expected exact/remembered document did not rank first")
    if metrics.get("exact_span_hit_rate", 1) < 1:
        findings.append("At least one inspected positive span was absent from final evidence")
    if metrics.get("required_document_recall", 1) < 1:
        findings.append("Required complementary document missing from final evidence")
    if metrics.get("wrong_topic_retrieval_rate", 0):
        findings.append("Known wrong-topic evidence entered the final K")
    if metrics["recall_at_k"] is not None and metrics["recall_at_k"] < case.get("minimum_recall", 0):
        findings.append("Labelled fixture recall below its declared requirement")
    if metrics["duplicate_evidence_rate"] > case.get("maximum_duplicate_evidence_rate", 1):
        findings.append("Duplicate evidence exceeds the declared limit")
    # Partial real labels expose development gaps without pretending other
    # documents are irrelevant. Controlled fixture failures are hard invariants.
    (failures if case["kind"] == "fixture" else warnings).extend(findings)
    return {"case_id": case["case_id"], "category": "retrieval_quality",
            "mode": "offline_fixture" if case["kind"] == "fixture" else "offline_corpus",
            "metrics": metrics, "passed": not failures, "failure_reasons": failures, "warnings": warnings,
            "details": {"question": case["question"], "query_category": case["category"], "k": case.get("k", 5),
                        "ranking_unit": "final canonical evidence chunks", "label_scope": "controlled_fixture" if case["kind"] == "fixture" else "partial",
                        "document_metric_unit": "unique document IDs within final chunks; precision denominator K",
                        "elapsed_ms": round(elapsed, 3), "retrieval_trace": trace,
                        "retrieved_evidence": [{"document_id": a.document_id, "page_number": a.page_number,
                                                "title": a.title} for a in artifacts]}}


def _delegate_case(store, case):
    if case["kind"] == "temporal":
        from evaluation.temporal_cases import run_temporal_cases
        dataset = load_dataset(DATASET_DIR / "temporal.json")
        dataset["cases"] = [row for row in dataset["cases"] if row["case_id"] == case["delegate_case_id"]]
        result = run_temporal_cases(store, dataset)[0]
    else:
        from evaluation.scenarios import run_conversation_cases
        dataset = load_dataset(DATASET_DIR / "conversations.json")
        selected = [row for row in dataset["cases"] if row["case_id"] == case["delegate_case_id"]]
        result = run_conversation_cases(selected)[0]
    result = {**result, "case_id": case["case_id"], "category": "retrieval_quality",
              "mode": "offline_compatibility", "details": {**result["details"], "delegate_case_id": case["delegate_case_id"],
              "scope": "Existing production context/lifecycle invariant rerun against the new retrieval; not an independent relevance benchmark"}}
    result["metrics"] = {**result["metrics"], "compatibility_pass_rate": int(result["passed"])}
    return result


def run_retrieval_quality_cases(store, dataset):
    rows = []
    for case in dataset["cases"]:
        try:
            if case["kind"] == "corpus":
                result = _run_case(store, case)
            elif case["kind"] == "fixture":
                with quality_fixture_store(case) as fixtures:
                    result = _run_case(fixtures, case)
            else:
                result = _delegate_case(store, case)
        except Exception as error:
            result = {"case_id": case["case_id"], "category": "retrieval_quality", "mode": "offline_corpus" if case["kind"] == "corpus" else "offline_fixture",
                      "metrics": {}, "passed": False, "failure_reasons": ["Retrieval quality execution failed: " + type(error).__name__], "details": {}}
        rows.append(result)
    return rows
