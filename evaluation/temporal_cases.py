"""Offline execution and labelled metrics for document lifecycle decisions.

Fixture recommendations are invented test-only evidence. Corpus labels are
partial, inspected documentary observations, not a claim of external currency.
No model judge, embedding service, or clock-dependent current date is required.
"""

from __future__ import annotations

from contextlib import contextmanager
import re
from unittest.mock import patch

from data_models import DocumentSource, KnowledgeArtifact
from database import KnowledgeStore
from evaluation.metrics import evaluate_answer


@contextmanager
def temporal_fixture_store(dataset, document_ids):
    """Build only the chosen synthetic family members in a disposable store."""
    requested = set(document_ids)
    documents = {item["document_id"]: item for item in dataset["documents"]}
    if requested - set(documents):
        raise ValueError("Unknown temporal fixture document IDs")
    if any(not doc_id.startswith("DOC9") for doc_id in requested):
        raise ValueError("Temporal test-only sources must use the DOC9xx namespace")
    with KnowledgeStore(":memory:") as store:
        for doc_id in document_ids:
            item = documents[doc_id]
            store.upsert_document_sources([DocumentSource(
                doc_id, item["title"], item["source_url"], None,
                doc_id + ".pdf", "pdf", year=item["year"], agency=item["agency"],
            )])
            store.ingest_chunk(KnowledgeArtifact(
                doc_id, item["title"], item["page_number"], item["original_text_chunk"],
                source_url=item["source_url"],
            ), [1.0, 0.0, 0.0])
        yield store


def _edge_key(edge, source_id=None):
    return (edge.get("source_id", edge.get("document_id", source_id)),
            edge.get("target_id"), edge.get("type"))


def _index_edges(index):
    return [{**edge, "source_id": doc_id}
            for doc_id, document in index.items() for edge in document.get("relations", [])]


def _canonical_span(store, document_id, page_number, exact_span):
    if not isinstance(exact_span, str) or not exact_span:
        return False
    chunks = store.connection.execute(
        "SELECT original_text_chunk FROM knowledge_artifacts WHERE document_id=? AND page_number=?",
        (document_id, page_number),
    ).fetchall()
    return any(exact_span in item[0] for item in chunks)


def evaluate_temporal_answer(answer, sources, selection, store, index):
    """Check rendered citation ownership and explicit revision assertions.

    This checks deterministic document-to-document replacement statements and
    canonical relationship spans. It does not claim arbitrary natural-language
    entailment or scientific correctness of the source's recommendation.
    """
    supplied = selection.get("evidence", selection.get("artifacts", []))
    # Engine scope/uncertainty disclosures describe the selection procedure, not
    # source findings. Exclude only their exact known lines from claim coverage;
    # arbitrary uncited prose, including generated replacement claims, remains.
    procedural_lines = {"- " + text for text in selection.get("uncertainties", [])}
    evidence_answer = "\n".join(line for line in answer.splitlines() if line.strip() not in procedural_lines)
    citation_checks = evaluate_answer(evidence_answer, supplied, sources, expect_claims=bool(supplied))
    canonical_keys = set()
    for edge in _index_edges(index):
        source_id, target_id, relation_type = _edge_key(edge)
        evidence = edge.get("evidence") or {}
        if (edge.get("source") == "explicit" and target_id in index
                and evidence.get("document_id") == source_id
                and _canonical_span(store, source_id, evidence.get("page_number"), evidence.get("exact_span"))):
            canonical_keys.add((source_id, target_id, relation_type))
    invalid_relationships = []
    for edge in selection.get("relationships", []):
        source_id, _, _ = _edge_key(edge)
        evidence = edge.get("evidence") or {}
        if (_edge_key(edge) not in canonical_keys or edge.get("source") != "explicit"
                or evidence.get("document_id") != source_id
                or not _canonical_span(store, source_id, evidence.get("page_number"), evidence.get("exact_span"))):
            invalid_relationships.append(_edge_key(edge))
    # The production renderer owns these directional assertions. A forged DOC
    # pair must be detected even if it has a syntactically valid citation.
    claims = re.finditer(
        r"\b(DOC\d{3,})\s+(?:explicitly\s+)?(supersedes|replaces|withdraws)\s+(DOC\d{3,})\b",
        answer, re.IGNORECASE,
    )
    unsupported = []
    for match in claims:
        identity = (match[1].upper(), match[3].upper(), match[2].lower())
        if identity not in canonical_keys:
            unsupported.append(identity)
    failures = [str(failure) for failure in citation_checks.get("failures", [])]
    if invalid_relationships:
        failures.append("Selected lifecycle relationship lacks canonical explicit evidence")
    if unsupported:
        failures.append("Rendered answer asserts an unsupported document replacement")
    return {
        "citation_validity_rate": citation_checks.get("citation_validity_rate"),
        "invalid_citation_count": citation_checks.get("invalid_citation_count", 0),
        "unsupported_supersession_claim_count": len(unsupported),
        "invalid_relationship_evidence_count": len(invalid_relationships),
        "failures": failures,
        "unsupported_claims": unsupported,
        "invalid_relationships": invalid_relationships,
    }


def _execute_case(store, case, dataset):
    import document_lifecycle
    import temporal

    failures = []
    metrics = {}
    details = {}
    for judgment in case.get("judgments", []):
        if not _canonical_span(store, judgment["document_id"], judgment["page_number"], judgment["exact_span"]):
            failures.append("Inspected temporal label no longer resolves: " + judgment["document_id"])
    index = document_lifecycle.get_lifecycles(store)
    all_edges = _index_edges(index)
    edge_keys = {_edge_key(edge) for edge in all_edges}
    relationship_checks = []
    for expected in case.get("expected_relationships", []):
        correct = _edge_key(expected) in edge_keys
        relationship_checks.append(correct)
        if not correct:
            failures.append("Expected lifecycle relationship missing: " + str(_edge_key(expected)))
    for forbidden in case.get("forbidden_relationships", []):
        correct = _edge_key(forbidden) not in edge_keys
        relationship_checks.append(correct)
        if not correct:
            failures.append("Unsupported lifecycle relationship extracted: " + str(_edge_key(forbidden)))
    if relationship_checks:
        metrics["version_relationship_accuracy"] = sum(relationship_checks) / len(relationship_checks)

    metadata_checks = []
    observed_metadata = {}
    invalid_metadata_evidence = 0
    for expected in case.get("metadata_expectations", []):
        doc_id = expected["document_id"]
        actual = index.get(doc_id, {})
        observed_metadata[doc_id] = {field: actual.get(field) for field in expected if field != "document_id"}
        for field, value in expected.items():
            if field == "document_id":
                continue
            correct = doc_id in index and actual.get(field) == value
            metadata_checks.append(correct)
            if not correct:
                failures.append(f"{doc_id}.{field}: expected {value!r}, observed {actual.get(field)!r}")
            signal = (actual.get("dates", {}).get(field) if field.endswith("_date")
                      else actual.get(field + "_evidence"))
            if isinstance(signal, dict) and signal.get("source") == "explicit":
                evidence = signal.get("evidence") or {}
                valid_evidence = (evidence.get("document_id") == doc_id and _canonical_span(
                    store, doc_id, evidence.get("page_number"), evidence.get("exact_span"),
                ))
                invalid_metadata_evidence += not valid_evidence
    if metadata_checks:
        metrics["lifecycle_metadata_accuracy"] = sum(metadata_checks) / len(metadata_checks)
        metrics["invalid_lifecycle_metadata_evidence_count"] = invalid_metadata_evidence
        if invalid_metadata_evidence:
            failures.append("Explicit lifecycle metadata has missing or invalid canonical evidence")
        details["observed_metadata"] = observed_metadata

    family_checks = []
    for first, second in case.get("expected_distinct_families", []):
        correct = bool(index.get(first, {}).get("family_id") and index.get(second, {}).get("family_id")
                       and index[first]["family_id"] != index[second]["family_id"])
        family_checks.append(correct)
        if not correct:
            failures.append("Different issuing authorities were merged into one guidance family")
    if family_checks:
        metrics["family_isolation_accuracy"] = sum(family_checks) / len(family_checks)

    if "question" in case:
        question = case["question"]
        as_of = case.get("as_of", dataset["as_of"])
        intent = temporal.detect_temporal_intent(question)
        if "expected_intent" in case:
            metrics["temporal_intent_accuracy"] = int(intent.mode == case["expected_intent"])
            if not metrics["temporal_intent_accuracy"]:
                failures.append(f"Expected {case['expected_intent']} intent, observed {intent.mode}")
        if "conversation" in case:
            import main
            previous = case["conversation"]
            old_artifacts = temporal._doc_artifacts(store, [previous["previous_document_id"]])
            history = [
                {"role": "user", "content": previous["previous_question"]},
                {"role": "assistant", "content": previous["previous_answer"], "sources": old_artifacts,
                 "context": {"active_subject": previous["active_subject"], "relation": "NEW_TOPIC",
                             "standalone_query": previous["previous_question"]}},
            ]
            captured = {}
            original_select = temporal.select_temporal_evidence

            def record_selection(*args, **kwargs):
                kwargs["as_of"] = as_of
                result = original_select(*args, **kwargs)
                captured["selection"] = result
                return result

            diagnostics = {}
            with patch.object(temporal, "select_temporal_evidence", record_selection):
                answer, preamble, sources = main.ask_chatbot_with_context(
                    question, store, history=history, diagnostics=diagnostics,
                )
            selection = captured.get("selection", {})
            details["context_diagnostics"] = diagnostics
            metrics["temporal_followup_context_accuracy"] = int(bool(
                selection and diagnostics.get("uses_history")
                and previous["active_subject"].casefold() in selection.get("question", "").casefold()
                and selection.get("intent") == "current"
            ))
            if not metrics["temporal_followup_context_accuracy"]:
                failures.append("Historical-to-current conversation did not retain subject and change temporal scope")
        else:
            selection = temporal.select_temporal_evidence(question, store, top_k=case.get("k", 5), as_of=as_of)
            answer, preamble, sources = temporal.render_temporal_answer(selection, store)
        selected = selection.get("selected_document_ids", [])
        first = selected[0] if selected else None
        if "expected_first_document_id" in case:
            expected_first = case["expected_first_document_id"]
            correct = first == expected_first
            metric = {"current": "correct_current_document_rate", "historical": "historical_query_accuracy",
                      "latest": "latest_report_accuracy"}.get(case["expected_intent"], "preferred_document_accuracy")
            metrics[metric] = int(correct)
            if not correct:
                failures.append(f"Expected preferred {expected_first}, observed {first}")
        forbidden_preference = first in set(case.get("forbidden_preferred_document_ids", []))
        if "forbidden_preferred_document_ids" in case:
            metrics["inapplicable_document_preference_errors"] = int(forbidden_preference)
            if forbidden_preference:
                failures.append("A superseded, draft, future-effective, withdrawn, or different-kind source was preferred")
        missing = set(case.get("expected_document_ids", [])) - set(selected)
        if "expected_document_ids" in case:
            metrics["required_temporal_sources_recall"] = 1 - len(missing) / len(case["expected_document_ids"])
            if missing:
                failures.append("Missing required temporal source(s): " + ", ".join(sorted(missing)))
        if case.get("expect_uncertainty"):
            metrics["uncertain_currentness_disclosed"] = int(bool(selection.get("uncertainties")))
            if not metrics["uncertain_currentness_disclosed"]:
                failures.append("Ambiguous family/currentness was not disclosed in selection uncertainties")
        expected_order = case.get("expected_relative_order")
        if expected_order:
            actual_order = [doc_id for doc_id in selected if doc_id in expected_order]
            metrics["historical_sequence_accuracy"] = int(actual_order == expected_order)
            if not metrics["historical_sequence_accuracy"]:
                failures.append("Version comparison did not preserve chronological source order")
        answer_checks = evaluate_temporal_answer(answer, sources, selection, store, index)
        failures.extend(answer_checks["failures"])
        metrics.update({name: value for name, value in answer_checks.items() if type(value) in (float, int)})
        details.update({"question": question, "as_of": as_of, "selected_document_ids": selected,
                        "decisions": selection.get("decisions", []), "uncertainties": selection.get("uncertainties", []),
                        "answer": answer, "preamble": preamble, "relationships": selection.get("relationships", []),
                        "citation_source_ids": list(dict.fromkeys(source.document_id for source in sources))})
    else:
        relevant_ids = {item["source_id"] for item in case.get("expected_relationships", [])}
        details["relationships"] = [edge for edge in all_edges if edge["source_id"] in relevant_ids]
    return {"case_id": case["case_id"], "category": "temporal",
            "mode": "offline_corpus" if case.get("mode") == "corpus" else "offline_fixture",
            "metrics": metrics, "passed": not failures, "failure_reasons": failures, "details": details}


def run_temporal_cases(store, dataset):
    """Run declarative fixture and real-corpus cases, emitting standard rows."""
    rows = []
    for case in dataset["cases"]:
        try:
            if case.get("mode") == "corpus":
                result = _execute_case(store, case, dataset)
            else:
                with temporal_fixture_store(dataset, case["fixture_document_ids"]) as fixtures:
                    result = _execute_case(fixtures, case, dataset)
        except Exception as error:
            result = {"case_id": case["case_id"], "category": "temporal",
                      "mode": "offline_corpus" if case.get("mode") == "corpus" else "offline_fixture",
                      "metrics": {}, "passed": False,
                      "failure_reasons": [f"Temporal execution failed: {type(error).__name__}: {error}"], "details": {}}
        rows.append(result)
    return rows
