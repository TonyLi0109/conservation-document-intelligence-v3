"""Offline execution of declarative Wiki and conversation regression scenarios.

Scripted responses exercise the production plumbing; their pass rates do not
measure a language model's ability to resolve intent or synthesize an answer.
The only stores mutated here are the caller's isolated Wiki evaluation copy and
an in-memory synthetic conversation corpus. No API client is invoked.
"""

from __future__ import annotations

import hashlib
import json
import re
from unittest.mock import patch

import chat_context
import conversation_history as threads
from database import KnowledgeStore
from evaluation.dataset import fixture_store
import main
import wiki_compiler as wiki


WIKI_FIELDS = {"concept_title", "summary", "important_facts", "related_entities", "supporting_evidence"}


def _forbid_provider(*args, **kwargs):
    raise AssertionError("A paid provider was called by an offline evaluation")


def _row(case: dict, category: str, mode: str, metrics: dict,
         failures: list[str], details: dict | None = None) -> dict:
    return {"case_id": case["case_id"], "category": category, "mode": mode,
            "metrics": metrics, "passed": not failures, "failure_reasons": failures,
            "details": details or {}}


def evaluate_wiki(result: dict, thresholds: dict | None = None) -> dict:
    """Measure structure and exact evidence ownership, not semantic prose support.

    Repeated quotes from different source locations are distinct provenance.
    Repeated links to the same document/page/span count as duplicates even if
    an alternative opaque handle was used. Source existence in the corpus is
    checked separately by ``run_wiki_cases`` against the supplied store.
    """
    limits = {"min_overview_characters": 40, "min_important_facts": 1,
              "min_evidence_count": 1, "min_related_entities": 0,
              "max_duplicate_evidence": 0, **(thresholds or {})}
    concept = result.get("concept", {})
    artifacts = result.get("artifacts", {})
    failures: list[str] = []
    schema_valid = isinstance(concept, dict) and set(concept) == WIKI_FIELDS
    if schema_valid:
        schema_valid = (
            all(isinstance(concept[name], str) and concept[name].strip()
                for name in ("concept_title", "summary"))
            and isinstance(concept["important_facts"], list)
            and all(isinstance(fact, str) and fact.strip() for fact in concept["important_facts"])
            and all(isinstance(concept[name], list)
                    for name in ("related_entities", "supporting_evidence"))
        )
    if not schema_valid:
        failures.append("Wiki page does not follow the required section schema")
    if not isinstance(concept, dict):
        concept = {}
    summary = concept.get("summary", "")
    facts = concept.get("important_facts", [])
    evidence = concept.get("supporting_evidence", [])
    related = concept.get("related_entities", [])
    summary = summary if isinstance(summary, str) else ""
    facts = facts if isinstance(facts, list) else []
    evidence = evidence if isinstance(evidence, list) else []
    related = related if isinstance(related, list) else []
    valid_count, invalid_count, duplicate_count, missing_relationship_evidence = 0, 0, 0, 0
    seen, sources, sources_without_url = set(), set(), set()
    for kind, entries in (("evidence", evidence), ("relationship", related)):
        for item in entries:
            required = {"evidence_id", "exact_span"}
            if kind == "relationship":
                required |= {"entity_name", "relationship_type"}
            valid = isinstance(item, dict) and set(item) == required
            valid = valid and all(isinstance(item[key], str) and item[key].strip() for key in required)
            if not valid:
                schema_valid = False
            artifact = artifacts.get(item.get("evidence_id")) if valid else None
            span = item.get("exact_span", "") if isinstance(item, dict) else ""
            valid = bool(valid and artifact and span in artifact.original_text_chunk
                         and artifact.document_id and artifact.page_number
                         and (artifact.source_url is None or artifact.source_url.startswith(("https://", "http://"))))
            if not valid:
                invalid_count += 1
                continue
            valid_count += 1
            sources.add(artifact.document_id)
            if artifact.source_url is None:
                sources_without_url.add(artifact.document_id)
            if kind == "evidence":
                identity = (artifact.document_id, artifact.page_number, span)
                duplicate_count += identity in seen
                seen.add(identity)
            elif (artifact.document_id, artifact.page_number, span) not in seen:
                missing_relationship_evidence += 1
    unique_facts = {" ".join(fact.casefold().split()) for fact in facts if isinstance(fact, str) and fact.strip()}
    metrics = {
        "schema_valid": bool(schema_valid), "overview_characters": len(summary.strip()),
        "important_fact_count": len(unique_facts), "duplicate_fact_count": len(facts) - len(unique_facts),
        "related_entity_count": len(related),
        "evidence_count": len(evidence), "evidence_source_count": len(sources),
        "missing_source_url_count": len(sources_without_url),
        "source_url_coverage": (len(sources) - len(sources_without_url)) / len(sources) if sources else 0.0,
        "citation_validity_rate": valid_count / (valid_count + invalid_count) if valid_count + invalid_count else 0.0,
        "invalid_citation_count": invalid_count, "duplicate_evidence_count": duplicate_count,
        "missing_relationship_evidence_count": missing_relationship_evidence,
    }
    if not schema_valid and not failures:
        failures.append("Wiki page does not follow the required section schema")
    for metric, threshold in (("overview_characters", "min_overview_characters"),
                              ("important_fact_count", "min_important_facts"),
                              ("evidence_count", "min_evidence_count"),
                              ("related_entity_count", "min_related_entities")):
        if metrics[metric] < limits[threshold]:
            failures.append(f"{metric}={metrics[metric]} is below {limits[threshold]}")
    if invalid_count:
        failures.append(f"{invalid_count} invalid Wiki evidence or relationship link(s)")
    if duplicate_count > limits["max_duplicate_evidence"]:
        failures.append(f"{duplicate_count} duplicate supporting evidence link(s)")
    if missing_relationship_evidence:
        failures.append(f"{missing_relationship_evidence} relationship quote(s) missing from Supporting Evidence")
    return {"metrics": metrics, "passed": not failures, "failure_reasons": failures}


def _canonical_wiki_links(result: dict, store: KnowledgeStore) -> list[str]:
    failures = []
    for handle, artifact in result.get("artifacts", {}).items():
        digest = hashlib.sha256(artifact.original_text_chunk.encode("utf-8")).hexdigest()
        canonical = store.resolve_source_reference(artifact.document_id, artifact.page_number, digest)
        if canonical != artifact:
            failures.append(f"Wiki artifact {handle} is not the canonical source at its document/page/hash")
    return failures


def evaluate_wiki_expectations(result: dict, case: dict) -> dict:
    """Check inspected content labels against individual, verbatim source quotes."""
    concept, artifacts = result.get("concept", {}), result.get("artifacts", {})
    evidence = concept.get("supporting_evidence", [])
    related = concept.get("related_entities", [])
    normalize = lambda value: " ".join(value.casefold().split())
    quoted = []
    supported = set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        artifact = artifacts.get(item.get("evidence_id"))
        span = item.get("exact_span")
        if artifact is not None and isinstance(span, str) and span and span in artifact.original_text_chunk:
            quoted.append((artifact.document_id, normalize(span)))
            supported.add((item["evidence_id"], span))
    failures, metrics = [], {}
    if "expected_evidence" in case:
        matches = 0
        for expected in case["expected_evidence"]:
            document_id, phrases = expected["document_id"], expected["contains"]
            if not isinstance(document_id, str) or not document_id.strip() or not isinstance(phrases, list) or not phrases:
                raise ValueError("Expected Wiki evidence needs a document ID and nonempty contains list")
            if any(not isinstance(phrase, str) or not phrase.strip() for phrase in phrases):
                raise ValueError("Expected Wiki evidence phrases must be nonempty strings")
            found = any(doc == document_id and all(normalize(phrase) in span for phrase in phrases)
                        for doc, span in quoted)
            matches += found
            if not found:
                failures.append(f"Expected Wiki evidence missing from {document_id}: {phrases!r}")
        metrics.update(expected_evidence_count=len(case["expected_evidence"]),
                       matched_expected_evidence_count=matches)
    if "expected_related_entities" in case:
        names = {normalize(item["entity_name"]) for item in related
                 if isinstance(item, dict) and isinstance(item.get("entity_name"), str)
                 and (item.get("evidence_id"), item.get("exact_span")) in supported}
        matches = 0
        for name in case["expected_related_entities"]:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Expected related entities must be nonempty strings")
            found = normalize(name) in names
            matches += found
            if not found:
                failures.append(f"Expected supported Wiki relationship missing: {name}")
        metrics.update(expected_related_entity_count=len(case["expected_related_entities"]),
                       matched_expected_related_entity_count=matches)
    return {"metrics": metrics, "passed": not failures, "failure_reasons": failures}


def run_wiki_cases(store: KnowledgeStore, cases: list[dict]) -> list[dict]:
    """Evaluate compiled corpus pages on a disposable store supplied by the runner."""
    rows = []
    for case in cases:
        operation = case.get("operation", "cached")
        failures: list[str] = []
        metrics: dict = {}
        details = {"topic": case["topic"], "operation": operation,
                   "semantic_quality_measured": False}
        try:
            with patch.object(wiki, "call_structured_llm", side_effect=_forbid_provider) as provider, \
                    patch.object(wiki, "generate_embedding", side_effect=_forbid_provider), \
                    patch.object(wiki.LOGGER, "exception"):
                topic = case["topic"]
                if operation == "missing_evidence":
                    rejected = False
                    try:
                        wiki.generate_extractive_wiki_concept(topic, store, force_refresh=True)
                    except RuntimeError as error:
                        rejected = "No safe evidence" in str(error)
                    metrics.update(missing_evidence_rejected=rejected, provider_calls=provider.call_count)
                    if not rejected:
                        failures.append("A Wiki page was compiled despite missing topic evidence")
                else:
                    metrics["initial_cache_available"] = store.get_compiled_concept(topic) is not None
                    before = wiki.generate_extractive_wiki_concept(topic, store)
                    result = before
                    if operation == "cached":
                        with patch.object(store, "retrieve", side_effect=AssertionError("Cache hit performed retrieval")):
                            result = wiki.generate_extractive_wiki_concept(topic, store)
                        metrics["cache_hit"] = bool(result.get("cached"))
                        if not metrics["cache_hit"]:
                            failures.append("Pre-generated Wiki selection did not load the cached page")
                    elif operation == "local_recompile":
                        result = wiki.generate_extractive_wiki_concept(topic, store, force_refresh=True)
                    elif operation in {"regenerate_fixture", "invalid_refresh", "provider_failure"}:
                        def scripted_compilation(system, prompt, schema, **kwargs):
                            marker = "LOCAL_COMPILATION_JSON (evidence-grounded starting material):\n"
                            payload = json.loads(prompt.split(marker, 1)[1])
                            if operation == "invalid_refresh":
                                payload["supporting_evidence"][0]["exact_span"] = "This evaluation deliberately fabricated an unsupported quote."
                            if operation == "provider_failure":
                                raise RuntimeError("scripted provider outage")
                            return json.dumps(payload)
                        provider.side_effect = scripted_compilation
                        if operation == "invalid_refresh":
                            rejected = False
                            try:
                                wiki.generate_wiki_concept(topic, store, force_refresh=True)
                            except ValueError as error:
                                rejected = "not verbatim" in str(error)
                            metrics["invalid_refresh_rejected"] = rejected
                            if not rejected:
                                failures.append("Unsupported regenerated evidence was not rejected")
                            result = store.get_compiled_concept(topic)
                            if result["concept"] != before["concept"]:
                                failures.append("Rejected refresh overwrote a valid cached Wiki page")
                        else:
                            result = wiki.generate_wiki_concept(topic, store, force_refresh=True)
                            if operation == "provider_failure":
                                metrics["fallback_preserved_cache"] = bool(result.get("refresh_error")) and result["concept"] == before["concept"]
                                if not metrics["fallback_preserved_cache"]:
                                    failures.append("Provider outage failed to preserve the cached Wiki page")
                    else:
                        raise ValueError(f"Unsupported Wiki operation: {operation}")
                    assessed = evaluate_wiki(result, case.get("thresholds"))
                    metrics.update(assessed["metrics"], provider_calls=provider.call_count)
                    failures.extend(assessed["failure_reasons"])
                    failures.extend(_canonical_wiki_links(result, store))
                    content = evaluate_wiki_expectations(result, case)
                    metrics.update(content["metrics"])
                    failures.extend(content["failure_reasons"])
                    details.update({key: case[key] for key in ("expected_evidence", "expected_related_entities")
                                    if key in case})
                    compatible = set(before["concept"]) == set(result["concept"])
                    metrics["structural_compatibility"] = compatible
                    if not compatible:
                        failures.append("Pre-generated and regenerated section schemas differ")
                    details.update(generation_version=result.get("generation_version"),
                                   model=result.get("model_name"),
                                   source_document_ids=sorted({a.document_id for a in result["artifacts"].values()}))
                expected_calls = 1 if operation in {"regenerate_fixture", "invalid_refresh", "provider_failure"} else 0
                if provider.call_count != expected_calls:
                    failures.append(f"Expected {expected_calls} scripted provider calls, observed {provider.call_count}")
        except Exception as error:
            failures.append(f"{type(error).__name__}: {error}")
        rows.append(_row(case, "wiki", "offline_provider_fixture" if operation in {
            "regenerate_fixture", "invalid_refresh", "provider_failure"} else "offline_corpus", metrics, failures, details))
    return rows


def _check_turn(expected: dict, actual: dict, prefix: str) -> list[str]:
    failures = []
    for key in ("relation", "active_subject", "uses_history", "needs_clarification",
                "resolver_calls", "history_messages_used", "source_count"):
        if key in expected and actual.get(key) != expected[key]:
            failures.append(f"{prefix}: {key} expected {expected[key]!r}, got {actual.get(key)!r}")
    for key, target, absent in (("query_contains", "retrieval_query", False),
                                ("query_not_contains", "retrieval_query", True),
                                ("answer_contains", "answer", False),
                                ("answer_not_contains", "answer", True),
                                ("resolver_prompt_contains", "resolver_prompt", False),
                                ("resolver_prompt_not_contains", "resolver_prompt", True)):
        for phrase in expected.get(key, []):
            present = phrase.casefold() in str(actual.get(target, "")).casefold()
            if present == absent:
                failures.append(f"{prefix}: {target} {'contains forbidden' if absent else 'is missing'} phrase {phrase!r}")
    ids = set(actual.get("source_document_ids", []))
    missing = set(expected.get("source_document_ids", [])) - ids
    forbidden = ids & set(expected.get("forbidden_document_ids", []))
    if missing:
        failures.append(f"{prefix}: missing cited fixture documents {sorted(missing)}")
    if forbidden:
        failures.append(f"{prefix}: wrong-topic cited fixture documents {sorted(forbidden)}")
    if "max_history_messages" in expected and actual.get("history_messages_used", 0) > expected["max_history_messages"]:
        failures.append(f"{prefix}: bounded context exceeded {expected['max_history_messages']} messages")
    return failures


def run_conversation_cases(cases: list[dict]) -> list[dict]:
    """Run real context, retrieval, provenance and thread state with scripted providers.

    ``query_embedding`` is an explicit fixture vector, independent of the expected
    assertions; it does not evaluate semantic embedding quality. Synthesis simply
    quotes the first retrieved canonical chunk(s), so topic drift remains visible.
    New browser pages use the real reset function; this is a state integration
    check, not a browser transport or Streamlit widget end-to-end test.
    """
    rows = []
    for case in cases:
        failures: list[str] = []
        turns = []
        state: dict = {}
        page_number = 1
        page_id = f"{case['case_id']}-page-{page_number}"
        book = threads.synchronize_page_session(state, page_id)
        aliases = {"A": book["active_id"]}
        resolver_total, synthesis_total, reset_checks = 0, 0, 0
        try:
            with (KnowledgeStore(":memory:") if case.get("empty_corpus") else fixture_store()) as store:
                for number, step in enumerate(case["turns"], 1):
                    action = step.get("action", "ask")
                    prefix = f"turn {number}"
                    if action == "refresh":
                        old_ids = set(book["conversations"])
                        page_number += 1
                        page_id = f"{case['case_id']}-page-{page_number}"
                        book = threads.synchronize_page_session(state, page_id)
                        aliases = {"A": book["active_id"]}
                        fresh = len(book["conversations"]) == 1 and not old_ids.intersection(book["conversations"])
                        fresh = fresh and not book["conversations"][book["active_id"]]["messages"]
                        reset_checks += 1
                        if not fresh:
                            failures.append(f"{prefix}: browser refresh did not discard prior history")
                        turns.append({"action": action, "history_cleared": bool(fresh)})
                        continue
                    if action == "rerun":
                        previous = book
                        book = threads.synchronize_page_session(state, page_id)
                        if book is not previous:
                            failures.append(f"{prefix}: same-page rerun discarded its conversation book")
                        turns.append({"action": action, "history_preserved": book is previous})
                        continue
                    alias = step.get("thread", "A")
                    if alias not in aliases:
                        aliases[alias] = threads.new_conversation(book)
                    identifier = aliases[alias]
                    book["active_id"] = identifier
                    if action == "switch":
                        actual_count = len(book["conversations"][identifier]["messages"])
                        if actual_count != step["expected_message_count"]:
                            failures.append(f"{prefix}: resumed thread has {actual_count} messages")
                        turns.append({"action": action, "thread": alias, "message_count": actual_count})
                        continue
                    if action != "ask":
                        raise ValueError(f"Unsupported conversation action: {action}")
                    history = list(book["conversations"][identifier]["messages"])
                    prompts, synthesis_inputs = [], []

                    def resolve_fixture(system, prompt, schema, **kwargs):
                        prompts.append(prompt)
                        if "resolver" not in step:
                            raise AssertionError("Unexpected intent resolver request in this fixture turn")
                        payload = step["resolver"]
                        if payload == "provider_failure":
                            raise RuntimeError("scripted resolver outage")
                        return json.dumps(payload)

                    def synthesize_fixture(system, prompt, artifacts, **kwargs):
                        synthesis_inputs.append(list(artifacts))
                        spans = {handle: " ".join(re.split(r"(?<=[.!?])\s+", artifact.original_text_chunk)[:step["sentence_count"]])
                                 if "sentence_count" in step else artifact.original_text_chunk
                                 for handle, artifact in artifacts.items()}
                        claims = [{"text": spans[handle],
                                   "evidence_ids": [handle],
                                   "supporting_spans": [spans[handle]]}
                                  for handle, artifact in list(artifacts.items())[:step.get("claim_count", 1)]]
                        return json.dumps({"preamble": "", "status": "answered", "claims": claims,
                                           "unsupported_facets": []})

                    diagnostics = {}
                    vector = step.get("query_embedding", [1.0, 0.0, 0.0])
                    with patch.object(chat_context, "call_structured_llm", side_effect=resolve_fixture), \
                            patch.object(main, "generate_embedding", return_value=vector), \
                            patch.object(main, "call_llm", side_effect=synthesize_fixture):
                        answer, preamble, sources = main.ask_chatbot_with_context(
                            step["question"], store, history=history, diagnostics=diagnostics,
                            top_k=step.get("top_k", 2))
                    actual = {**diagnostics, "thread": alias, "answer": answer,
                              "preamble": preamble, "resolver_calls": len(prompts),
                              "resolver_prompt": "\n".join(prompts), "source_count": len(sources),
                              "source_document_ids": [a.document_id for a in sources]}
                    if prompts and "resolver" not in step:
                        failures.append(f"{prefix}: standalone fixture unexpectedly called the intent provider")
                    if sources and "could not be validated" in preamble:
                        failures.append(f"{prefix}: scripted canonical synthesis unexpectedly fell back")
                    failures.extend(_check_turn(step.get("expected", {}), actual, prefix))
                    threads.append_message(book, identifier, {"role": "user", "content": step["question"]})
                    threads.append_message(book, identifier, {"role": "assistant", "content": answer,
                                                             "sources": sources, "context": diagnostics})
                    resolver_total += len(prompts)
                    synthesis_total += len(synthesis_inputs)
                    # Keep the report compact and avoid repeating whole fixture transcripts.
                    turns.append({key: value for key, value in actual.items()
                                  if key not in {"resolver_prompt", "answer", "preamble", "selected_context", "original_query"}})
                for alias, count in case.get("expected_thread_message_counts", {}).items():
                    actual_count = len(book["conversations"][aliases[alias]]["messages"])
                    if actual_count != count:
                        failures.append(f"Thread {alias} has {actual_count} messages, expected {count}")
                for conversation in book["conversations"].values():
                    if not all(message.get("timestamp") for message in conversation["messages"]):
                        failures.append("Conversation messages lost their timestamps")
        except Exception as error:
            failures.append(f"{type(error).__name__}: {error}")
        metrics = {"turn_count": len(turns), "scripted_resolver_calls": resolver_total,
                   "scripted_synthesis_calls": synthesis_total, "refresh_checks": reset_checks,
                   "conversation_count": len(book["conversations"])}
        rows.append(_row(case, "conversation", "offline_fixture", metrics, failures,
                         {"turns": turns, "semantic_quality_measured": False,
                          "fixture_embedding_quality_measured": False,
                          "scope": "Real routing, retrieval, provenance and page-session state; scripted providers"}))
    return rows
