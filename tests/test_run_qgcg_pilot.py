"""Offline audit/provenance tests: no provider credentials or network calls."""

from dataclasses import replace
import csv
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_qgcg_pilot as pilot


def decode_envelope(response):
    value = json.loads(response.raw_text)
    return pilot.DecodedResponse(value["text"], value["model"], "fixture-revision-1")


def reqs():
    return [{"requirement": "Establish the requested finding.",
             "why_indispensable": "The question asks for the finding."}]


def make_groups(payload):
    page_id = next(iter(payload["pages"]))
    return {"mappings": [{**reqs()[0], "evidence_options": [[page_id]],
                         "source_excerpts": [{"page_id": page_id, "quote": payload["pages"][page_id],
                                              "explanation": "Documents the finding."}],
                         "status": "required", "issues": []}]}


def finding():
    return [{"requirement_reference": "G0001", "issue_type": "unsupported_evidence",
             "severity": "high", "evidence_based_explanation": "Revise the supporting evidence.",
             "recommended_action": "revise"}]


def providers(request, decode=decode_envelope):
    return {role: pilot.ProviderSpec(model, request, decode, "offline-subscription-fixture", "test-1")
            for role, model in pilot.DEFAULT_MODELS.items()}


def log_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qgcg-audit-tests-")
        self.root = Path(self.temporary.name)
        self.questions = self.root / "questions.json"
        self.questions.write_text(json.dumps({"Q2": "Exact Q2?\r\n", "Q3": "Exact Q3?"}), encoding="utf-8")
        self.pool = self.root / "pool.csv"
        self.corpus = self.root / "pages.jsonl"
        self.output = self.root / "output"
        rows, pages = [], []
        for qid, core_count, relevant_count in (("Q2", 4, 13), ("Q3", 22, 41), ("QOTHER", 36, 73)):
            for index in range(relevant_count):
                page_id = f"{qid}_P{index:04d}"
                rows.append({"query_id": qid, "page_id": page_id,
                             "core": str(index < core_count).lower(), "relevant": "true"})
                pages.append({"page_id": page_id, "page_text": f"  Exact text for {page_id}.\r\nNext line. "})
        with self.pool.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["query_id", "page_id", "core", "relevant"])
            writer.writeheader()
            writer.writerows(rows)
        self.corpus.write_text("\n".join(json.dumps(p) for p in pages), encoding="utf-8")
        self.calls = []

    def tearDown(self):
        self.temporary.cleanup()

    def request_ok(self, request):
        self.calls.append(request)
        payload = json.loads(request.user_prompt)
        value = reqs() if request.role == "model_a" else make_groups(payload) if request.role == "model_b" else []
        return pilot.RawProviderResponse(json.dumps({"text": json.dumps(value), "model": request.model}))

    def run_pilot(self, **kwargs):
        return pilot.run_pilot(questions_path=self.questions, pool_path=self.pool,
                               corpus_path=self.corpus, output_root=self.output,
                               providers=providers(self.request_ok), **kwargs)

    def make_adapter(self, request, *, decode=decode_envelope, attempts=3):
        specs = providers(request, decode)
        audit = pilot.ArmAudit(self.root / "adapter", "E1", specs, seed=None,
                               max_attempts=attempts, max_repair_rounds=1, run_id="test-run")
        return pilot.AuditedLLMAdapter(audit, specs, "Q2", max_attempts=attempts, sleeper=lambda _: None)

    def call_a(self, adapter):
        return adapter(role="model_a", system_prompt=pilot.qgcg.MODEL_A_SYSTEM_PROMPT,
                       payload={"question": "Exact question"}, json_schema=pilot.qgcg.PASS1_ADAPTER.json_schema())

    def test_atomic_output_retries_transient_windows_file_lock(self):
        path = self.root / "atomic.json"
        path.write_bytes(b"old")
        real_replace = pilot.os.replace
        attempts = []
        def replace_once_locked(source, destination):
            attempts.append(1)
            if len(attempts) == 1:
                error = PermissionError("OneDrive has a temporary sharing lock")
                error.winerror = 32
                raise error
            return real_replace(source, destination)
        with patch.object(pilot.os, "replace", side_effect=replace_once_locked), patch.object(pilot.time, "sleep"):
            pilot.write_durable(path, b"new", replace=True)
        self.assertEqual(path.read_bytes(), b"new")
        self.assertEqual(len(attempts), 2)

    def test_selected_questions_are_frozen_and_recovery_cannot_change_selection(self):
        statuses = self.run_pilot(query_ids=("Q3",), arms=("E1",))
        self.assertEqual(statuses, {"E1": "completed"})
        manifest = json.loads((self.output / "E1/run_manifest.json").read_bytes())
        self.assertEqual(manifest["query_ids"], ["Q3"])
        self.assertEqual(manifest["pool_counts"]["selected_per_query"], {"Q3": 22})
        self.assertTrue(all(call.query_id == "Q3" for call in self.calls))
        self.assertFalse((self.output / "E1/inputs/Q2").exists())
        self.assertFalse((self.output / "E2").exists())
        before = (self.output / "E1/run_manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "original query selection"):
            self.run_pilot(resume_failed=True, arms=("E1",))
        self.assertEqual((self.output / "E1/run_manifest.json").read_bytes(), before)

    def test_invalid_query_selection_is_rejected_before_output_creation(self):
        for ids in ((), ("Q1", "Q1"), ("../../escape",)):
            with self.assertRaises(ValueError):
                self.run_pilot(query_ids=ids)
        self.assertFalse(self.output.exists())

    def test_complete_pipeline_accepts_explicit_gemini_api_only(self):
        specs = providers(self.request_ok)
        specs["model_c"] = replace(specs["model_c"], billing_mode="developer_api",
                                    transport_name="GeminiDeveloperAPI", settings={"api_authorized": True})
        statuses = pilot.run_pilot(questions_path=self.questions, pool_path=self.pool,
                                   corpus_path=self.corpus, output_root=self.output, providers=specs,
                                   query_ids=("Q2",), arms=("E1",))
        self.assertEqual(statuses, {"E1": "completed"})
        self.assertEqual([r.role for r in self.calls], ["model_a", "model_b", "model_c"])
        manifest = json.loads((self.output / "E1/run_manifest.json").read_bytes())
        self.assertEqual(manifest["billing_policy"], "gemini_api_only_gpt_claude_subscription_no_fallback")
        self.assertEqual(manifest["models"]["model_a"]["billing_mode"], "subscription")
        self.assertEqual(manifest["models"]["model_c"]["billing_mode"], "developer_api")
        self.assertEqual(len(manifest["models"]["model_c"]["observed_calls"]), 1)

    def test_atomic_output_preserves_old_file_on_permanent_failure(self):
        path = self.root / "atomic.json"
        path.write_bytes(b"old")
        with patch.object(pilot.os, "replace", side_effect=PermissionError("permanent")):
            with self.assertRaises(PermissionError):
                pilot.write_durable(path, b"new", replace=True)
        self.assertEqual(path.read_bytes(), b"old")

    def test_recovery_uses_frozen_inputs_and_preserves_completed_trials_and_logs(self):
        def fail_one_trial(request):
            if request.arm == "E1" and request.query_id == "Q2" and request.role == "model_b":
                raise pilot.ProviderFailure("simulated local transport interruption")
            return self.request_ok(request)
        arguments = dict(questions_path=self.questions, pool_path=self.pool,
                         corpus_path=self.corpus, output_root=self.output)
        statuses = pilot.run_pilot(**arguments, providers=providers(fail_one_trial))
        self.assertEqual(statuses, {"E1": "partial_failure", "E2": "completed"})
        old_errors = (self.output / "E1/error_logs.jsonl").read_bytes()
        old_raw = (self.output / "E1/raw_responses.jsonl").read_bytes()
        # Another process can own E2: selecting E1 must not inspect/mutate E2.
        e2_manifest = self.output / "E2/run_manifest.json"
        ongoing = json.loads(e2_manifest.read_bytes())
        ongoing.update(status="running", finished_at_utc=None)
        e2_manifest.write_bytes(pilot.json_bytes(ongoing))
        e2_before = {p.name: p.read_bytes() for p in (self.output / "E2").glob("*.json*")}
        self.calls.clear()
        # Original sources need not exist: only immutable archived bytes are used.
        self.questions.unlink()
        self.corpus.unlink()
        self.pool.unlink()
        statuses = pilot.run_pilot(**arguments, providers=providers(self.request_ok), resume_failed=True, arms=("E1",))
        self.assertEqual(statuses, {"E1": "completed"})
        self.assertEqual(len(self.calls), 2)  # The verified Pass 1 response is reused.
        self.assertEqual([c.role for c in self.calls], ["model_b", "model_c"])
        self.assertTrue(all(c.arm == "E1" and c.query_id == "Q2" for c in self.calls))
        self.assertTrue(all(c.call_id.startswith("recovery-") for c in self.calls))
        self.assertEqual((self.output / "E1/error_logs.jsonl").read_bytes(), old_errors)
        self.assertTrue((self.output / "E1/raw_responses.jsonl").read_bytes().startswith(old_raw))
        self.assertEqual(e2_before, {p.name: p.read_bytes() for p in (self.output / "E2").glob("*.json*")})
        manifest = json.loads((self.output / "E1/run_manifest.json").read_bytes())
        reuse = [r for r in log_records(self.output / "E1/requests.jsonl")
                 if r["event"] == "validated_response_reused"]
        self.assertEqual(len(reuse), 1)
        self.assertFalse(reuse[0]["provider_request_made"])
        self.assertEqual(reuse[0]["original_call_id"], "Q2-0001-model_a")
        self.assertEqual(manifest["recovery_segments"][0]["prior_query_outcomes"]["Q2"]["status"], "failed")
        for record in manifest["artifacts"] + list(manifest["output_files"].values()):
            self.assertEqual(pilot.file_sha256(Path(record["path"])), record["sha256"])

    def test_recovery_rejects_modified_archive_before_calling_any_provider(self):
        self.run_pilot()
        (self.output / "E2/inputs/Q2/question.txt").write_bytes(b"tampered")
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "Archived input changed"):
            self.run_pilot(resume_failed=True)
        self.assertEqual(self.calls, [])

    def test_recovery_never_reuses_invalid_quotes_and_seeds_all_mismatches_for_correction(self):
        def request_bad_quote(request):
            if request.arm == "E1" and request.query_id == "Q2" and request.role == "model_b":
                result = make_groups(json.loads(request.user_prompt))
                excerpts = result["mappings"][0]["source_excerpts"]
                excerpts[0]["quote"] = "invented quotation"
                excerpts.append({**excerpts[0], "quote": "another invented quotation"})
                return pilot.RawProviderResponse(json.dumps({"model": request.model, "text": json.dumps(result)}))
            return self.request_ok(request)
        args = dict(questions_path=self.questions, pool_path=self.pool, corpus_path=self.corpus,
                    output_root=self.output)
        pilot.run_pilot(**args, providers=providers(request_bad_quote))
        self.calls.clear()
        pilot.run_pilot(**args, providers=providers(self.request_ok), resume_failed=True, arms=("E1",))
        self.assertEqual([c.role for c in self.calls], ["model_b", "model_c"])
        feedback = json.loads(self.calls[0].validation_feedback)
        self.assertEqual([q["submitted_quote"] for q in feedback["invalid_quotes"]],
                         ["invented quotation", "another invented quotation"])
        result = json.loads((self.output / "E1/groups.json").read_bytes())["queries"]["Q2"]["result"]
        self.assertNotIn("invented quotation", json.dumps(result["groups"]))

    def test_recovery_redecodes_saved_success_without_another_provider_request(self):
        def obsolete_decoder(response):
            if "Q2_P0000" in response.raw_text:
                raise pilot.ProviderFailure("Obsolete local envelope constraint")
            return decode_envelope(response)
        args = dict(questions_path=self.questions, pool_path=self.pool, corpus_path=self.corpus,
                    output_root=self.output, arms=("E1",))
        pilot.run_pilot(**args, providers=providers(self.request_ok, obsolete_decoder))
        saved = (self.output / "E1/raw_responses.jsonl").read_bytes()
        self.calls.clear()
        statuses = pilot.run_pilot(**args, providers=providers(self.request_ok), resume_failed=True)
        self.assertEqual(statuses, {"E1": "completed"})
        self.assertEqual([c.role for c in self.calls], ["model_c"])
        self.assertTrue((self.output / "E1/raw_responses.jsonl").read_bytes().startswith(saved))
        events = log_records(self.output / "E1/requests.jsonl")
        recovered = [r for r in events if r["event"] == "saved_provider_response_redecoded"]
        self.assertEqual(len(recovered), 1)
        self.assertFalse(recovered[0]["provider_request_made"])
        self.assertEqual(recovered[0]["call_id"], "Q2-0002-model_b")

    def test_all_four_trials_are_isolated_and_artifact_hashes_match(self):
        self.assertEqual(self.run_pilot(), {"E1": "completed", "E2": "completed"})
        self.assertEqual(len(self.calls), 12)
        for call in self.calls:
            payload = json.loads(call.user_prompt)
            self.assertEqual(payload["question"], "Exact Q2?\r\n" if call.query_id == "Q2" else "Exact Q3?")
            if call.role != "model_a":
                expected = {("E1", "Q2"): 4, ("E1", "Q3"): 22,
                            ("E2", "Q2"): 13, ("E2", "Q3"): 41}[call.arm, call.query_id]
                self.assertEqual(len(payload["pages"]), expected)
                self.assertTrue(all(page.startswith(call.query_id) for page in payload["pages"]))
        for arm in pilot.ARMS:
            directory = self.output / arm
            manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
            for artifact in manifest["artifacts"] + list(manifest["output_files"].values()):
                self.assertEqual(pilot.file_sha256(Path(artifact["path"])), artifact["sha256"])
            results = json.loads((directory / "groups.json").read_text(encoding="utf-8"))
            self.assertEqual(set(results["queries"]), {"Q2", "Q3"})
            self.assertEqual(manifest["pool_counts"]["global_memberships"], {"core": 62, "relevant": 127})
            self.assertEqual((directory / "inputs/Q2/question.txt").read_bytes(), b"Exact Q2?\r\n")
            self.assertEqual(len(log_records(directory / "raw_responses.jsonl")), 12)
            for model in manifest["models"].values():
                self.assertEqual(len(model["observed_calls"]), 2)
                self.assertTrue(all(c["backend_revision"] == "fixture-revision-1" for c in model["observed_calls"]))

    def test_raw_text_is_durably_saved_before_provider_envelope_parsing(self):
        raw = "provider returned invalid envelope\r\n"
        def request(_):
            return pilot.RawProviderResponse(raw, stderr="original diagnostics\n")
        def decode(response):
            saved = log_records(self.root / "adapter/raw_responses.jsonl")
            self.assertEqual(saved[0]["raw_text"], raw)
            self.assertEqual(saved[1]["raw_text"], "original diagnostics\n")
            return decode_envelope(response)
        adapter = self.make_adapter(request, decode=decode)
        with self.assertRaises(json.JSONDecodeError):
            self.call_a(adapter)
        error = log_records(self.root / "adapter/error_logs.jsonl")[0]
        self.assertEqual(error["event"], "json_parsing_error")
        self.assertEqual(error["query_id"], "Q2")
        self.assertEqual(error["pass_number"], 1)

    def test_model_json_and_pydantic_failures_are_logged_without_rewriting(self):
        responses = iter(["```json\n[]\n```", '[{"requirement":123,"why_indispensable":"x"}]'])
        def request(call):
            return pilot.RawProviderResponse(json.dumps({"text": next(responses), "model": call.model}))
        adapter = self.make_adapter(request)
        self.assertEqual(self.call_a(adapter), "```json\n[]\n```")
        self.assertIn('"requirement":123', self.call_a(adapter))
        errors = log_records(self.root / "adapter/error_logs.jsonl")
        self.assertEqual([e["event"] for e in errors], ["json_parsing_error", "pydantic_validation_error"])
        self.assertIn("validation_errors", errors[1])

    def test_subscription_schema_retry_includes_feedback_and_preserves_both_responses(self):
        outputs = iter(["not-json", json.dumps(reqs())])
        calls = []
        def request(call):
            calls.append(call)
            return pilot.RawProviderResponse(json.dumps({"text": next(outputs), "model": call.model}))
        adapter = self.make_adapter(request)
        adapter.providers = {role: replace(spec, retry_invalid_output=True) for role, spec in adapter.providers.items()}
        self.assertEqual(json.loads(self.call_a(adapter)), reqs())
        self.assertIsNone(calls[0].validation_feedback)
        self.assertIn("Expecting value", calls[1].validation_feedback)
        self.assertIn("previous_model_response", json.loads(calls[1].validation_feedback))
        errors = log_records(self.root / "adapter/error_logs.jsonl")
        self.assertEqual([e["event"] for e in errors], ["json_parsing_error", "schema_response_retry"])

    def test_retry_logs_include_partial_responses_and_attempt_numbers(self):
        attempts = []
        def request(call):
            attempts.append(call.attempt)
            if call.attempt == 1:
                raise pilot.ProviderFailure("Temporary subscription transport failure", retryable=True,
                                             raw_response=pilot.RawProviderResponse("partial unparsed text"),
                                             retry_after_seconds=0)
            return self.request_ok(call)
        adapter = self.make_adapter(request)
        self.call_a(adapter)
        self.assertEqual(attempts, [1, 2])
        errors = log_records(self.root / "adapter/error_logs.jsonl")
        self.assertEqual([e["event"] for e in errors], ["provider_error", "provider_retry"])
        self.assertEqual(errors[1]["next_attempt"], 2)
        raw = log_records(self.root / "adapter/raw_responses.jsonl")
        self.assertEqual(raw[0]["raw_text"], "partial unparsed text")
        self.assertEqual([r["attempt"] for r in raw], [1, 2, 2])

    def test_retry_exhaustion_does_not_switch_model_or_provider(self):
        seen = []
        def request(call):
            seen.append(call.model)
            raise pilot.ProviderFailure("Temporary failure", retryable=True, retry_after_seconds=0)
        adapter = self.make_adapter(request, attempts=2)
        with self.assertRaises(pilot.ProviderFailure):
            self.call_a(adapter)
        self.assertEqual(seen, [pilot.DEFAULT_MODELS["model_a"]] * 2)
        errors = log_records(self.root / "adapter/error_logs.jsonl")
        self.assertEqual(sum(e["event"] == "provider_retry" for e in errors), 1)
        self.assertFalse(errors[-1]["will_retry"])

    def test_model_mismatch_is_recorded_and_rejected(self):
        def request(_):
            return pilot.RawProviderResponse(json.dumps({"text": "[]", "model": "wrong-model"}))
        adapter = self.make_adapter(request)
        with self.assertRaises(pilot.ProviderFailure):
            self.call_a(adapter)
        self.assertEqual(adapter.audit.manifest["models"]["model_a"]["observed_calls"][0]["reported_model"], "wrong-model")
        self.assertEqual(log_records(self.root / "adapter/error_logs.jsonl")[-1]["event"],
                         "model_or_transport_verification_error")

    def test_swallowed_repair_errors_are_also_in_error_logs(self):
        def request(call):
            payload = json.loads(call.user_prompt)
            if call.role == "model_a":
                value = reqs()
            elif call.role == "model_c":
                value = finding()
            else:
                value = make_groups(payload)
                if "verifier_issues" in payload:
                    value["mappings"][0]["source_excerpts"][0]["quote"] = "invented text"
            return pilot.RawProviderResponse(json.dumps({"text": json.dumps(value), "model": call.model}))
        status = pilot.run_pilot(questions_path=self.questions, pool_path=self.pool, corpus_path=self.corpus,
                                  output_root=self.output, providers=providers(request))
        self.assertEqual(status, {"E1": "completed", "E2": "completed"})
        for arm in pilot.ARMS:
            errors = log_records(self.output / arm / "error_logs.jsonl")
            self.assertEqual(sum(e["event"] == "repair_validation_error" for e in errors), 2)
            self.assertEqual(sum(e["event"] == "pydantic_validation_error" for e in errors), 2)
            groups = json.loads((self.output / arm / "groups.json").read_text())
            self.assertTrue(all(q["result"]["review_status"] == "needs_review" for q in groups["queries"].values()))

    def test_prepare_only_never_calls_any_transport_and_refuses_overwrite(self):
        self.assertEqual(self.run_pilot(prepare_only=True), {"E1": "prepared_only", "E2": "prepared_only"})
        self.assertEqual(self.calls, [])
        before = (self.output / "E1/run_manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_pilot()
        self.assertEqual((self.output / "E1/run_manifest.json").read_bytes(), before)

    def test_missing_inputs_leave_failed_manifests_and_logs(self):
        self.questions.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_pilot()
        for arm in pilot.ARMS:
            manifest = json.loads((self.output / arm / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "blocked_input")
            self.assertEqual(log_records(self.output / arm / "error_logs.jsonl")[0]["event"], "input_preflight_error")
        self.assertEqual(self.calls, [])

    def test_wrong_global_pool_counts_are_rejected(self):
        with self.pool.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows.pop()
        with self.pool.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "Wrong consensus pool"):
            pilot.load_pools(self.pool)

    def test_sqlite_canonical_pages_and_chunk_refusal(self):
        db = self.root / "canonical.db"
        with sqlite3.connect(db) as connection:
            connection.execute("CREATE TABLE pages (page_id TEXT, page_text TEXT)")
            connection.execute("INSERT INTO pages VALUES (?,?)", ("P1", "exact\r\ntext"))
        connection.close()
        before = db.read_bytes()
        pages, _, provenance = pilot.load_canonical_pages(db, {"P1"})
        self.assertEqual(pages, {"P1": "exact\r\ntext"})
        self.assertEqual(db.read_bytes(), before)
        self.assertTrue(any("sqlite_query" in item for item in provenance))
        chunk_db = self.root / "chunks.db"
        with sqlite3.connect(chunk_db) as connection:
            connection.execute("CREATE TABLE knowledge_artifacts (original_text_chunk TEXT)")
        connection.close()
        with self.assertRaisesRegex(ValueError, "Do not concatenate chunks"):
            pilot.load_canonical_pages(chunk_db, {"P1"})

    def test_subscription_only_and_missing_page_guards(self):
        specs = providers(self.request_ok)
        specs["model_a"] = replace(specs["model_a"], billing_mode="paid_api")
        with self.assertRaisesRegex(ValueError, "subscription-only"):
            pilot.validate_providers(specs)
        with self.assertRaisesRegex(ValueError, "Missing canonical page"):
            pilot.load_canonical_pages(self.corpus, {"NONEXISTENT"})

    def test_manual_abort_stops_run_and_preserves_both_arm_manifests(self):
        transport = pilot.HITLClipboardTransport(input_fn=lambda _: ":abort", print_fn=lambda _: None,
                                                 copy_fn=lambda _: None)
        with self.assertRaises(pilot.ManualRunAborted):
            pilot.run_pilot(questions_path=self.questions, pool_path=self.pool,
                            corpus_path=self.corpus, output_root=self.output,
                            providers=pilot.hitl_providers(transport))
        for arm in pilot.ARMS:
            manifest = json.loads((self.output / arm / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "interrupted")
        self.assertTrue(any(e["event"] == "manual_aborted"
                            for e in log_records(self.output / "E1/error_logs.jsonl")))

    def test_cli_prepare_only_uses_manual_transport_without_interaction(self):
        result = subprocess.run(
            [sys.executable, str(Path(pilot.__file__)), "--data-root", str(self.root),
             "--questions", str(self.questions), "--pool", str(self.pool), "--corpus", str(self.corpus),
             "--output-root", str(self.output), "--prepare-only", "--transport", "manual"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"E1": "prepared_only", "E2": "prepared_only"})
        manifest = json.loads((self.output / "E1/run_manifest.json").read_text())
        self.assertTrue(all(spec["transport"] == "manual_web_subscription_clipboard"
                            for spec in manifest["models"].values()))

    def make_manual_adapter(self, lines, *, copy_fn=None, paste_fn=None):
        iterator = iter(lines)
        self.printed, self.copied = [], []
        def get_line(_):
            try:
                return next(iterator)
            except StopIteration:
                raise EOFError("Test input exhausted")
        transport = pilot.HITLClipboardTransport(
            input_fn=get_line, print_fn=self.printed.append,
            copy_fn=self.copied.append if copy_fn is None else copy_fn, paste_fn=paste_fn,
        )
        specs = pilot.hitl_providers(transport)
        audit = pilot.ArmAudit(self.root / "manual", "E1", specs, seed=None,
                               max_attempts=1, max_repair_rounds=1, run_id="manual-test")
        return pilot.AuditedLLMAdapter(audit, specs, "Q2", max_attempts=1)

    def test_hitl_retries_json_and_schema_errors_without_clobbering_clipboard(self):
        valid = json.dumps(reqs(), indent=2)
        bad_schema = '[{"requirement":123,"why_indispensable":"why"}]'
        adapter = self.make_manual_adapter(["GPT-5.6", "not json", "EOF", bad_schema, "EOF",
                                            *valid.split("\n"), "EOF"])
        self.assertEqual(self.call_a(adapter), valid)
        self.assertEqual(len(self.copied), 1)
        errors = log_records(self.root / "manual/error_logs.jsonl")
        self.assertEqual([e["event"] for e in errors],
                         ["json_parsing_error", "manual_response_retry", "pydantic_validation_error", "manual_response_retry"])
        raw = [r for r in log_records(self.root / "manual/raw_responses.jsonl") if r["event"] == "provider_raw"]
        self.assertEqual([r["raw_text"] for r in raw], ["not json", bad_schema, valid])
        self.assertEqual([r["attempt"] for r in raw], [1, 2, 3])
        self.assertTrue(any("VALIDATION FAILED" in line for line in self.printed))
        saved_prompt = next((self.root / "manual/requests").glob("*/clipboard_prompt.txt"))
        self.assertEqual(saved_prompt.read_bytes().decode("utf-8"), self.copied[0])
        self.assertIn(pilot.qgcg.MODEL_A_SYSTEM_PROMPT, self.copied[0])
        self.assertIn('"question": "Exact question"', self.copied[0])
        observations = adapter.audit.manifest["models"]["model_a"]["observed_calls"]
        self.assertTrue(all(o["reported_model"] is None and not o["provider_verified"] for o in observations))

    def test_hitl_file_capture_preserves_exact_response_newlines(self):
        response = self.root / "response with spaces.txt"
        raw = json.dumps(reqs(), indent=2).replace("\n", "\r\n") + "\r\n"
        response.write_bytes(raw.encode("utf-8"))
        adapter = self.make_manual_adapter(["GPT-5.6", f':file "{response}"'])
        self.assertEqual(self.call_a(adapter), raw)
        record = log_records(self.root / "manual/raw_responses.jsonl")[0]
        self.assertEqual(record["capture_method"], "utf8_file")
        self.assertEqual(record["capture_source"], str(response.resolve()))

    def test_hitl_clipboard_capture_and_copy_failure_fallback_are_audited(self):
        raw = json.dumps(reqs())
        def unavailable(_):
            raise RuntimeError("clipboard busy")
        adapter = self.make_manual_adapter(["GPT-5.6", ":clipboard"], copy_fn=unavailable, paste_fn=lambda: raw)
        self.assertEqual(self.call_a(adapter), raw)
        errors = log_records(self.root / "manual/error_logs.jsonl")
        self.assertEqual(errors[0]["event"], "clipboard_error")
        self.assertTrue(any("Copy the complete saved prompt" in line for line in self.printed))

    def test_hitl_eof_preserves_partial_response_and_aborts(self):
        adapter = self.make_manual_adapter(["GPT-5.6", "partial text"])
        with self.assertRaises(pilot.ManualRunAborted):
            self.call_a(adapter)
        self.assertEqual(log_records(self.root / "manual/raw_responses.jsonl")[0]["raw_text"], "partial text")
        self.assertEqual(log_records(self.root / "manual/error_logs.jsonl")[-1]["event"], "manual_aborted")

    def test_hitl_validates_pass2_source_quotes_before_accepting_correction(self):
        payload = {"question": "Q2", "requirements": reqs(), "pages": {"P1": "exact source"}}
        valid = make_groups(payload)
        invalid = make_groups(payload)
        invalid["mappings"][0]["source_excerpts"][0]["quote"] = "invented"
        adapter = self.make_manual_adapter(["Claude Sonnet 5", json.dumps(invalid), "EOF", json.dumps(valid), "EOF"])
        raw = adapter(role="model_b", system_prompt=pilot.qgcg.MODEL_B_SYSTEM_PROMPT,
                      payload=payload, json_schema=pilot.qgcg.Pass2Output.model_json_schema())
        self.assertEqual(json.loads(raw), valid)
        self.assertIn("Non-verbatim quote", log_records(self.root / "manual/error_logs.jsonl")[0]["message"])

    def test_hitl_validates_pass3_reference_before_accepting_correction(self):
        invalid = finding()
        invalid[0]["requirement_reference"] = "G9999"
        adapter = self.make_manual_adapter(["Gemini 3.1", json.dumps(invalid), "EOF", "[]", "EOF"])
        raw = adapter(role="model_c", system_prompt=pilot.qgcg.MODEL_C_SYSTEM_PROMPT,
                      payload={"requirement_references": {"G0001": "finding"}},
                      json_schema=pilot.qgcg.PASS3_ADAPTER.json_schema())
        self.assertEqual(raw, "[]")
        self.assertIn("Unknown requirement reference", log_records(self.root / "manual/error_logs.jsonl")[0]["message"])

    def test_complete_hitl_pipeline_all_four_trials_with_scripted_human_input(self):
        prepared = pilot.load_pilot_inputs(self.questions, self.pool, self.corpus)
        lines, seen = [], set()
        for arm in pilot.ARMS:
            for qid in pilot.QUERY_IDS:
                selected = {page: prepared.pages[page] for page in prepared.pools[arm][qid]}
                for role, value in (("model_a", reqs()), ("model_b", make_groups({"pages": selected})), ("model_c", [])):
                    if role not in seen:
                        lines.append(pilot.WEB_MODELS[role][1])
                        seen.add(role)
                    lines.extend([json.dumps(value), "EOF"])
        iterator, copied = iter(lines), []
        transport = pilot.HITLClipboardTransport(input_fn=lambda _: next(iterator), print_fn=lambda _: None,
                                                 copy_fn=copied.append)
        statuses = pilot.run_pilot(questions_path=self.questions, pool_path=self.pool, corpus_path=self.corpus,
                                   output_root=self.output, providers=pilot.hitl_providers(transport))
        self.assertEqual(statuses, {"E1": "completed", "E2": "completed"})
        self.assertEqual(len(copied), 12)
        for arm in pilot.ARMS:
            manifest = json.loads((self.output / arm / "run_manifest.json").read_text())
            for artifact in manifest["artifacts"] + list(manifest["output_files"].values()):
                self.assertEqual(pilot.file_sha256(Path(artifact["path"])), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
