"""Offline regression checks for experimental scoring, blinding and provenance."""
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_qgcg_pilot as ev


def group(options, status="required", name="G1"):
    return {"group_id": name, "requirement": "Complete the necessary comparison.",
            "why_indispensable": "The question requests the comparison.", "status": status,
            "evidence_options": options, "source_excerpts": [], "issues": []}


class EvaluationTests(unittest.TestCase):
    def test_or_of_and_requires_all_pages_of_one_complete_option(self):
        options = [["A"], ["B", "C"]]
        self.assertTrue(ev.boolean_hit(options, {"A"}))
        self.assertTrue(ev.boolean_hit(options, {"B", "C"}))
        self.assertFalse(ev.boolean_hit(options, {"B"}))
        self.assertFalse(ev.boolean_hit(options, {"C"}))
        self.assertFalse(ev.boolean_hit([], {"A", "B", "C"}))
        self.assertFalse(ev.boolean_hit([[]], set()))

    def test_required_unresolved_supplementary_denominators(self):
        groups = [group([["A"]]), group([["B", "C"]]), group([], "unresolved"),
                  group([["D"]], "supplementary")]
        m = ev.group_metrics(groups, {"A", "B", "D"})
        self.assertEqual(m["hit_required_groups"], 1)
        self.assertEqual(m["CoreGroupRecall_excluding_unresolved"], 0.5)
        self.assertEqual(m["CoreGroupRecall_including_unresolved_as_misses"], 1 / 3)
        self.assertEqual(m["supplementary_groups"], 1)
        m = ev.group_metrics([group([], "unresolved")], set())
        self.assertIsNone(m["CoreGroupRecall_excluding_unresolved"])
        self.assertEqual(m["CoreGroupRecall_including_unresolved_as_misses"], 0)

    def test_original_cutoff_and_unavailable_depth_not_zero(self):
        spec = {"run_key": "frozen", "System_Name": "System", "collection": "batch",
                "protocol": "fixed", "target_depth": 100, "source": "test.csv", "source_sha256": "sha",
                "ranks": {"Q2": {"A": 20, "B": 21}}}
        sets = {"Q2": {arm: [group([["A", "B"]])] for arm in ev.ARMS}}
        rows, _ = ev.diagnostic_recall([spec], sets, query_ids=("Q2",))
        self.assertEqual(rows[0]["CoreGroupRecall_excluding_unresolved"], 0)
        self.assertEqual(rows[3]["CoreGroupRecall_excluding_unresolved"], 1)
        spec["target_depth"] = 20
        rows, _ = ev.diagnostic_recall([spec], sets, query_ids=("Q2",))
        self.assertEqual(rows[3]["status"], "N/A_depth_not_available")
        self.assertIsNone(rows[3]["CoreGroupRecall_excluding_unresolved"])
        spec["ranks"] = {}
        rows, _ = ev.diagnostic_recall([spec], sets, query_ids=("Q2",))
        self.assertEqual(rows[0]["status"], "N/A_missing_query")

    def test_e0_preserves_singleton_or_membership(self):
        source = {"groups": [{"group_id": "Q1_G03", "facet_description": "Schedule",
                              "why_this_facet_is_core": "Necessary", "member_page_ids": ["A", "B"]}]}
        normalized = ev.normalize_baseline(source, {"A": "a" * 1400, "B": "complete"})
        self.assertEqual(normalized[0]["evidence_options"], [["A"], ["B"]])
        self.assertTrue(normalized[0]["source_excerpts"][0]["truncated"])
        self.assertTrue(ev.boolean_hit(normalized[0]["evidence_options"], {"B"}))

    def test_blinding_is_persistent_private_and_rejects_changed_groups(self):
        sets = {q: {arm: [group([["A"]], name=f"{arm}-secret-original")]
                    for arm in ev.ARMS} for q in ev.PILOT_IDS}
        with tempfile.TemporaryDirectory() as folder:
            key_path = Path(folder) / "blind_key.json"
            key = ev.blind_assignment(sets, key_path, 12345)
            before = key_path.read_bytes()
            self.assertEqual(key, ev.blind_assignment(sets, key_path))
            self.assertEqual(before, key_path.read_bytes())
            for qid in ev.PILOT_IDS:
                self.assertEqual(set(key["assignments"][qid]), {"Set_A", "Set_B", "Set_C"})
                self.assertEqual(set(key["assignments"][qid].values()), set(ev.ARMS))
            rows = ev.make_blind_rows(sets, {q: "Question?" for q in ev.PILOT_IDS}, key)
            html = ev.blind_html(rows, {"A": "Canonical text </script> <script>alert(1)</script>"}, key)
            for leaked in ("secret-original", '"assignments"', '"seed"', "E0", "E1", "E2"):
                self.assertNotIn(leaked, html)
                self.assertNotIn(leaked, ev.csv_bytes(rows).decode("utf-8-sig"))
            self.assertNotIn('<script>alert(1)</script>', html)
            self.assertIn('id="export"', html)
            changed = copy.deepcopy(sets)
            changed["Q2"]["E0"][0]["status"] = "unresolved"
            with self.assertRaisesRegex(ValueError, "Blind groups changed"):
                ev.blind_assignment(changed, key_path)

    def test_renamed_groups_requires_exact_manifest_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            renamed = root / "groups-E1.json"
            renamed.write_bytes(b'{"value":1}')
            audit = ev.Provenance()
            self.assertEqual(ev.resolve_output(root, "groups.json", ev.digest(renamed.read_bytes()), audit), renamed)
            self.assertEqual(len(audit.relocations), 1)
            with self.assertRaisesRegex(ValueError, "uniquely resolve"):
                ev.resolve_output(root, "groups.json", "wrong-hash", audit)
            with self.assertRaises(ValueError):
                ev.safe_path(root, "../escape")

    def test_operations_do_not_double_count_streams_redecoding_or_replays(self):
        request = {"query_id": "Q2", "event": "provider_attempt", "call_id": "call-1",
                   "attempt": 1, "timestamp_utc": "2026-01-01T00:00:00+00:00"}
        raw = {**request, "event": "provider_raw", "pass_number": 2,
               "timestamp_utc": "2026-01-01T00:00:02+00:00"}
        trial = {"repair_history": [], "initial_issues": [], "issues": []}
        data = {"manifest": {"arm": "E1", "query_ids": ["Q2"],
                             "billing_policy": "subscription_only_no_api_key_fallback",
                             "started_at_utc": request["timestamp_utc"], "finished_at_utc": raw["timestamp_utc"],
                             "queries": {"Q2": {"status": "completed", "started_at_utc": request["timestamp_utc"],
                                                   "finished_at_utc": raw["timestamp_utc"]}}},
                "trials": {"Q2": trial}, "errors": [],
                "requests": [request, {**request, "call_id": "local-failure"},
                             {**request, "event": "validated_response_reused"}],
                "raw": [raw, {**raw, "event": "provider_stream"}, {**raw, "event": "model_text"}]}
        row = ev.operational_stats(data)[0]
        self.assertEqual(row["provider_attempts"], 2)
        self.assertEqual(row["cli_invocations_with_raw_capture"], 1)
        self.assertEqual(row["attempts_without_raw_capture"], 1)
        self.assertEqual(row["observed_attempt_elapsed_seconds"], 2)
        self.assertEqual(row["validated_response_reuses"], 1)

    def test_mixed_api_and_subscription_calls_and_usage_are_separate(self):
        start, end = "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:02+00:00"
        a = {"query_id": "Q1", "event": "provider_attempt", "call_id": "cli", "attempt": 1,
             "role": "model_a", "pass_number": 1, "timestamp_utc": start}
        c = {**a, "call_id": "api", "role": "model_c", "pass_number": 3}
        cli_raw = {**a, "event": "provider_raw", "timestamp_utc": end, "raw_text": "CLI"}
        api_raw = {**c, "event": "provider_raw", "timestamp_utc": end, "exit_code": 0,
                   "raw_text": json.dumps({"usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5,
                                                             "thoughtsTokenCount": 3, "totalTokenCount": 28}})}
        data = {"manifest": {"arm": "E1", "query_ids": ["Q1"],
                             "billing_policy": "gemini_api_only_gpt_claude_subscription_no_fallback",
                             "models": {"model_a": {"billing_mode": "subscription"},
                                        "model_c": {"billing_mode": "developer_api"}},
                             "queries": {"Q1": {"status": "completed", "started_at_utc": start, "finished_at_utc": end}},
                             "started_at_utc": start, "finished_at_utc": end},
                "trials": {"Q1": {"repair_history": [], "initial_issues": [], "issues": []}}, "errors": [],
                "requests": [a, c, {**c, "event": "api_request_started"}],
                "raw": [cli_raw, api_raw, {**api_raw, "event": "provider_http_body"}]}
        stats = ev.operational_stats(data)[0]
        self.assertEqual(stats["cli_invocations_with_raw_capture"], 1)
        self.assertEqual(stats["developer_api_calls"], 1)
        self.assertEqual(stats["developer_api_responses_captured"], 1)
        self.assertEqual(stats["provider_responses_with_raw_capture"], 2)
        self.assertEqual(json.loads(stats["api_token_usage"])["totalTokenCount"], 28)

    def test_requirement_rewrite_is_traced_through_accepted_history(self):
        before = group([["A"]])
        after = {**before, "requirement": "A revised, source-qualified requirement.", "evidence_options": [["A", "B"]]}
        issue = {"requirement_reference": "G0001", "issue_type": "unsupported_evidence", "severity": "high",
                 "recommended_action": "revise", "evidence_based_explanation": "The initial scope was unsupported."}
        result = {"initial_groups": {"mappings": [before]}, "groups": {"mappings": [after]},
                  "initial_issues": [issue], "issues": [],
                  "repair_history": [{"round": 1, "accepted": True, "before": {"mappings": [before]},
                                      "candidate": {"mappings": [after]}}]}
        row = ev.intervention_rows({"trials": {"Q7": result}})[0]
        self.assertEqual(row["after_requirement"], after["requirement"])
        self.assertEqual(row["after_options"], [["A", "B"]])
        self.assertEqual(row["correspondence"], "unique requirement rewrite in accepted repair history")
        # Without an accepted transition, a changed ordinal is not an identity proof.
        result["repair_history"][0]["accepted"] = False
        row = ev.intervention_rows({"trials": {"Q7": result}})[0]
        self.assertEqual(row["correspondence"], "not uniquely established")
        self.assertEqual(row["after_requirement"], "see complete repair history")


if __name__ == "__main__":
    unittest.main()
