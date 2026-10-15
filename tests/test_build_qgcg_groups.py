"""Offline behavioral checks for QGCG schemas, repairs, and orchestration."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_qgcg_groups as q


QUESTION = "Compare the 2020 and 2025 totals."
REQUIREMENTS = [{
    "requirement": "Compare the totals in 2020 and 2025.",
    "why_indispensable": "The question explicitly asks for this comparison.",
}]
PAGES = {
    "A": "In 2020 the total was 10; in 2025 it was 20.",
    "B": "The 2020 total was 10.",
    "C": "The 2025 total was 20.",
}


def proposal(options=None):
    return {"mappings": [{
        **REQUIREMENTS[0],
        "evidence_options": [["A"], ["B", "C"]] if options is None else options,
        "source_excerpts": [
            {"page_id": page_id, "quote": text, "explanation": "Supports the totals."}
            for page_id, text in PAGES.items()
        ],
        "status": "required",
        "issues": [],
    }]}


def issue(reference="G0001", issue_type="unsupported_evidence", action="revise"):
    return {
        "requirement_reference": reference,
        "issue_type": issue_type,
        "severity": "high",
        "evidence_based_explanation": "Page B gives only 2020; add C to complete the comparison.",
        "recommended_action": action,
    }


class ScriptedAdapter:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, **request):
        self.calls.append(request)
        return json.dumps(next(self.responses))


class QGCGTests(unittest.TestCase):
    def setUp(self):
        self.token = q._CONFIGURATION.set(None)
        self.requirements = q.PASS1_ADAPTER.validate_python(REQUIREMENTS)
        self.context = {"requirements": self.requirements, "pages": PAGES}

    def tearDown(self):
        q._CONFIGURATION.reset(self.token)

    def validate_groups(self, value, context=None):
        return q.Pass2Output.model_validate_json(
            json.dumps(value), context=context or self.context
        )

    def run_pipeline(self, adapter, *, max_repair_rounds=1):
        q.configure_qgcg(adapter, max_repair_rounds=max_repair_rounds)
        return q.build_qgcg(
            QUESTION, [{"page_id": key, "text": text} for key, text in PAGES.items()]
        )

    def test_json_schema_shapes_and_strict_objects(self):
        self.assertEqual(q.PASS1_ADAPTER.json_schema()["type"], "array")
        self.assertEqual(q.PASS3_ADAPTER.json_schema()["type"], "array")
        schema = q.Pass2Output.model_json_schema()
        self.assertFalse(schema["additionalProperties"])
        options = schema["$defs"]["RequirementMapping"]["properties"]["evidence_options"]
        self.assertEqual(options["type"], "array")
        self.assertEqual(options["items"]["type"], "array")
        self.assertEqual(options["items"]["minItems"], 1)
        self.assertEqual(options["items"]["items"]["type"], "string")
        for model in (q.Requirement, q.RequirementMapping, q.SourceExcerpt, q.VerificationIssue):
            self.assertEqual(model.model_config, {"extra": "forbid", "strict": True})

    def test_pass1_rejects_missing_blank_extra_duplicate_and_coerced_values(self):
        invalid = [[], {"requirements": REQUIREMENTS},
                   [{**REQUIREMENTS[0], "requirement": " \n"}],
                   [{**REQUIREMENTS[0], "requirement": 2}],
                   [{**REQUIREMENTS[0], "optional": True}],
                   [REQUIREMENTS[0], REQUIREMENTS[0]]]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                q.PASS1_ADAPTER.validate_json(json.dumps(value))

    def test_accepts_or_and_and_mixed_options(self):
        for options in ([["A"]], [["B", "C"]], [["A"], ["B", "C"]]):
            with self.subTest(options=options):
                self.assertEqual(self.validate_groups(proposal(options)).mappings[0].evidence_options, options)

    def test_rejects_malformed_duplicate_and_redundant_options(self):
        invalid = [[], [[]], ["A"], [[["A"]]], [[1]], [[True]], [[""]], [[" "]],
                   [["A", "A"]], [["B", "C"], ["C", "B"]],
                   [["A"], ["A", "B"]], [["B", "C"], ["A", "B", "C"]]]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValidationError):
                self.validate_groups(proposal(options))

    def test_rejects_unknown_pages_and_quotes_even_outside_options(self):
        cases = []
        unknown = proposal([["UNKNOWN"]])
        unknown["mappings"][0]["source_excerpts"].append(
            {"page_id": "UNKNOWN", "quote": "Invented", "explanation": "Invented"}
        )
        cases.append(unknown)
        bad_quote = proposal([["A"]])
        bad_quote["mappings"][0]["source_excerpts"][1]["quote"] = "The total was 999."
        cases.append(bad_quote)
        missing_quote = proposal()
        missing_quote["mappings"][0]["source_excerpts"] = []
        cases.append(missing_quote)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                self.validate_groups(value)

    def test_quote_matching_preserves_whitespace(self):
        value = proposal([["A"]])
        value["mappings"][0]["source_excerpts"] = [{
            "page_id": "A", "quote": "one\ntwo", "explanation": "Two lines."
        }]
        context = {"requirements": self.requirements, "pages": {"A": "one\r\ntwo"}}
        with self.assertRaises(ValidationError):
            self.validate_groups(value, context)
        value["mappings"][0]["source_excerpts"][0]["quote"] = "one\r\ntwo"
        self.validate_groups(value, context)

    def test_status_and_requirement_preservation(self):
        for update in ({"status": "supplementary"}, {"requirement": "A different demand"},
                       {"why_indispensable": "A changed rationale"}, {"extra": "ignored?"}):
            value = proposal()
            value["mappings"][0].update(update)
            with self.subTest(update=update), self.assertRaises(ValidationError):
                self.validate_groups(value)
        unresolved = proposal([])
        unresolved["mappings"][0].update(status="unresolved", issues=["Evidence is incomplete."])
        self.validate_groups(unresolved)
        unresolved["mappings"][0]["issues"] = []
        with self.assertRaises(ValidationError):
            self.validate_groups(unresolved)

    def test_pass3_validates_reference_sentinel_and_action(self):
        context = {"requirement_references": {"G0001": REQUIREMENTS[0]["requirement"]}}
        for valid in ([], [issue()], [issue("NEW_REQUIREMENT", "missing_requirement", "add")]):
            q.PASS3_ADAPTER.validate_json(json.dumps(valid), context=context)
        for invalid in ([issue("G9999")], [issue("NEW_REQUIREMENT")],
                        [issue("G0001", "missing_requirement", "add")],
                        [issue("NEW_REQUIREMENT", "missing_requirement", "keep")],
                        [{**issue(), "recommended_action": "rewrite"}], {"issues": []}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                q.PASS3_ADAPTER.validate_json(json.dumps(invalid), context=context)

    def test_page_normalization_rejects_duplicate_ids_without_trimming(self):
        pages = [{"page_id": "A", "text": " x\r\ny "}]
        self.assertEqual(q.normalize_pages(pages), {"A": " x\r\ny "})
        with self.assertRaises(ValueError):
            q.normalize_pages(pages + pages)
        with self.assertRaises(ValidationError):
            q.normalize_pages({"A": 42})

    def test_three_pass_pipeline_and_independent_inputs(self):
        adapter = ScriptedAdapter(REQUIREMENTS, proposal(), [])
        result = self.run_pipeline(adapter)
        self.assertEqual([c["role"] for c in adapter.calls], ["model_a", "model_b", "model_c"])
        self.assertEqual(adapter.calls[0]["payload"], {"question": QUESTION})
        self.assertNotIn("verifier_issues", adapter.calls[2]["payload"])
        self.assertEqual(result["review_status"], "verified")
        self.assertEqual(result["groups"], proposal())
        self.assertEqual(result["repair_history"], [])
        self.assertEqual(len(result["source_pages_sha256"]), 64)
        json.dumps(result)

    def test_programmatic_validation_stops_bad_proposal_before_model_c(self):
        invalid = proposal()
        invalid["mappings"][0]["source_excerpts"][0]["quote"] = "Fabricated text"
        adapter = ScriptedAdapter(REQUIREMENTS, invalid)
        with self.assertRaises(ValidationError):
            self.run_pipeline(adapter)
        self.assertEqual(len(adapter.calls), 2)

    def test_repair_is_revalidated_reverified_and_audited(self):
        incomplete = proposal([["B"]])  # Syntactically valid; semantically incomplete.
        adapter = ScriptedAdapter(REQUIREMENTS, incomplete, [issue()], proposal(), [])
        result = self.run_pipeline(adapter)
        self.assertEqual([c["role"] for c in adapter.calls],
                         ["model_a", "model_b", "model_c", "model_b", "model_c"])
        self.assertEqual(result["review_status"], "verified")
        self.assertEqual(result["initial_groups"], incomplete)
        self.assertEqual(result["initial_issues"], [issue()])
        self.assertEqual(result["issues"], [])
        self.assertTrue(result["repair_history"][0]["accepted"])
        self.assertNotIn("verifier_issues", adapter.calls[-1]["payload"])

    def test_invalid_repair_preserves_previous_proposal(self):
        before = proposal([["B"]])
        invalid = proposal()
        invalid["mappings"][0]["source_excerpts"][0]["quote"] = "Invented quote"
        adapter = ScriptedAdapter(REQUIREMENTS, before, [issue()], invalid)
        result = self.run_pipeline(adapter)
        self.assertEqual(result["groups"], before)
        self.assertEqual(result["review_status"], "needs_review")
        self.assertIsNotNone(result["repair_error"])
        self.assertFalse(result["repair_history"][0]["accepted"])

    def test_repair_cannot_modify_unaffected_groups(self):
        second = {"requirement": "Establish the comparison's scope.",
                  "why_indispensable": "The scope qualifies the interpretation."}
        requirements = REQUIREMENTS + [second]
        before = proposal([["B"]])
        before["mappings"].append({**deepcopy(proposal()["mappings"][0]), **second})
        tampered = deepcopy(before)
        tampered["mappings"][0]["evidence_options"] = [["A"]]
        tampered["mappings"][1]["why_indispensable"] = "Silently changed rationale."
        adapter = ScriptedAdapter(requirements, before, [issue()], tampered)
        result = self.run_pipeline(adapter)
        self.assertEqual(result["groups"], before)
        self.assertIn("unaffected", result["repair_error"])

    def test_missing_requirement_can_be_added_as_unresolved(self):
        missing = issue("NEW_REQUIREMENT", "missing_requirement", "add")
        corrected = proposal()
        corrected["mappings"].append({
            "requirement": "Establish a missing question constraint.",
            "why_indispensable": "It is a material constraint.",
            "evidence_options": [], "source_excerpts": [], "status": "unresolved",
            "issues": ["No supplied page establishes the constraint."],
        })
        adapter = ScriptedAdapter(REQUIREMENTS, proposal(), [missing], corrected, [])
        result = self.run_pipeline(adapter)
        self.assertEqual(len(result["groups"]["mappings"]), 2)
        self.assertEqual(result["review_status"], "needs_review")

    def test_persistent_findings_stop_at_repair_limit(self):
        adapter = ScriptedAdapter(REQUIREMENTS, proposal(), [issue()], proposal(), [issue()])
        result = self.run_pipeline(adapter)
        self.assertEqual(len(adapter.calls), 5)
        self.assertEqual(result["issues"], [issue()])
        self.assertEqual(result["review_status"], "needs_review")

    def test_zero_repair_rounds_and_keep_findings_do_not_trigger_repair(self):
        for rounds, finding in ((0, issue()), (1, issue(action="keep"))):
            with self.subTest(rounds=rounds):
                adapter = ScriptedAdapter(REQUIREMENTS, proposal(), [finding])
                result = self.run_pipeline(adapter, max_repair_rounds=rounds)
                self.assertEqual(len(adapter.calls), 3)
                self.assertEqual(result["review_status"], "needs_review")
                self.assertEqual(result["issues"], [finding])

    def test_programmatic_check_detects_mutated_instances(self):
        groups = self.validate_groups(proposal())
        groups.mappings[0].source_excerpts[0].quote = "Tampered after validation"
        with self.assertRaises(ValidationError):
            q.check_page_ids_and_quotes(groups, PAGES)


if __name__ == "__main__":
    unittest.main()
