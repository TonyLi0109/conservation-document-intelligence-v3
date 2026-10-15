"""Offline checks for account-only routing and raw CLI response preservation."""
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import qgcg_subscription_transports as t
from scripts.run_qgcg_pilot import ProviderRequest, ProviderFailure, RawProviderResponse


class SubscriptionTransportTests(unittest.TestCase):
    def test_api_keys_and_alternate_routes_are_removed_without_mutating_parent(self):
        with patch.dict(t.os.environ, {"OPENAI_API_KEY": "test", "ANTHROPIC_API_KEY": "test",
                                      "GEMINI_API_KEY": "test", "AGY_ADC_AUTH": "true",
                                      "GOOGLE_APPLICATION_CREDENTIALS": "test", "PATH": "preserve"}, clear=True):
            self.assertEqual(t.account_environment(), {"PATH": "preserve"})
            self.assertIn("OPENAI_API_KEY", t.os.environ)

    def test_codex_decodes_final_message_without_reformatting(self):
        raw = ' [\n {"requirement":"x"}\n] '
        decoded = t.CodexSubscription.decode(RawProviderResponse("events", "model: gpt-5.6-sol\n", final_text=raw))
        self.assertEqual(decoded.text, raw)
        self.assertEqual(decoded.reported_model, "gpt-5.6-sol")
        with self.assertRaises(ProviderFailure):
            t.CodexSubscription.decode(RawProviderResponse("events", "no model", final_text=raw))

    def test_claude_requires_structured_output_and_one_reported_model(self):
        envelope = {"num_turns": 2, "is_error": False, "modelUsage": {"claude-sonnet-5": {}},
                    "result": "Done", "structured_output": {"mappings": []}, "session_id": "fixture"}
        decoded = t.ClaudeSubscription.decode(RawProviderResponse(json.dumps(envelope)))
        self.assertEqual(decoded.text, '{"mappings": []}')
        self.assertEqual(decoded.reported_model, "claude-sonnet-5")
        envelope["num_turns"] = 3  # CLI counts the additional structured submission.
        self.assertEqual(t.ClaudeSubscription.decode(RawProviderResponse(json.dumps(envelope))).text,
                         '{"mappings": []}')
        envelope["num_turns"] = 4
        self.assertEqual(t.ClaudeSubscription.decode(RawProviderResponse(json.dumps(envelope))).reported_model,
                         "claude-sonnet-5")
        envelope["num_turns"] = 0
        with self.assertRaises(ProviderFailure):
            t.ClaudeSubscription.decode(RawProviderResponse(json.dumps(envelope)))
        envelope["num_turns"] = 2
        envelope["modelUsage"]["another-model"] = {}
        with self.assertRaises(ProviderFailure):
            t.ClaudeSubscription.decode(RawProviderResponse(json.dumps(envelope)))

    def test_claude_rejects_unstructured_text_and_preserves_structured_json_bytes(self):
        prefix = '{"num_turns": 1, "modelUsage": {"claude-sonnet-5": {}}, '
        with self.assertRaises(ProviderFailure):
            t.ClaudeSubscription.decode(RawProviderResponse(prefix + '"result":"```json...```"}'))
        raw = prefix + '"structured_output": { "mappings" : [ ] }}'
        self.assertEqual(t.ClaudeSubscription.decode(RawProviderResponse(raw)).text, '{ "mappings" : [ ] }')

    def test_json_member_extraction_preserves_lexical_json(self):
        original = '{"before":[{"x":"a,b:c"}], "result": {"structured_output": {"response": [  {"x":"\\u4e2d"}  ] }}}'
        self.assertEqual(t.json_member_text(original, ("result", "structured_output", "response")),
                         '[  {"x":"\\u4e2d"}  ]')
        with self.assertRaises(ValueError):
            t.json_member_text(original, ("unknown",))

    def test_gemini_decode_checks_tools_conversation_and_success(self):
        init = {"event": "init", "conversation_id": "fixture", "init": {"model": "gemini-3.1-pro-low"}}
        result = {"event": "result", "result": {"status": "SUCCESS", "num_turns": 1,
                  "conversation_id": "fixture", "structured_output": {"response": []}}}
        raw = json.dumps(init) + "\n" + json.dumps(result) + "\n"
        decoded = t.GeminiSubscription.decode(RawProviderResponse(raw))
        self.assertEqual(decoded.text, "[]")
        self.assertEqual(decoded.reported_model, "gemini-3.1-pro-low")
        tool = {"event": "step_update", "step_update": {"conversation_id": "fixture",
                 "step_type": "tool", "tool_name": "read_file"}}
        with self.assertRaises(ProviderFailure):
            t.GeminiSubscription.decode(RawProviderResponse(json.dumps(init)+"\n"+json.dumps(tool)+"\n"+json.dumps(result)))

    def test_gemini_isolation_disables_extra_credits_and_custom_providers(self):
        with tempfile.TemporaryDirectory() as folder:
            profile, env = t.GeminiSubscription.isolated_profile(Path(folder))
            settings = json.loads((profile / ".gemini/antigravity-cli/settings.json").read_text())
            self.assertIs(settings["useG1Credits"], False)
            self.assertNotIn("modelProvider", settings)
            self.assertEqual(env["USERPROFILE"], str(profile))

    def test_gemini_stream_is_logged_before_init_parsing_and_input_release(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            captured = []
            init = {"event": "init", "init": {"model": "gemini-3.1-pro-low", "agent": t.AGENT_NAME,
                     "permission_mode": "request-review", "cwd": str(workspace)}}
            line = json.dumps(init) + "\n"
            class InputBuffer(BytesIO):
                def write(inner, data):
                    self.assertEqual(captured[0], ("provider_stream", line))
                    return super().write(data)
            class Process:
                stdin = InputBuffer()
                stdout = BytesIO(line.encode())
                def wait(self, timeout=None): return 0
                def poll(self): return 0
            request = ProviderRequest("c", 1, "Q2", "E1", "model_c", "gemini-3.1-pro-low",
                                      "system", "{}", {}, None, 10,
                                      raw_logger=lambda kind, text: captured.append((kind, text)))
            with patch.object(t.subprocess, "Popen", return_value=Process()):
                response = t.GeminiSubscription._stream(request, ["fake-cli"], workspace, {}, "input\n")
            self.assertEqual(response.raw_text, line)


if __name__ == "__main__":
    unittest.main()
