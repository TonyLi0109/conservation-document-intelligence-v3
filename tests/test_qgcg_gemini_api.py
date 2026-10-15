"""No-network checks for the Gemini-only API exception and exact response capture."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import qgcg_gemini_api as api
from scripts import run_qgcg_pilot as pilot


class GeminiApiTests(unittest.TestCase):
    def client(self):
        with patch.object(api, "load_api_key", return_value=("private-test-key", "test")):
            return api.GeminiAPI(Path("."))

    def request(self, **kwargs):
        return pilot.ProviderRequest("fixture", 1, "Q1", "E1", "model_c", api.API_MODEL,
                                     "Exact system prompt", '{"question":"Exact?"}',
                                     {"type": "array", "items": {"type": "object"}}, None, 10, **kwargs)

    def test_only_explicit_gemini_api_is_permitted(self):
        stub = lambda _: None
        providers = {role: pilot.ProviderSpec(model, stub, stub, "subscription", "1")
                     for role, model in pilot.DEFAULT_MODELS.items()}
        providers["model_c"] = replace(providers["model_c"], transport_name="GeminiDeveloperAPI",
                                        billing_mode="developer_api", settings={"api_authorized": True})
        pilot.validate_providers(providers)
        self.assertEqual(pilot.billing_policy(providers), "gemini_api_only_gpt_claude_subscription_no_fallback")
        providers["model_a"] = replace(providers["model_a"], billing_mode="developer_api")
        with self.assertRaises(ValueError):
            pilot.validate_providers(providers)
        providers["model_a"] = replace(providers["model_a"], billing_mode="subscription")
        providers["model_c"] = replace(providers["model_c"], settings={})
        with self.assertRaises(ValueError):
            pilot.validate_providers(providers)

    def test_request_preserves_raw_before_parse_and_never_archives_key(self):
        client = self.client()
        events, artifacts = [], {}
        request = self.request(artifact_writer=lambda name, data: artifacts.update({name: data}),
                               raw_logger=lambda kind, text: events.append((kind, text)),
                               event_logger=lambda kind, **data: events.append((kind, data)))
        body = b' {"candidates":[{"content":{"parts":[{"text":" [ ] "}]},"finishReason":"STOP"}],"modelVersion":"gemini-3.1-pro-preview"} '
        with patch.object(client, "_http", return_value=(200, body, {"Content-Type": "application/json"})), \
                patch.object(api.json, "loads", side_effect=AssertionError("Premature parsing")):
            response = client.request(request)
        self.assertEqual(response.raw_text, body.decode())
        self.assertEqual(artifacts["http_response_body.bin"], body)
        self.assertEqual(events[0][0], "api_request_started")
        self.assertEqual(events[1], ("provider_http_body", body.decode()))
        self.assertNotIn(b"private-test-key", b"".join(artifacts.values()))
        decoded = client.decode(response)
        self.assertEqual(decoded.text, " [ ] ")
        payload = json.loads(artifacts["http_request_body.json"])
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], request.system_prompt)
        self.assertEqual(payload["generationConfig"]["thinkingConfig"], {"thinkingLevel": "low"})
        self.assertNotIn("tools", payload)

    def test_http_retry_errors_keep_original_body(self):
        client = self.client()
        raw = []
        with patch.object(client, "_http", return_value=(429, b'{"error":"quota"}', {"Retry-After": "9"})):
            with self.assertRaises(pilot.ProviderFailure) as caught:
                client.request(self.request(raw_logger=lambda kind, text: raw.append(text)))
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(caught.exception.retry_after_seconds, 9)
        self.assertEqual(caught.exception.raw_response.exit_code, 429)
        self.assertEqual(raw, ['{"error":"quota"}'])

    def test_decode_rejects_incomplete_response_and_tools(self):
        value = {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "[]"}]}}],
                 "modelVersion": api.API_MODEL}
        with self.assertRaises(pilot.ProviderFailure):
            api.GeminiAPI.decode(pilot.RawProviderResponse(json.dumps(value)))
        value["candidates"][0]["finishReason"] = "STOP"
        value["candidates"][0]["content"]["parts"] = [{"functionCall": {"name": "fetch"}}]
        with self.assertRaises(pilot.ProviderFailure):
            api.GeminiAPI.decode(pilot.RawProviderResponse(json.dumps(value)))

    def test_reads_only_named_key_without_mutating_environment(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(api.os.environ, {}, clear=True):
            root = Path(folder)
            (root / ".env.local").write_text('UNRELATED=hidden\nGEMINI_API_KEY="private-test-key" # local\n', encoding="utf-8")
            key, source = api.load_api_key(root)
            self.assertEqual(key, "private-test-key")
            self.assertNotIn("GEMINI_API_KEY", api.os.environ)
            self.assertNotIn(key, source)


if __name__ == "__main__":
    unittest.main()
