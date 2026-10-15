"""Explicit Gemini-only developer API transport; GPT/Claude remain subscriptions.

REST reference: https://ai.google.dev/api/generate-content
No SDK retries, redirects, alternate endpoints, tools, or automatic model fallback.
The existing audit adapter persists provider_raw before decode and model_text before
Pydantic validation. HTTP response bytes and non-secret metadata are also archived.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
from uuid import uuid4

from scripts.run_qgcg_pilot import (
    DecodedResponse, ProviderFailure, ProviderRequest, ProviderSpec,
    RawProviderResponse, json_bytes, sha256, utc_now, write_durable,
)
from scripts.qgcg_subscription_transports import subscription_providers, _user_input

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models/"
API_MODEL = "gemini-3.1-pro-preview"
TRANSPORT_VERSION = "gemini-rest-v1beta-qgcg-1.0"


def load_api_key(data_root: Path):
    """Read just the named credential, without exporting or snapshotting secrets."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(name, "").strip():
            return os.environ[name].strip(), "environment:" + name
    path = data_root / ".env.local"
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            match = re.fullmatch(r"\s*(?:export\s+)?GEMINI_API_KEY\s*=\s*(.*?)\s*", line)
            if not match:
                continue
            value = match[1]
            if value.startswith(('"', "'")):
                quote = value[0]
                end = value.find(quote, 1)
                if end < 0 or value[end + 1:].strip() and not value[end + 1:].lstrip().startswith("#"):
                    raise ProviderFailure("Malformed GEMINI_API_KEY entry in .env.local")
                value = value[1:end]
            else:
                value = value.split(" #", 1)[0].strip()
            if value:
                return value, "local_config:GEMINI_API_KEY"
    raise ProviderFailure("Gemini API requires GEMINI_API_KEY or GOOGLE_API_KEY; no subscription/API fallback")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward the credential to a redirected host.


class GeminiAPI:
    def __init__(self, data_root: Path, model=API_MODEL):
        if model != API_MODEL:
            raise ProviderFailure(f"This experiment pins Gemini API model {API_MODEL}")
        self.model = model
        self._key, self.credential_source = load_api_key(data_root)
        self._opener = urllib.request.build_opener(NoRedirect())

    def _http(self, *, body: bytes | None, timeout: float):
        url = API_ROOT + self.model + (":generateContent" if body is not None else "")
        req = urllib.request.Request(url, data=body, method="POST" if body is not None else "GET",
                                     headers={"x-goog-api-key": self._key, "Content-Type": "application/json"})
        try:
            with self._opener.open(req, timeout=timeout) as response:
                return response.status, response.read(), response.headers
        except urllib.error.HTTPError as error:
            with error:
                return error.code, error.read(), error.headers
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            # Exception text may contain request details; keep only its class.
            raise ProviderFailure(f"Gemini API transport failed ({type(error).__name__}); no response body",
                                  retryable=True) from None

    def preflight(self, output_root: Path):
        folder = output_root / "preflight" / (uuid4().hex + "-gemini-api")
        request_meta = {"timestamp_utc": utc_now(), "method": "GET", "url": API_ROOT + self.model,
                        "inference_started": False, "credential_source": self.credential_source}
        write_durable(folder / "request.json", json_bytes(request_meta))
        status, body, headers = self._http(body=None, timeout=30)
        write_durable(folder / "response_body.bin", body)  # Persist before JSON decode.
        write_durable(folder / "response.json", json_bytes({"timestamp_utc": utc_now(), "http_status": status,
                      "body_sha256": sha256(body), "content_type": headers.get("Content-Type")}))
        if status != 200:
            raise ProviderFailure(f"Gemini model-access preflight returned HTTP {status}; see saved preflight response")
        metadata = json.loads(body)
        if metadata.get("name") != "models/" + self.model or "generateContent" not in metadata.get("supportedGenerationMethods", []):
            raise ProviderFailure("Gemini API model metadata differs from the requested experiment model")
        return {"authentication": "Google developer API key", "credential_source": self.credential_source,
                "endpoint": API_ROOT + self.model + ":generateContent", "model_metadata": metadata,
                "preflight_directory": str(folder.resolve()), "preflight_body_sha256": sha256(body),
                "thinking_level": "low", "temperature": 1.0, "max_output_tokens": 16384,
                "tools": "none", "api_authorized": True, "api_fallback": False,
                "capture_boundary": "exact HTTP response bytes plus decoded model text",
                "http_retries": "only outer audited runner retries; no SDK retries",
                "comparison_note": "User authorized Gemini API for stress checks; pilot used gemini-3.1-pro-low account CLI. Same family/Low setting does not prove identical backends or prompt wrappers.",
                "documentation": ["https://ai.google.dev/api/generate-content",
                                  "https://ai.google.dev/gemini-api/docs/generate-content/gemini-3"]}

    def request(self, request: ProviderRequest):
        if request.role != "model_c" or request.model != self.model:
            raise ProviderFailure("Gemini API is authorized only for the pinned Pass 3 model")
        generation = {"responseMimeType": "application/json", "responseJsonSchema": request.json_schema,
                      "thinkingConfig": {"thinkingLevel": "low"}, "temperature": 1.0,
                      "candidateCount": 1, "maxOutputTokens": 16384}
        if request.seed is not None:
            generation["seed"] = request.seed
        body = json_bytes({"systemInstruction": {"parts": [{"text": request.system_prompt}]},
                           "contents": [{"role": "user", "parts": [{"text": _user_input(request)}]}],
                           "generationConfig": generation})
        if request.artifact_writer:
            request.artifact_writer("http_request_body.json", body)
            request.artifact_writer("http_request_metadata.json", json_bytes({
                "method": "POST", "url": API_ROOT + self.model + ":generateContent",
                "authentication_header": "x-goog-api-key (value omitted)", "body_sha256": sha256(body),
                "transport_version": TRANSPORT_VERSION}))
        if request.event_logger:
            request.event_logger("api_request_started", endpoint=API_ROOT + self.model + ":generateContent",
                                 body_sha256=sha256(body), billing_mode="developer_api")
        status, raw_bytes, headers = self._http(body=body, timeout=request.timeout_seconds)
        raw = RawProviderResponse(raw_bytes.decode("utf-8", errors="replace"), exit_code=0 if status == 200 else status,
                                  capture_method="gemini_developer_api_http")
        # Capture immediately before further persistence or parsing can fail.
        if request.raw_logger:
            request.raw_logger("provider_http_body", raw.raw_text)
        if request.artifact_writer:
            request.artifact_writer("http_response_body.bin", raw_bytes)
            request.artifact_writer("http_response_metadata.json", json_bytes({
                "status": status, "content_type": headers.get("Content-Type"),
                "retry_after": headers.get("Retry-After"), "body_sha256": sha256(raw_bytes)}))
        if status != 200:
            retry_after = headers.get("Retry-After", "")
            delay = float(retry_after) if re.fullmatch(r"\d+(?:\.\d+)?", retry_after) else None
            raise ProviderFailure(f"Gemini generateContent returned HTTP {status}; raw body saved",
                                  retryable=status in {408, 429, 500, 502, 503, 504}, raw_response=raw,
                                  retry_after_seconds=delay)
        return raw

    @staticmethod
    def decode(response: RawProviderResponse):
        value = json.loads(response.raw_text)
        candidates = value.get("candidates", [])
        if value.get("error") or len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
            raise ProviderFailure("Gemini did not return one complete STOP candidate; inspect raw response")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts or any(set(p) - {"text", "thought", "thoughtSignature"} for p in parts):
            raise ProviderFailure("Gemini returned unexpected tool or non-text content")
        text = "".join(p["text"] for p in parts if not p.get("thought") and isinstance(p.get("text"), str))
        if not text or not isinstance(value.get("modelVersion"), str):
            raise ProviderFailure("Gemini response lacks model text or modelVersion")
        return DecodedResponse(text, value["modelVersion"], backend_revision=value["modelVersion"],
                               provider_request_id=value.get("responseId"), finish_reason="STOP",
                               identity_source="google_generate_content_modelVersion")


def mixed_providers(data_root: Path, output_root: Path):
    providers = subscription_providers(data_root, gemini_model="gemini-3.1-pro-low", include_gemini=False)
    client = GeminiAPI(data_root)
    state = client.preflight(output_root)
    providers["model_c"] = ProviderSpec(API_MODEL, client.request, client.decode, "GeminiDeveloperAPI",
                                        TRANSPORT_VERSION, billing_mode="developer_api", seed_supported=True,
                                        settings=state, implementation_files=(Path(__file__).resolve(),),
                                        retry_invalid_output=True)
    return providers
