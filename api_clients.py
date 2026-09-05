"""Isolated OpenAI embedding and structured-generation clients for V3."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from config import CHAT_MODEL_OPTIONS, SETTINGS
from data_models import KnowledgeArtifact, is_knowledge_artifact


LLM_MODEL = SETTINGS.models.llm_model
EMBEDDING_MODEL = SETTINGS.models.embedding_model
EMBEDDING_DIMENSION = SETTINGS.models.embedding_dimension
MAX_OUTPUT_TOKENS = SETTINGS.models.chatbot_output_tokens
REQUEST_TIMEOUT_SECONDS = SETTINGS.models.request_timeout_seconds
EMBEDDING_BATCH_SIZE = SETTINGS.models.embedding_batch_size


def _load_local_environment() -> None:
    """Load local development values without overriding exported variables."""

    load_dotenv(Path(__file__).resolve().with_name(".env.local"), override=False)


def _api_key() -> str:
    """Read the API key from the environment without logging secret material."""

    _load_local_environment()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Set it in the process environment "
            "or in new-V3/.env.local before using the production API clients."
        )
    return key


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """Construct one bounded OpenAI client per process."""

    return OpenAI(
        api_key=_api_key(),
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=1,
    )


def generate_embeddings(
    texts: list[str],
    *,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """Generate embeddings in bounded batches while preserving input order."""

    if not isinstance(texts, list) or not texts:
        raise ValueError("texts must be a non-empty list")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("every embedding input must be a non-empty string")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = [text.strip() for text in texts[start : start + batch_size]]
        raw_response = _client().embeddings.with_raw_response.create(
            model=EMBEDDING_MODEL,
            input=batch,
            encoding_format="float",
            dimensions=EMBEDDING_DIMENSION,
        )
        payload: Any = raw_response.http_response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(batch):
            raise RuntimeError("embedding API returned an unexpected result count")
        ordered = sorted(data, key=lambda item: int(item.get("index", -1)))
        for item in ordered:
            raw_embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(raw_embedding, list):
                raise RuntimeError("embedding API response omitted a numeric vector")
            embedding = [float(value) for value in raw_embedding]
            if len(embedding) != EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"embedding API returned dimension {len(embedding)}; "
                    f"expected {EMBEDDING_DIMENSION}"
                )
            embeddings.append(embedding)
    return embeddings


def generate_embedding(text: str) -> list[float]:
    """Compatibility helper for a single query embedding."""

    return generate_embeddings([text])[0]


def _synthesis_json_schema(
    evidence_ids: list[str], *, max_claims: int = 5
) -> dict[str, object]:
    """Build the exact model-output schema with request-local handle enums."""

    return {
        "type": "json_schema",
        "name": "v3_synthesis_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "preamble": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": [
                        "answered",
                        "partially_answered",
                        "insufficient_evidence",
                    ],
                },
                "claims": {
                    "type": "array",
                    "maxItems": max_claims,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "text": {"type": "string"},
                            "evidence_ids": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "string",
                                    "enum": evidence_ids,
                                },
                            },
                            "supporting_spans": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "text",
                            "evidence_ids",
                            "supporting_spans",
                        ],
                    },
                },
                "unsupported_facets": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["preamble", "status", "claims", "unsupported_facets"],
        },
    }


def call_llm(
    system_prompt: str,
    user_prompt: str,
    artifacts: dict[str, KnowledgeArtifact],
    *,
    model: str | None = None,
    max_claims: int = 5,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> str:
    """Call the LLM and return strictly schema-constrained JSON text."""

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("user_prompt must be a non-empty string")
    if not isinstance(artifacts, dict) or any(
        not isinstance(handle, str)
        or not is_knowledge_artifact(artifact)
        for handle, artifact in artifacts.items()
    ):
        raise TypeError("artifacts must be a dict[str, KnowledgeArtifact]")
    if not artifacts:
        raise ValueError("at least one artifact is required for LLM synthesis")
    selected_model = model or LLM_MODEL
    if selected_model not in CHAT_MODEL_OPTIONS:
        raise ValueError(f"Unsupported chatbot model: {selected_model}")
    if (
        not isinstance(max_claims, int)
        or isinstance(max_claims, bool)
        or not 1 <= max_claims <= 5
    ):
        raise ValueError("max_claims must be an integer from 1 through 5")

    return call_structured_llm(
        system_prompt,
        user_prompt,
        _synthesis_json_schema(list(artifacts), max_claims=max_claims),
        model=selected_model,
        max_output_tokens=max_output_tokens,
    )


def call_structured_llm(
    system_prompt: str,
    user_prompt: str,
    response_format: dict[str, object],
    *,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    model: str | None = None,
) -> str:
    """Call Chat Completions with a caller-supplied strict JSON schema.

    The raw HTTP response is parsed deliberately so canonical validation remains
    application-owned. This also avoids importing the SDK's generic Responses
    parsing models, which are incompatible with some Pydantic runtime releases.
    """

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("user_prompt must be a non-empty string")
    if not isinstance(response_format, dict):
        raise TypeError("response_format must be a JSON-schema format dictionary")
    if (
        not isinstance(max_output_tokens, int)
        or isinstance(max_output_tokens, bool)
        or max_output_tokens <= 0
    ):
        raise ValueError("max_output_tokens must be a positive integer")

    if response_format.get("type") != "json_schema":
        raise ValueError("response_format must have type='json_schema'")
    schema_payload = {
        key: value for key, value in response_format.items() if key != "type"
    }
    raw_response = _client().chat.completions.with_raw_response.create(
        model=model or LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=max_output_tokens,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": schema_payload,
        },
    )
    payload: Any = raw_response.http_response.json()
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM API response omitted completion choices")
    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    finish_reason = (
        first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
    )
    refusal = message.get("refusal") if isinstance(message, dict) else None
    if isinstance(refusal, str) and refusal.strip():
        raise RuntimeError(f"LLM refused the structured request: {refusal.strip()}")
    if finish_reason == "length":
        raise RuntimeError(
            f"LLM structured output was truncated at {max_output_tokens} tokens"
        )
    if finish_reason == "content_filter":
        raise RuntimeError("LLM structured output was blocked by the content filter")
    if finish_reason not in {None, "stop"}:
        raise RuntimeError(f"LLM structured output ended unexpectedly: {finish_reason}")
    output = message.get("content", "").strip() if isinstance(message, dict) else ""
    if not output:
        raise RuntimeError("LLM API returned an empty structured response")
    return output
