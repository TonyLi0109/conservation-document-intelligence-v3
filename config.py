"""Small, centralized runtime configuration for Conservation Intelligence V3.

Environment variables permit deployment overrides without introducing a heavy
configuration framework. Provenance-bearing data remains in the source catalog,
not in runtime settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


V3_ROOT = Path(__file__).resolve().parent

# Deliberately bounded to models compatible with this application's existing
# Chat Completions + strict JSON-schema integration.  UI input must never be
# passed through as an arbitrary API model identifier.
CHAT_MODEL_OPTIONS: tuple[str, ...] = (
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
)


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be numeric") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class ModelSettings:
    llm_model: str = field(
        default_factory=lambda: os.environ.get("V3_LLM_MODEL", "gpt-4.1-mini")
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "V3_EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )
    embedding_dimension: int = field(
        default_factory=lambda: _positive_int("V3_EMBEDDING_DIMENSION", 1_536)
    )
    chatbot_output_tokens: int = field(
        default_factory=lambda: _positive_int("V3_CHATBOT_OUTPUT_TOKENS", 2_500)
    )
    request_timeout_seconds: float = field(
        default_factory=lambda: _positive_float("V3_REQUEST_TIMEOUT_SECONDS", 30.0)
    )
    embedding_batch_size: int = field(
        default_factory=lambda: _positive_int("V3_EMBEDDING_BATCH_SIZE", 100)
    )


@dataclass(frozen=True, slots=True)
class ChunkingSettings:
    target_words: int = field(
        default_factory=lambda: _positive_int("V3_CHUNK_TARGET_WORDS", 750)
    )
    overlap_words: int = field(
        default_factory=lambda: _positive_int("V3_CHUNK_OVERLAP_WORDS", 100)
    )
    page_timeout_seconds: int = field(
        default_factory=lambda: _positive_int("V3_PAGE_TIMEOUT_SECONDS", 60)
    )

    def __post_init__(self) -> None:
        if self.overlap_words >= self.target_words:
            raise RuntimeError("V3_CHUNK_OVERLAP_WORDS must be smaller than target words")


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    top_k: int = field(default_factory=lambda: _positive_int("V3_TOP_K", 5))


@dataclass(frozen=True, slots=True)
class WikiSettings:
    top_k: int = field(default_factory=lambda: _positive_int("V3_WIKI_TOP_K", 5))
    output_tokens: int = field(
        default_factory=lambda: _positive_int("V3_WIKI_OUTPUT_TOKENS", 3_000)
    )
    max_spans_per_artifact: int = field(
        default_factory=lambda: _positive_int("V3_WIKI_SPANS_PER_ARTIFACT", 8)
    )
    compiler_version: str = field(
        default_factory=lambda: os.environ.get("V3_WIKI_COMPILER_VERSION", "v3.1")
    )


@dataclass(frozen=True, slots=True)
class StorageSettings:
    database_path: Path = V3_ROOT / "data" / "corpus.db"
    source_catalog_path: Path = V3_ROOT / "data" / "source_catalog.csv"


@dataclass(frozen=True, slots=True)
class V3Settings:
    models: ModelSettings = field(default_factory=ModelSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    wiki: WikiSettings = field(default_factory=WikiSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)


SETTINGS = V3Settings()
