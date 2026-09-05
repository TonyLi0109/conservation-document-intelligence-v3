"""Regression tests for the latency and simple-answer issues found in review."""

from data_models import KnowledgeArtifact
from database import KnowledgeStore
from main import _direct_existence_answer
import wiki_compiler
from wiki_compiler import (
    EXTRACTIVE_MODEL_NAME,
    generate_extractive_wiki_concept,
    generate_wiki_concept,
)


MISSOURI_CARP_EVIDENCE = (
    "Some of the most well-known aquatic invasive species in Missouri include "
    "invasive carp such as bighead, silver, and black carp."
)


def _store(tmp_path) -> KnowledgeStore:
    store = KnowledgeStore(tmp_path / "test.db")
    store.ingest_chunk(
        KnowledgeArtifact(
            document_id="DOC036",
            title="2022 Missouri Comprehensive Conservation Strategy",
            page_number="88",
            original_text_chunk=MISSOURI_CARP_EVIDENCE,
        ),
        [1.0, 0.0],
    )
    return store


def test_wiki_page_is_generated_locally_then_loaded_from_cache(tmp_path) -> None:
    store = _store(tmp_path)
    first = generate_extractive_wiki_concept("Invasive carp", store)
    second = generate_extractive_wiki_concept("Invasive carp", store)

    assert first["model_name"] == EXTRACTIVE_MODEL_NAME
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["concept"]["important_facts"] == [MISSOURI_CARP_EVIDENCE]
    store.close()


def test_simple_missouri_carp_question_gets_direct_grounded_yes(tmp_path) -> None:
    store = _store(tmp_path)
    result = _direct_existence_answer(
        "are there invasive carps in Missouri",
        store,
    )

    assert result is not None
    answer, preamble, sources = result
    assert answer.startswith("- Yes.")
    assert "DOC036" in answer
    assert preamble == ""
    assert [source.document_id for source in sources] == ["DOC036"]
    store.close()


def test_failed_ai_refresh_keeps_pre_generated_page(tmp_path, monkeypatch) -> None:
    store = _store(tmp_path)
    original = generate_extractive_wiki_concept("Invasive carp", store)

    def timeout(*_args, **_kwargs):
        raise TimeoutError("Request timed out")

    monkeypatch.setattr(wiki_compiler, "call_structured_llm", timeout)
    refreshed = generate_wiki_concept(
        "Invasive carp", store, force_refresh=True
    )

    assert refreshed["knowledge_id"] == original["knowledge_id"]
    assert refreshed["concept"] == original["concept"]
    assert refreshed["cached"] is True
    assert refreshed["refresh_error"] == "Request timed out"
    store.close()
