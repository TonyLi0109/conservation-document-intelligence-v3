"""Regression tests for the latency and simple-answer issues found in review."""

from types import SimpleNamespace

import api_clients
from data_models import KnowledgeArtifact
from database import KnowledgeStore, prepare_runtime_database
from main import (
    _direct_existence_answer,
    _direct_native_status_answer,
    _ensure_polar_answer_prefix,
)
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
ZEBRA_EVIDENCE = "Zebra mussels can disrupt aquatic food webs and native mussels."
CLIMATE_EVIDENCE = "Climate change is expected to affect habitats across the state."
CARP_GLOSSARY = (
    "invasive carp A collective term for bighead carp, black carp, grass carp, "
    "and silver carp. Also known as Asian carp. invasive species With regard to "
    "a particular ecosystem, a non-native organism whose introduction causes or "
    "is likely to cause economic or environmental harm or harm to human, animal, "
    "or plant health (3 CFR 13751)."
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


def test_runtime_database_copy_accepts_wiki_writes(tmp_path) -> None:
    seed_store = _store(tmp_path)
    seed_store.close()
    runtime_path = prepare_runtime_database(
        tmp_path / "test.db", tmp_path / "runtime"
    )

    runtime_store = KnowledgeStore(runtime_path)
    page = generate_extractive_wiki_concept("Invasive carp", runtime_store)

    assert runtime_path != tmp_path / "test.db"
    assert page["knowledge_id"]
    assert runtime_store.get_compiled_concept("Invasive carp") is not None
    runtime_store.close()


def test_generated_polar_answer_is_normalized_to_explicit_yes() -> None:
    envelope = {
        "status": "answered",
        "claims": [{
            "text": "Invasive carp are present in Missouri.",
            "evidence_ids": ["K1"],
            "supporting_spans": [MISSOURI_CARP_EVIDENCE],
        }],
        "unsupported_facets": [],
    }

    _ensure_polar_answer_prefix("Are there invasive carps in Missouri?", envelope)

    assert envelope["claims"][0]["text"] == (
        "Yes. Invasive carp are present in Missouri."
    )


def test_title_supplies_missouri_scope_for_zebra_mussel_answer(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "zebra.db")
    store.ingest_chunk(
        KnowledgeArtifact(
            document_id="DOC036",
            title="2022 Missouri Comprehensive Conservation Strategy",
            page_number="298",
            original_text_chunk=ZEBRA_EVIDENCE,
        ),
        [1.0, 0.0],
    )

    result = _direct_existence_answer("Are zebra mussels found in Missouri?", store)

    assert result is not None
    assert result[0].startswith("- Yes. Zebra mussels are found in Missouri.")
    assert [source.document_id for source in result[2]] == ["DOC036"]
    store.close()


def test_curly_possessive_and_title_scope_for_climate_question(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "climate.db")
    store.ingest_chunk(
        KnowledgeArtifact(
            document_id="DOC036",
            title="2022 Missouri Comprehensive Conservation Strategy",
            page_number="132",
            original_text_chunk=CLIMATE_EVIDENCE,
        ),
        [1.0, 0.0],
    )

    result = _direct_existence_answer(
        "Is climate change discussed in Missouri’s conservation strategy?", store
    )

    assert result is not None
    assert result[0].startswith(
        "- Yes. Climate change is discussed in Missouri’s conservation strategy."
    )
    assert [source.document_id for source in result[2]] == ["DOC036"]
    store.close()


def test_cached_store_proxy_survives_streamlit_hot_reload(tmp_path) -> None:
    store = _store(tmp_path)

    class CachedStoreProxy:
        """Behaves like a store but deliberately fails concrete isinstance checks."""

        def __getattr__(self, name):
            return getattr(store, name)

    from main import ask_chatbot_with_context

    answer, preamble, sources = ask_chatbot_with_context(
        "Are there invasive carps in Missouri?",
        CachedStoreProxy(),
    )

    assert answer.startswith("- Yes.")
    assert preamble == ""
    assert [source.document_id for source in sources] == ["DOC036"]
    store.close()


def test_native_question_reaches_grounded_negative_synthesis(tmp_path, monkeypatch) -> None:
    import json
    import main

    evidence = "Invasive carp are nonnative fish managed in Missouri waterways."
    store = KnowledgeStore(tmp_path / "native.db")
    store.ingest_chunk(
        KnowledgeArtifact(
            document_id="DOC036",
            title="2022 Missouri Comprehensive Conservation Strategy",
            page_number="298",
            original_text_chunk=evidence,
        ),
        [1.0, 0.0],
    )
    monkeypatch.setattr(main, "generate_embedding", lambda _question: [1.0, 0.0])
    monkeypatch.setattr(
        main,
        "call_llm",
        lambda *_args, **_kwargs: json.dumps({
            "preamble": "",
            "status": "answered",
            "claims": [{
                "text": "No. Invasive carp are not native to Missouri.",
                "evidence_ids": ["K1"],
                "supporting_spans": [evidence],
            }],
            "unsupported_facets": [],
        }),
    )

    class ReloadedStoreProxy:
        """Return structurally valid artifacts from a previous module identity."""

        def retrieve(self, *args, **kwargs):
            return [
                SimpleNamespace(
                    document_id=item.document_id,
                    title=item.title,
                    page_number=item.page_number,
                    original_text_chunk=item.original_text_chunk,
                    source_url=item.source_url,
                    printed_page_label=item.printed_page_label,
                )
                for item in store.retrieve(*args, **kwargs)
            ]

        def retrieve_document_matches(self, *args, **kwargs):
            return []

    answer, preamble, sources = main.ask_chatbot_with_context(
        "Are invasive carp native to Missouri?", ReloadedStoreProxy()
    )

    assert answer.startswith("- No.")
    assert preamble == ""
    assert [source.document_id for source in sources] == ["DOC036"]
    store.close()


def test_api_client_accepts_structural_artifact_after_reload(monkeypatch) -> None:
    legacy_artifact = SimpleNamespace(
        document_id="DOC036",
        title="2022 Missouri Comprehensive Conservation Strategy",
        page_number="298",
        original_text_chunk="Invasive carp are nonnative fish in Missouri.",
        source_url=None,
        printed_page_label=None,
    )
    monkeypatch.setattr(
        api_clients,
        "call_structured_llm",
        lambda *_args, **_kwargs: "{}",
    )

    result = api_clients.call_llm(
        "System instructions",
        "User question",
        {"K1": legacy_artifact},
    )

    assert result == "{}"


def test_native_status_uses_definition_and_missouri_classification(tmp_path) -> None:
    store = KnowledgeStore(tmp_path / "native_direct.db")
    store.ingest_chunk(
        KnowledgeArtifact(
            document_id="DOC012",
            title="Invasive Carp Strategic Science Plan",
            page_number="1",
            original_text_chunk=CARP_GLOSSARY,
        ),
        [1.0, 0.0],
    )
    store.ingest_chunk(
        KnowledgeArtifact(
            document_id="DOC001",
            title="Missouri State Wildlife Action Plan",
            page_number="171",
            original_text_chunk=(
                "Aquatic invasive species in Missouri include zebra mussels and "
                "Asian carp."
            ),
        ),
        [1.0, 0.0],
    )

    result = _direct_native_status_answer(
        "Are invasive carp native to Missouri?", store
    )

    assert result is not None
    assert result[0].startswith("- No. Invasive carp are not native to Missouri.")
    assert {source.document_id for source in result[2]} == {"DOC001", "DOC012"}
    store.close()


def test_polar_context_with_unsupported_facet_is_not_prefixed_yes() -> None:
    envelope = {
        "status": "partially_answered",
        "claims": [{
            "text": "Invasive carp are present in Missouri.",
            "evidence_ids": ["K1"],
            "supporting_spans": [MISSOURI_CARP_EVIDENCE],
        }],
        "unsupported_facets": ["whether invasive carp are native to Missouri"],
    }

    _ensure_polar_answer_prefix("Are invasive carp native to Missouri?", envelope)

    assert envelope["claims"][0]["text"] == "Invasive carp are present in Missouri."
