"""Offline regression coverage for immediately available, evidence-rich Wiki pages."""

import ast
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import wiki_compiler as wiki
from data_models import DocumentSource, KnowledgeArtifact
from database import KnowledgeStore


DEFINITION = "Invasive carp are nonnative fish that include bighead and silver carp."
INCLUDES = "Invasive carp include bighead carp, black carp, grass carp, and silver carp."
LOCATION = "Invasive carp occur in Missouri and threaten native aquatic communities."
IMPACT = "Invasive carp compete with native fish for food in river habitats."
CONTROL = "Invasive carp removal supports efforts to protect native aquatic species."
UNRELATED = "Missouri and the Great Lakes have separate monitoring reports."


@pytest.fixture
def store(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Interactive API request is forbidden")

    monkeypatch.setattr(wiki, "call_structured_llm", forbidden)
    monkeypatch.setattr(wiki, "generate_embedding", forbidden)
    with KnowledgeStore(tmp_path / "wiki.db") as result:
        for i, text in enumerate([
            " ".join([DEFINITION, INCLUDES, LOCATION, IMPACT, CONTROL, UNRELATED]),
            DEFINITION + " Invasive carp monitoring helps assess changes in river populations.",
        ], 1):
            document_id = f"DOC{i:03d}"
            result.upsert_document_sources([DocumentSource(
                document_id, "Carp science", f"https://example.org/{i}.pdf", None,
                f"{i}.pdf", "pdf",
            )])
            result.ingest_chunk(KnowledgeArtifact(
                document_id, "Carp science", str(i), text, printed_page_label=f"iv-{i}",
            ), [1.0, 0.0])
        yield result


def assert_provenance(result):
    concept = result["concept"]
    assert set(concept) == {"concept_title", "summary", "important_facts",
                            "related_entities", "supporting_evidence"}
    for item in concept["supporting_evidence"] + concept["related_entities"]:
        artifact = result["artifacts"][item["evidence_id"]]
        assert item["exact_span"] in artifact.original_text_chunk
        assert artifact.source_url.startswith("https://example.org/")
        assert artifact.printed_page_label == f"iv-{artifact.page_number}"


def test_rich_local_page_and_cache_without_api(store, monkeypatch):
    page = wiki.generate_extractive_wiki_concept("Invasive carp", store)
    concept = page["concept"]
    assert concept["summary"].startswith(DEFINITION)
    assert {INCLUDES, LOCATION, IMPACT, CONTROL} <= set(concept["important_facts"])
    assert concept["important_facts"].count(DEFINITION) == 1
    assert len(concept["supporting_evidence"]) > 5
    assert {r["entity_name"] for r in concept["related_entities"]} == {"Silver carp", "Missouri"}
    assert_provenance(page)
    monkeypatch.setattr(store, "retrieve", lambda *a, **kw: pytest.fail("Cache hit retrieved evidence"))
    cached = wiki.generate_extractive_wiki_concept("Invasive carp", store)
    assert cached["cached"]
    assert cached["concept"] == page["concept"]
    assert_provenance(cached)


def test_actual_entity_selection_only_loads_cached_page(store, monkeypatch):
    wiki.generate_extractive_wiki_concept("Invasive carp", store)
    monkeypatch.setattr(store, "retrieve", lambda *a, **kw: pytest.fail("Selection performed retrieval"))
    # Execute the real renderer independently of Streamlit's module startup.
    source = Path(wiki.__file__).with_name("app.py").read_text(encoding="utf-8")
    renderer = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "render_wiki_tab")
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    st = MagicMock()
    st.session_state = State()
    st.columns.return_value = [MagicMock(), MagicMock()]
    st.columns.return_value[0].selectbox.return_value = "Species"
    st.columns.return_value[1].selectbox.return_value = "Invasive carp"
    st.button.return_value = False
    namespace = {
        "st": st, "KnowledgeStore": KnowledgeStore,
        "generate_extractive_wiki_concept": wiki.generate_extractive_wiki_concept,
        "generate_wiki_concept": lambda *a, **kw: pytest.fail("Selection requested LLM generation"),
        "format_artifact_location": lambda a: a.page_number,
    }
    exec(compile(ast.Module(body=[renderer], type_ignores=[]), "app.py", "exec"), namespace)
    namespace["render_wiki_tab"](store, wiki.LLM_MODEL)
    assert st.session_state["v3_wiki_result"]["cached"]
    st.error.assert_not_called()
    st.markdown.assert_any_call(st.session_state["v3_wiki_result"]["concept"]["summary"], unsafe_allow_html=False)


def test_regeneration_shares_schema_evidence_and_relationships(store, monkeypatch):
    original = wiki.generate_extractive_wiki_concept("Invasive carp", store)
    def llm(system, prompt, schema, **kwargs):
        assert "LOCAL_COMPILATION_JSON" in prompt
        assert schema["name"] == "v3_wiki_concept"
        payload = dict(original["concept"])
        payload["summary"] = "Invasive carp are nonnative fish documented in Missouri."
        payload["supporting_evidence"] = payload["supporting_evidence"][:1]
        payload["related_entities"] = []
        return json.dumps(payload)
    monkeypatch.setattr(wiki, "call_structured_llm", llm)
    refreshed = wiki.generate_wiki_concept("Invasive carp", store, force_refresh=True)
    assert set(original["concept"]) == set(refreshed["concept"])
    assert refreshed["concept"]["supporting_evidence"] == original["concept"]["supporting_evidence"]
    assert refreshed["concept"]["related_entities"] == original["concept"]["related_entities"]
    assert_provenance(refreshed)
    assert wiki.generate_extractive_wiki_concept("Invasive carp", store)["concept"] == refreshed["concept"]


def test_precompile_upgrades_old_local_pages_and_preserves_ai(store):
    page = wiki.generate_extractive_wiki_concept("Invasive carp", store)
    old = {**page["concept"], "summary": "Old placeholder", "related_entities": []}
    store.save_compiled_concept("Invasive carp", old, page["artifacts"],
                                model_name=wiki.EXTRACTIVE_MODEL_NAME, generation_version="v3.2-extractive")
    store.save_compiled_concept("Missouri", page["concept"], page["artifacts"],
                                model_name=wiki.LLM_MODEL, generation_version="v3.1")
    assert wiki.precompile_all_wiki_concepts(store) > 0
    upgraded = store.get_compiled_concept("Invasive carp")
    assert upgraded["generation_version"] == wiki.EXTRACTIVE_COMPILER_VERSION
    assert upgraded["concept"]["summary"].startswith(DEFINITION)
    assert store.get_compiled_concept("Missouri")["model_name"] == wiki.LLM_MODEL
    assert wiki.precompile_all_wiki_concepts(store) == 0


def test_invalid_refresh_does_not_replace_valid_cache(store, monkeypatch):
    original = wiki.generate_extractive_wiki_concept("Invasive carp", store)
    payload = dict(original["concept"])
    payload["supporting_evidence"] = [{"evidence_id": "K1", "exact_span": "Fabricated quote."}]
    monkeypatch.setattr(wiki, "call_structured_llm", lambda *a, **kw: json.dumps(payload))
    with pytest.raises(ValueError, match="not verbatim"):
        wiki.generate_wiki_concept("Invasive carp", store, force_refresh=True)
    assert store.get_compiled_concept("Invasive carp")["concept"] == original["concept"]


def test_missing_evidence_never_calls_api(store):
    with pytest.raises(RuntimeError, match="No safe evidence"):
        wiki.generate_extractive_wiki_concept("Unmentioned animal", store)


def test_negation_and_keyword_overlap_do_not_create_relationships():
    page = wiki._extractive_payload("Invasive carp", {"K1": [
        "There is no evidence that invasive carp occur in Missouri.",
        "Invasive carp monitoring continues while the Great Lakes have separate reports.",
        "Invasive carp include neither silver carp nor other native species in this fictional classification.",
    ]})
    assert page["related_entities"] == []


def test_duplicate_quotes_keep_distinct_document_provenance(store):
    page = wiki.generate_extractive_wiki_concept("Invasive carp", store)
    quotes = [item for item in page["concept"]["supporting_evidence"]
              if item["exact_span"] == DEFINITION]
    assert {page["artifacts"][q["evidence_id"]].document_id for q in quotes} == {"DOC001", "DOC002"}
    keys = [(q["evidence_id"], q["exact_span"]) for q in page["concept"]["supporting_evidence"]]
    assert len(keys) == len(set(keys))


def test_old_cache_can_upgrade_locally_on_selection(store):
    page = wiki.generate_extractive_wiki_concept("Invasive carp", store)
    store.save_compiled_concept("Invasive carp", {**page["concept"], "summary": "Placeholder"},
                                page["artifacts"], model_name=wiki.EXTRACTIVE_MODEL_NAME,
                                generation_version="v3.2-extractive")
    upgraded = wiki.generate_extractive_wiki_concept("Invasive carp", store)
    assert upgraded["generation_version"] == wiki.EXTRACTIVE_COMPILER_VERSION
    assert upgraded["concept"]["summary"].startswith(DEFINITION)
