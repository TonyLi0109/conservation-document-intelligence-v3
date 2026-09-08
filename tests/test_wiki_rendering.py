"""Exercise the actual Wiki renderer with persisted pages and no provider calls."""

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import wiki_compiler as wiki
from data_models import DocumentSource, KnowledgeArtifact
from database import KnowledgeStore
from validator import format_artifact_location


ENTITY = "U.S. Army Corps of Engineers"
SPAN = "The U.S. Army Corps of Engineers and U.S. Fish and Wildlife Service jointly monitor wetlands."


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


@pytest.fixture
def render_page(tmp_path, monkeypatch):
    source = Path(wiki.__file__).with_name("app.py").read_text(encoding="utf-8")
    renderer = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "render_wiki_tab")
    st = MagicMock()
    st.session_state = State()
    st.columns.return_value = [MagicMock(), MagicMock()]
    st.columns.return_value[0].selectbox.return_value = "Agencies"
    st.columns.return_value[1].selectbox.return_value = ENTITY
    st.button.return_value = False

    def forbidden(*args, **kwargs):
        pytest.fail("Selecting a persisted Wiki page must not request generation or retrieval")

    monkeypatch.setattr(wiki, "call_structured_llm", forbidden)
    monkeypatch.setattr(wiki, "generate_embedding", forbidden)
    namespace = {
        "st": st,
        "KnowledgeStore": KnowledgeStore,
        "generate_extractive_wiki_concept": wiki.generate_extractive_wiki_concept,
        "generate_wiki_concept": forbidden,
        "format_artifact_location": format_artifact_location,
    }
    exec(compile(ast.Module(body=[renderer], type_ignores=[]), "app.py", "exec"), namespace)

    with KnowledgeStore(tmp_path / "render.db") as store:
        artifact = KnowledgeArtifact(
            "DOC999", "Wetland monitoring", "7", SPAN,
            source_url="https://example.org/monitoring.pdf", printed_page_label="iv",
        )
        store.upsert_document_sources([DocumentSource(
            artifact.document_id, artifact.title, artifact.source_url,
            None, "monitoring.pdf", "pdf",
        )])
        store.ingest_chunk(artifact, [1.0, 0.0])
        monkeypatch.setattr(store, "list_wiki_entities", lambda: {"Agencies": [ENTITY]})
        monkeypatch.setattr(store, "retrieve", forbidden)

        def render(related_entities):
            concept = {
                "concept_title": ENTITY,
                "summary": SPAN,
                "important_facts": [SPAN],
                "related_entities": related_entities,
                "supporting_evidence": [{"evidence_id": "K1", "exact_span": SPAN}],
            }
            store.save_compiled_concept(
                ENTITY, concept, {"K1": artifact},
                model_name=wiki.EXTRACTIVE_MODEL_NAME,
                generation_version=wiki.EXTRACTIVE_COMPILER_VERSION,
                generation_method="deterministic_extractive",
            )
            namespace["render_wiki_tab"](store, wiki.LLM_MODEL)
            assert st.session_state.v3_wiki_result["cached"]
            st.error.assert_not_called()
            st.markdown.assert_any_call(SPAN, unsafe_allow_html=False)
            st.markdown.assert_any_call(f"- {SPAN}", unsafe_allow_html=False)
            st.text.assert_any_call(SPAN)
            st.link_button.assert_any_call(
                "Open source document", artifact.source_url,
                width="content", key="wiki_source_1_DOC999",
            )
            return st, artifact

        yield render


def test_empty_related_entities_section_remains_visible(render_page):
    st, _ = render_page([])
    headings = [call.args[0] for call in st.markdown.call_args_list
                if call.args[0].startswith("#### ")]
    assert headings == ["#### Important facts", "#### Related entities", "#### Supporting evidence"]
    st.caption.assert_any_call(
        "No supported relationships were found in the evidence selected for this page."
    )
    st.dataframe.assert_not_called()


def test_relationship_table_identifies_its_canonical_supporting_evidence(render_page):
    st, artifact = render_page([{
        "entity_name": "U.S. Fish and Wildlife Service",
        "relationship_type": "joint wetland monitoring",
        "evidence_id": "K1",
        "exact_span": SPAN,
    }])
    st.markdown.assert_any_call("#### Related entities")
    st.dataframe.assert_called_once_with([{
        "Entity": "U.S. Fish and Wildlife Service",
        "Relationship": "joint wetland monitoring",
        "Source": f"Evidence 1 · [DOC999, {format_artifact_location(artifact)}]",
    }], width="stretch", hide_index=True)
    labels = [call.args[0] for call in st.expander.call_args_list]
    assert any(label.startswith("Evidence 1") and "DOC999" in label
               and format_artifact_location(artifact) in label for label in labels)
    assert not any("No supported relationships" in str(call) for call in st.caption.call_args_list)
