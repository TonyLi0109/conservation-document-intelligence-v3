"""Protect application routing, local temporal prose, and additive migration."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from data_models import DocumentSource, KnowledgeArtifact
from database import KnowledgeStore
import main


@pytest.fixture
def store(tmp_path):
    with KnowledgeStore(tmp_path / "corpus.db") as result:
        for doc_id, year, extra in (("DOC951", "2020", ""), ("DOC952", "2024", "This guidance supersedes DOC951.")):
            title = f"Invasive Carp Guidance {year}"
            result.upsert_document_sources([DocumentSource(doc_id, title, "https://example.org/fixture.pdf", None,
                                                          doc_id + ".pdf", "pdf", year=year, agency="Test-only agency")])
            result.ingest_chunk(KnowledgeArtifact(doc_id, title, "1",
                                f"Published: {year}.\nStatus: final.\n{extra}\nInvasive carp guidance recommends targeted harvest and monitoring.",
                                source_url="https://example.org/fixture.pdf"), [1., 0.])
        yield result


@pytest.mark.parametrize("question,expected", [
    ("What is the current guidance for invasive carp?", "DOC952"),
    ("What did the 2020 invasive carp guidance recommend?", "DOC951"),
    ("What does the latest report say about invasive carp?", "DOC952"),
])
def test_temporal_chat_and_search_do_not_call_providers(store, monkeypatch, question, expected):
    def forbidden(*args, **kwargs):
        pytest.fail("Temporal requests should use canonical lifecycle evidence without API calls")
    monkeypatch.setattr(main, "generate_embedding", forbidden)
    monkeypatch.setattr(main, "call_llm", forbidden)
    diagnostics = {}
    answer, preamble, sources = main.ask_chatbot_with_context(question, store, top_k=1, diagnostics=diagnostics)
    assert diagnostics["temporal_selected_document_ids"][0] == expected
    assert "Source excerpt" in answer and "harvest" in answer
    assert sources and "indexed corpus" in preamble
    assert main.search_corpus(question, store, top_k=1)[0].document_id == expected


@pytest.mark.parametrize("fabricated_claim", [
    "DOC952 supersedes every previous report.",
    "The 2020 guidance has been replaced.",
    "The previous guidance is obsolete.",
    "This guidance is current.",
    "The earlier recommendations have been formally withdrawn.",
    "These recommendations remain valid.",
])
def test_model_cannot_invent_supersession_in_normal_answer_or_preamble(store, monkeypatch, fabricated_claim):
    artifact = store.retrieve([1., 0.], 1)[0]
    monkeypatch.setattr(main, "generate_embedding", lambda text: [1., 0.])
    def synthesize(system, prompt, artifacts, **kwargs):
        span = next(iter(artifacts.values())).original_text_chunk.splitlines()[-1]
        return json.dumps({"status": "answered", "preamble": fabricated_claim,
                           "claims": [{"text": fabricated_claim,
                                       "evidence_ids": ["K1"], "supporting_spans": [span]}],
                           "unsupported_facets": []})
    monkeypatch.setattr(main, "call_llm", synthesize)
    answer, preamble, sources = main.ask_chatbot_with_context("Tell me about invasive carp.", store)
    assert fabricated_claim not in answer + preamble
    assert sources


@pytest.mark.parametrize("claim", [
    "Electrical current deters invasive carp movement.",
    "Current velocity affects fish movement and sediment transport.",
    "The report says obsolete equipment should be replaced.",
    "The guidance recommends replacing obsolete monitoring equipment.",
    "Wetland conditions are currently suitable for migratory birds.",
])
def test_lifecycle_guard_preserves_ordinary_ecological_and_equipment_claims(claim):
    from temporal import is_lifecycle_assertion
    assert not is_lifecycle_assertion(claim)


def test_ordinary_search_preserves_existing_semantic_path(store, monkeypatch):
    calls = []
    def embed(text):
        calls.append(text)
        return [1., 0.]
    monkeypatch.setattr(main, "generate_embedding", embed)
    results = main.search_corpus("invasive carp population monitoring", store, top_k=2, mode="dense")
    assert calls == ["invasive carp population monitoring"]
    assert results == store.retrieve([1., 0.], 2)


@pytest.mark.integration
def test_explicit_migration_preserves_ids_vectors_and_source_registry(store):
    before = {table: [tuple(row) for row in store.connection.execute(f"SELECT * FROM {table}")]
              for table in ("knowledge_artifacts", "vector_embeddings", "documents")}
    command = [sys.executable, "-X", "utf8", "main.py", "--database", store.database_path, "--backfill-lifecycle"]
    for _ in range(2):
        result = subprocess.run(command, cwd=Path(main.__file__).parent, capture_output=True, text=True, encoding="utf-8", timeout=20)
        assert result.returncode == 0, result.stderr
        assert "2 documents" in result.stdout
    after = {table: [tuple(row) for row in store.connection.execute(f"SELECT * FROM {table}")]
             for table in before}
    assert before == after
    assert store.connection.execute("SELECT COUNT(*) FROM document_lifecycles").fetchone()[0] == 2
