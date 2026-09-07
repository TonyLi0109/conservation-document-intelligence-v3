"""FTS indexes preserve canonical evidence and follow the caller's transaction."""

import sqlite3

import pytest

from data_models import DocumentSource, KnowledgeArtifact
from database import KnowledgeStore
from retrieval_index import (
    CHUNK_INDEX, DOCUMENT_INDEX, document_candidates, ensure_retrieval_index,
    index_stats, lexical_candidates,
)


def source(store, docid, title, *, agency="Synthetic Conservation Agency", topic="test", year="2024"):
    store.upsert_document_sources([DocumentSource(
        docid, title, "https://example.org/" + docid, None, docid + ".pdf", "pdf",
        year, agency, topic,
    )])


def chunk(store, docid, text, *, title="Synthetic Report", page="1"):
    return store.ingest_chunk(KnowledgeArtifact(docid, title, page, text), [1.0, 0.0, 0.0])


@pytest.fixture
def store(tmp_path):
    with KnowledgeStore(tmp_path / "retrieval.db") as instance:
        yield instance


def canonical_snapshot(store):
    return {table: [tuple(row) for row in store.connection.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in ("documents", "knowledge_artifacts", "vector_embeddings")}


def test_initial_build_and_forced_rebuild_preserve_canonical_text_ids_metadata_vectors(store):
    source(store, "DOC951", "Invasive Carp Guidance")
    artifact_id = chunk(store, "DOC951", "Carp harvesting reduced biomass by 60 percent.", page="23")
    before = canonical_snapshot(store)
    assert lexical_candidates(store, "carp harvesting") == [artifact_id]
    assert document_candidates(store, "carp guidance") == ["DOC951"]
    assert index_stats(store)["build_count"] == 1
    ensure_retrieval_index(store, force=True)
    assert index_stats(store)["build_count"] == 2
    assert canonical_snapshot(store) == before
    # Use the real provenance adapter after ranking: indexed text is never
    # substituted for the canonical page, quote or trusted source URL.
    evidence = store._artifacts_by_ranked_ids([artifact_id])[0]
    assert evidence.page_number == "23"
    assert evidence.original_text_chunk == "Carp harvesting reduced biomass by 60 percent."
    assert evidence.source_url == "https://example.org/DOC951"


def test_reopen_uses_persisted_postings_and_queries_do_not_load_whole_corpus(tmp_path):
    path = tmp_path / "reopen.db"
    with KnowledgeStore(path) as instance:
        source(instance, "DOC951", "Invasive Carp Guidance")
        artifact_id = chunk(instance, "DOC951", "Carp harvesting outcomes.")
        ensure_retrieval_index(instance)
    with KnowledgeStore(path) as reopened:
        assert lexical_candidates(reopened, "harvest") == [artifact_id]
        assert index_stats(reopened)["build_count"] == 1
        statements = []
        reopened.connection.set_trace_callback(statements.append)
        assert lexical_candidates(reopened, "harvest") == [artifact_id]
        assert document_candidates(reopened, "carp") == ["DOC951"]
        reopened.connection.set_trace_callback(None)
        assert not any("rebuild" in sql.lower() or "create " in sql.lower() for sql in statements)
        assert not any("select original_text_chunk from" in sql.lower() for sql in statements)


def test_insert_update_delete_triggers_follow_canonical_and_metadata_changes(store):
    ensure_retrieval_index(store)
    source(store, "DOC951", "Carp Guidance", agency="River Agency", topic="harvesting")
    first = chunk(store, "DOC951", "Electrofishing effectiveness data.")
    assert lexical_candidates(store, "electrofishing") == [first]
    assert document_candidates(store, "harvesting") == ["DOC951"]
    with store.connection:
        store.connection.execute("UPDATE knowledge_artifacts SET original_text_chunk=? WHERE artifact_id=?",
                                 ("Acoustic barrier monitoring.", first))
        store.connection.execute("UPDATE documents SET title=?,agency=?,topic=? WHERE document_id=?",
                                 ("Mussel Guidance", "Lake Agency", "decontamination", "DOC951"))
    assert lexical_candidates(store, "electrofishing") == []
    assert lexical_candidates(store, "acoustic") == [first]
    assert document_candidates(store, "harvesting") == []
    assert document_candidates(store, "decontamination") == ["DOC951"]
    with store.connection:
        store.connection.execute("DELETE FROM vector_embeddings WHERE artifact_id=?", (first,))
        store.connection.execute("DELETE FROM knowledge_artifacts WHERE artifact_id=?", (first,))
        store.connection.execute("DELETE FROM documents WHERE document_id=?", ("DOC951",))
    assert lexical_candidates(store, "acoustic") == []
    assert document_candidates(store, "decontamination") == []
    assert index_stats(store)["indexed_chunks"] == 0
    assert index_stats(store)["indexed_documents"] == 0
    assert index_stats(store)["build_count"] == 1


@pytest.mark.parametrize("query", [
    'carp" OR * NOT - NEAR(}', "title:carp", "carp'); DROP TABLE documents; --",
    "carp AND mussel", "carp*", '"carp', "carp + (",
])
def test_user_sql_and_fts_syntax_are_literal_tokens(store, query):
    artifact_id = chunk(store, "DOC951", "Carp outcomes.")
    assert artifact_id in lexical_candidates(store, query)
    assert store.artifact_count == 1
    assert store.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_empty_scope_and_punctuation_are_empty_and_scope_is_parameterized(store):
    first = chunk(store, "DOC951", "Carp reductions.")
    second = chunk(store, "DOC952", "Carp reductions.")
    assert lexical_candidates(store, "carp", document_ids=["DOC952"]) == [second]
    assert lexical_candidates(store, "carp", document_ids=[]) == []
    assert lexical_candidates(store, "carp", document_ids=["DOC952') OR 1=1 --"]) == []
    assert lexical_candidates(store, "carp") == [first, second]
    for query in ("", "  ", '"*:{}+_-'):
        assert lexical_candidates(store, query) == []
        assert document_candidates(store, query) == []
    with pytest.raises(TypeError, match="sequence"):
        lexical_candidates(store, "carp", document_ids="DOC951")


def test_quoted_phrase_is_an_additional_literal_rank_signal_without_requiring_all_terms(store):
    separated = chunk(store, "DOC951", "Silver management differs from grass carp management.")
    exact = chunk(store, "DOC952", "Silver carp management differs from grass management.")
    assert lexical_candidates(store, '"silver carp"') == [exact, separated]


def test_document_metadata_recovers_interior_long_report_evidence(store):
    source(store, "DOC951", "Missouri State Wildlife Action Plan", topic="wetland restoration")
    source(store, "DOC952", "Synthetic Ecology Research", topic="forest birds")
    for number in range(60):
        chunk(store, "DOC951", f"Background land cover inventory section {number}.", page=str(number + 1))
    interior = chunk(store, "DOC951", "Floodplain restoration reduced peak flows by 25 percent.", page="301")
    chunk(store, "DOC952", "Floodplain restoration had uncertain outcomes.")
    docs = document_candidates(store, "Missouri wildlife action plan")
    assert docs[0] == "DOC951"
    assert lexical_candidates(store, "floodplain restoration percent", top_k=1, document_ids=docs[:1]) == [interior]
    assert document_candidates(store, "synthetic conservation agency") == ["DOC952", "DOC951"]


def test_initial_build_and_trigger_mutations_respect_outer_rollback(store):
    first = chunk(store, "DOC951", "Original carp evidence.")
    store.connection.execute("BEGIN")
    ensure_retrieval_index(store)
    assert store.connection.in_transaction
    store.connection.rollback()
    assert store.connection.execute("SELECT 1 FROM sqlite_master WHERE name=?", (CHUNK_INDEX,)).fetchone() is None
    assert lexical_candidates(store, "original") == [first]
    store.connection.execute("BEGIN")
    store.connection.execute("UPDATE knowledge_artifacts SET original_text_chunk=? WHERE artifact_id=?", ("Changed mussel evidence.", first))
    assert lexical_candidates(store, "mussel") == [first]
    ensure_retrieval_index(store, force=True)
    assert store.connection.in_transaction
    store.connection.rollback()
    assert lexical_candidates(store, "mussel") == []
    assert lexical_candidates(store, "original") == [first]
    assert index_stats(store)["build_count"] == 1


def test_external_connection_writes_update_persisted_index(store):
    first = chunk(store, "DOC951", "Carp outcomes.")
    ensure_retrieval_index(store)
    with sqlite3.connect(store.database_path) as other:
        other.execute("UPDATE knowledge_artifacts SET original_text_chunk=? WHERE artifact_id=?", ("Mussel outcomes.", first))
    assert lexical_candidates(store, "carp") == []
    assert lexical_candidates(store, "mussel") == [first]
    assert index_stats(store)["build_count"] == 1


def test_index_integrity_after_catalog_upsert(store):
    source(store, "DOC951", "First title")
    ensure_retrieval_index(store)
    source(store, "DOC951", "Updated title", year="2025")
    assert document_candidates(store, "first") == []
    assert document_candidates(store, "updated 2025") == ["DOC951"]
    for index in (CHUNK_INDEX, DOCUMENT_INDEX):
        # rank=1 compares the persisted postings to the external canonical rows.
        store.connection.execute(f"INSERT INTO {index}({index},rank) VALUES('integrity-check',1)")
