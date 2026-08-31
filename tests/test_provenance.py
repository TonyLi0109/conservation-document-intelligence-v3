"""Offline safety checks for V3's canonical and compiled provenance layers."""

from data_models import DocumentSource, KnowledgeArtifact
from database import KnowledgeStore


def test_compiled_relationship_round_trip() -> None:
    store = KnowledgeStore(":memory:")
    artifact = KnowledgeArtifact(
        "DOC999", "Test source", "1",
        "Wetlands support migratory birds and flood protection.",
    )
    store.upsert_document_sources(
        [DocumentSource(
            "DOC999", "Test source", "https://example.org/test.pdf", None,
            "test.pdf", "pdf",
        )]
    )
    store.ingest_chunk(artifact, [1.0, 0.0])
    span = artifact.original_text_chunk
    concept = {
        "concept_title": "Wetlands",
        "summary": "Grounded test",
        "important_facts": ["Wetlands support birds."],
        "related_entities": [{
            "entity_name": "migratory birds",
            "relationship_type": "supported habitat",
            "evidence_id": "K1",
            "exact_span": span,
        }],
        "supporting_evidence": [{"evidence_id": "K1", "exact_span": span}],
    }
    knowledge_id = store.save_compiled_concept(
        "Wetlands", concept, {"K1": artifact},
        model_name="offline-test", generation_version="test",
    )
    cached = store.get_compiled_concept("Wetlands")
    assert cached is not None
    assert cached["knowledge_id"] == knowledge_id
    assert cached["concept"]["related_entities"][0]["exact_span"] == span
    store.close()


def test_compiled_relationship_rejects_nonverbatim_span() -> None:
    store = KnowledgeStore(":memory:")
    artifact = KnowledgeArtifact("DOC999", "Test", "1", "Canonical text.")
    store.ingest_chunk(artifact, [1.0, 0.0])
    concept = {
        "concept_title": "Test",
        "summary": "Test",
        "important_facts": [],
        "related_entities": [{
            "entity_name": "Other", "relationship_type": "related",
            "evidence_id": "K1", "exact_span": "Invented text.",
        }],
        "supporting_evidence": [{"evidence_id": "K1", "exact_span": "Canonical text."}],
    }
    try:
        store.save_compiled_concept(
            "Test", concept, {"K1": artifact}, model_name="test",
            generation_version="test",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-verbatim relationship evidence was accepted")
    finally:
        store.close()
