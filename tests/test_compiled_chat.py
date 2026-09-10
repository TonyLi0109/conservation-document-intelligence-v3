"""Offline coverage of compiled knowledge in the actual chat synthesis path."""
import json

import pytest

import main
import retrieval
from compiled_context import add_compiled_context
from data_models import KnowledgeArtifact
from database import KnowledgeStore


@pytest.fixture
def compiled_store():
    with KnowledgeStore(':memory:') as store:
        artifact = KnowledgeArtifact('DOC901', 'Wetland study', '4',
                                     'Wetlands support migratory birds and flood protection.')
        store.ingest_chunk(artifact, [1., 0.])
        concept = {
            'concept_title': 'Wetlands', 'summary': 'Wetlands provide habitat and flood protection.',
            'important_facts': ['Wetlands support migratory birds.'],
            'related_entities': [{'entity_name': 'migratory birds', 'relationship_type': 'habitat',
                                  'evidence_id': 'K1', 'exact_span': artifact.original_text_chunk}],
            'supporting_evidence': [{'evidence_id': 'K1', 'exact_span': artifact.original_text_chunk}],
        }
        store.save_compiled_concept('wetland habitats', concept, {'K1': artifact},
                                    model_name='test', generation_version='test')
        yield store, artifact


def test_question_search_uses_stored_key(compiled_store):
    store, _ = compiled_store
    rows = store.search_compiled_knowledge('Explain how wetlands support wildlife')
    assert len(rows) == 1
    assert rows[0]['concept_key'] == 'wetland habitats'
    assert store.search_compiled_knowledge('Explain invasive carp control') == []


def test_actual_chat_uses_compilation_and_cites_original_source(compiled_store, monkeypatch):
    store, artifact = compiled_store
    monkeypatch.setattr(main, 'generate_embedding', lambda _: [1., 0.])
    # Compiled evidence must be usable even when raw retrieval misses its chunk.
    monkeypatch.setattr(retrieval, 'retrieve_evidence', lambda *a, **k: [])
    captured = {}
    def synthesize(system, prompt, handles, **kwargs):
        captured['prompt'] = prompt
        assert 'COMPILED_KNOWLEDGE_JSON' in prompt
        assert 'Wetlands provide habitat and flood protection.' in prompt
        assert handles['K1'].original_text_chunk == artifact.original_text_chunk
        return json.dumps({'preamble': '', 'status': 'answered', 'unsupported_facets': [],
                           'claims': [{'text': 'Wetlands support migratory birds.',
                                       'evidence_ids': ['K1'],
                                       'supporting_spans': [artifact.original_text_chunk]}]})
    monkeypatch.setattr(main, 'call_llm', synthesize)
    diagnostics = {}
    answer, _, sources = main.ask_chatbot_with_context('Explain wetlands benefits', store,
                                                       diagnostics=diagnostics)
    assert captured and diagnostics['compiled_knowledge_ids']
    assert 'DOC901' in answer
    assert sources[0].original_text_chunk == artifact.original_text_chunk


def test_handles_remapped_and_deduplicated(compiled_store):
    store, artifact = compiled_store
    other = KnowledgeArtifact('DOC902', 'Other source', '1', 'Different evidence.')
    handles = {'K1': other, 'K2': artifact}
    prompt, ids = add_compiled_context(store, 'Explain wetlands', handles)
    payload = json.loads(prompt.split('metadata.\n', 1)[1])
    assert len(handles) == 2 and ids
    assert payload[0]['supporting_evidence'][0]['evidence_id'] == 'K2'
    assert payload[0]['related_entities'][0]['evidence_id'] == 'K2'


def test_invalidated_or_corrupted_cache_is_not_used(compiled_store):
    store, _ = compiled_store
    store.connection.execute("UPDATE compiled_evidence SET exact_span='Invented quote'")
    handles = {}
    assert add_compiled_context(store, 'Explain wetlands', handles) == ('', [])
    assert handles == {}
    store.invalidate_compiled_knowledge('DOC901')
    assert add_compiled_context(store, 'Explain wetlands', handles) == ('', [])


def test_budget_and_missing_cache_leave_raw_evidence_intact(compiled_store):
    store, artifact = compiled_store
    handles = {}
    assert add_compiled_context(store, 'Explain wetlands', handles, max_sources=0) == ('', [])
    assert handles == {}
    handles['K1'] = artifact
    assert add_compiled_context(store, 'Explain invasive carp', handles) == ('', [])
    assert handles == {'K1': artifact}


def test_derived_prose_cannot_be_used_as_verbatim_evidence(compiled_store, monkeypatch):
    store, artifact = compiled_store
    monkeypatch.setattr(main, 'generate_embedding', lambda _: [1., 0.])
    monkeypatch.setattr(retrieval, 'retrieve_evidence', lambda *a, **k: [])
    def synthesize(system, prompt, handles, **kwargs):
        return json.dumps({'preamble': '', 'status': 'answered', 'unsupported_facets': [],
                           'claims': [{'text': 'Wetlands provide habitat and flood protection.',
                                       'evidence_ids': ['K1'],
                                       'supporting_spans': ['Wetlands provide habitat and flood protection.']}]})
    monkeypatch.setattr(main, 'call_llm', synthesize)
    answer, preamble, sources = main.ask_chatbot_with_context('Explain wetlands benefits', store)
    assert 'Wetlands provide habitat and flood protection.' not in answer
    assert 'could not be validated' in preamble
    assert sources[0].original_text_chunk == artifact.original_text_chunk
