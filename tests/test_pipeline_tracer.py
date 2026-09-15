import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from pipeline_tracer import PipelineTracer, capture, active, STAGES


def test_snapshots_and_request_isolation(tmp_path):
    def run(value):
        with PipelineTracer(tmp_path) as tracer:
            mutable = {'value': [value]}
            capture('original_query', mutable)
            mutable['value'].append('changed')
        assert not active()
        return json.loads(tracer.path.read_text(encoding='utf-8'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ['one', 'two']))
    assert [r['stages']['original_query']['value'] for r in results] == [
        {'value': ['one']}, {'value': ['two']}]
    assert results[0]['trace_id'] != results[1]['trace_id']


def test_export_failure_preserves_exception(tmp_path):
    blocked = tmp_path / 'file'
    blocked.write_text('file')
    with pytest.raises(ValueError, match='pipeline error'):
        with PipelineTracer(blocked):
            raise ValueError('pipeline error')
    assert not active()


def test_complete_trace_preserves_answer_and_raw_prompt(tmp_path, monkeypatch):
    import api_clients
    import main
    from data_models import KnowledgeArtifact
    from database import KnowledgeStore
    raw = '  ' + json.dumps({'preamble': '', 'status': 'answered', 'claims': [
        {'text': 'Wetlands support birds.', 'evidence_ids': ['K1'],
         'supporting_spans': ['Wetlands support birds.']}], 'unsupported_facets': []}) + '  '
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(http_response=SimpleNamespace(json=lambda: {
            'choices': [{'message': {'content': raw}, 'finish_reason': 'stop'}]}))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=create))))
    monkeypatch.setattr(api_clients, '_client', lambda: client)
    monkeypatch.setattr(main, 'generate_embedding', lambda q: [1., 0.])
    with KnowledgeStore(':memory:') as store:
        store.ingest_chunk(KnowledgeArtifact('DOC999', 'Wetland habitat', '1',
            'Wetlands support birds.'), [1., 0.])
        query = 'Explain wetland habitat benefits'
        plain = main.ask_chatbot_with_context(query, store)
        with PipelineTracer(tmp_path) as tracer:
            traced = main.ask_chatbot_with_context(query, store)
    assert traced == plain
    data = json.loads(tracer.path.read_text(encoding='utf-8'))
    assert not data['errors']
    assert all(data['stages'][s]['status'] == 'captured' for s in STAGES)
    assert data['stages']['draft_answer']['value'] == raw
    assert data['stages']['llm_context_payload']['value'] == requests[-1]
    assert data['stages']['validation_results']['value']['accepted']
    searches = data['stages']['initial_retrieval']['value']['raw_searches']
    assert any(e['stage'] == 'dense_search' and e['value']['hits'][0]['score'] == 1. for e in searches)
    assert any(e['stage'] == 'lexical_search' and e['value']['hits'] for e in searches)


def test_partial_trace_and_failed_serialization(tmp_path):
    with PipelineTracer(tmp_path) as tracer:
        capture('original_query', 'query')
        capture('reranked_evidence', object())
    data = json.loads(tracer.path.read_text(encoding='utf-8'))
    assert data['errors'][0]['stage'] == 'reranked_evidence'
    assert data['stages']['draft_answer']['status'] == 'not_reached'
