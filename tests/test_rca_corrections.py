"""Behavioral tests for planning horizons, title routing, and two-source coverage."""
import json
from datetime import date
import pytest
from data_models import DocumentSource, KnowledgeArtifact
from database import KnowledgeStore
from document_lifecycle import extract_lifecycles
from document_targets import resolve_document_targets
from temporal import (active_planning_horizon, detect_temporal_intent, render_temporal_answer,
                      select_temporal_evidence)
from retrieval import retrieve_evidence, select_diverse_evidence

Q4 = 'How does the 2022 Missouri Comprehensive Conservation Strategy differ from the 2015 Missouri State Wildlife Action Plan in its treatment of aquatic invasive species?'
DOCS = [dict(document_id='DOC036', title='2022 Missouri Comprehensive Conservation Strategy', year='2022'),
        dict(document_id='DOC001', title='Missouri State Wildlife Action Plan', year='2015')]


def lifecycle(text):
    return extract_lifecycles([dict(document_id='DOC002', title='Wetland Program Plan', year='2010')],
        [dict(document_id='DOC002', page_number='1', original_text_chunk=text)])['DOC002']


@pytest.mark.parametrize('separator', ['-', '\u2013', '\u2014', 'to', 'through'])
def test_own_planning_period_exact_evidence(separator):
    text = f'This wetland program plan includes the years 2023 {separator} 2028 with reviews and updates anticipated.'
    doc = lifecycle(text)
    period = doc['planning_period']
    assert (period['start_year'], period['end_year']) == (2023, 2028)
    assert period['evidence']['exact_span'] in text
    assert doc['publication_date'] == '2010'
    assert active_planning_horizon(doc, date(2023, 1, 1))
    assert active_planning_horizon(doc, date(2028, 12, 31))
    assert not active_planning_horizon(doc, date(2029, 1, 1))
    assert not active_planning_horizon(doc, date(2022, 12, 31))
    assert not active_planning_horizon({**doc, 'status': 'withdrawn'}, date(2026, 1, 1))
    assert not active_planning_horizon({**doc, 'status': 'draft'}, date(2026, 1, 1))


@pytest.mark.parametrize('text', [
    'A referenced management plan covered 2023-2028.',
    'This plan does not cover 2023-2028.',
    'This plan may cover 2023-2028.',
    'Planning period: 2028-2023.',
    'Planning period: 2023-2028. Planning period: 2024-2029.',
])
def test_external_invalid_and_conflicting_ranges_not_promoted(text):
    assert lifecycle(text)['planning_period'] is None


def test_title_years_are_not_temporal_intent():
    ids, residual = resolve_document_targets(Q4, DOCS)
    assert set(ids) == {'DOC001', 'DOC036'}
    assert '2015' not in residual and '2022' not in residual
    assert detect_temporal_intent(Q4, documents=DOCS).mode == 'none'
    assert detect_temporal_intent(Q4 + ' as of 2020', documents=DOCS).as_of == '2020'
    assert detect_temporal_intent('Which is current: ' + Q4, documents=DOCS).mode == 'current'
    assert detect_temporal_intent('Compare guidance from 2015 to 2022', documents=DOCS).mode == 'comparison'
    assert detect_temporal_intent('How has guidance changed over time?', documents=DOCS).mode == 'comparison'


def test_ambiguous_and_wrong_edition_not_resolved():
    duplicate = {**DOCS[1], 'document_id': 'DOC999'}
    ids, _ = resolve_document_targets('2015 Missouri State Wildlife Action Plan', DOCS + [duplicate])
    assert ids == []
    ids, _ = resolve_document_targets('2018 Missouri State Wildlife Action Plan', DOCS)
    assert ids == []


def add_document(store, doc, texts):
    store.upsert_document_sources([DocumentSource(doc['document_id'], doc['title'],
        'https://example.org/source.pdf', None, 'source.pdf', 'pdf', year=doc['year'])])
    for text in texts:
        store.ingest_chunk(KnowledgeArtifact(doc['document_id'], doc['title'], '1', text), [1., 0.])


def test_current_plan_beats_historical_publication(tmp_path, monkeypatch):
    with KnowledgeStore(':memory:') as store:
        add_document(store, dict(document_id='DOC100', title='Wetland Guidance', year='2020'),
            ['Status: final. Wetland management guidance recommends monitoring.'])
        add_document(store, dict(document_id='DOC002', title='Wetland Program Plan', year='2010'),
            ['Planning period: 2023-2028. Wetland management guidance recommends monitoring.'])
        result = select_temporal_evidence('Which wetland guidance is current?', store, top_k=1, as_of='2026-09-15')
        assert result['selected_document_ids'] == ['DOC002']
        assert '2023-2028' in result['decisions'][0]['reason']
        log = tmp_path / 'temporal-provenance.jsonl'
        monkeypatch.setenv('V3_PROVENANCE_LOG', str(log))
        answer, _, _ = render_temporal_answer(result, store)

    assert answer.startswith('**Conclusion:**')
    assert '- **DOC002:** Active planning period: 2023-2028' in answer
    assert 'Source excerpt:' not in answer
    assert 'Version/date evidence:' not in answer
    records = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
    assert any('2023-2028' in (record['supporting_evidence_span'] or '')
               and record['validation_status'] == 'SUPPORTED' for record in records)

def test_diversity_preserves_cross_doc_duplicates_and_reports_capacity():
    artifacts = {i: KnowledgeArtifact(doc, 'Title', '1', 'Same canonical statement.')
                 for i, doc in [(1, 'A'), (2, 'A'), (3, 'B')]}
    ids, _, trace = select_diverse_evidence([1, 2, 3], artifacts, 2, ['A', 'B'])
    assert ids == [1, 3]
    assert not trace['missing_target_ids']
    ids, _, trace = select_diverse_evidence([1, 2, 3], artifacts, 1, ['A', 'B'])
    assert len(ids) == 1 and trace['capacity_limited']
    assert trace['missing_target_ids'] == ['B']


def test_q4_reaches_synthesis_and_retains_both_sources(monkeypatch):
    import main
    calls = []
    def llm(system_prompt, user_prompt, handles, **kwargs):
        calls.append(user_prompt)
        by_doc = {}
        for key, artifact in handles.items():
            by_doc.setdefault(artifact.document_id, (key, artifact))
        assert set(by_doc) == {'DOC001', 'DOC036'}
        return json.dumps({'preamble': '', 'status': 'answered', 'unsupported_facets': [],
            'claims': [{'text': artifact.original_text_chunk, 'evidence_ids': [key],
                        'supporting_spans': [artifact.original_text_chunk]}
                       for key, artifact in by_doc.values()]})
    monkeypatch.setattr(main, 'generate_embedding', lambda q: [1., 0.])
    monkeypatch.setattr(main, 'call_llm', llm)
    with KnowledgeStore(':memory:') as store:
        add_document(store, DOCS[0], [f'Aquatic invasive species monitoring increased by {i} percent in Missouri.' for i in range(1, 81)])
        add_document(store, DOCS[1], ['Aquatic invasive species monitoring requires surveys in Missouri.'])
        add_document(store, dict(document_id='DOC999', title='Chesapeake Bay', year='2022'),
                     ['Aquatic invasive species monitoring aquatic invasive species.'])
        trace = {}
        artifacts = retrieve_evidence(store, Q4, top_k=5, diagnostics=trace)
        assert {a.document_id for a in artifacts} == {'DOC001', 'DOC036'}
        assert not trace['diversity']['missing_target_ids']
        assert any(d['previous_count'] > 0 for d in trace['diversity']['selection'])
        diagnostics = {}
        _, _, sources = main.ask_chatbot_with_context(Q4, store, top_k=5, diagnostics=diagnostics)
        assert calls and diagnostics['route'] == 'multi_doc_synthesis'
        assert {a.document_id for a in sources} == {'DOC001', 'DOC036'}
