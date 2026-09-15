"""Audit lifecycle extraction vs catalog dates and the actual temporal selection."""
import json
import re
from debug_retrieval_common import (
    Q1, arguments, copied_store, actual_temporal_route, save_report, PipelineTracer, capture,
)
from document_lifecycle import extract_lifecycles

def diagnose(store, query, top_k):
    metadata = [dict(r) for r in store.connection.execute('SELECT document_id,title,year,agency,topic FROM documents')]
    chunks = [dict(r) for r in store.connection.execute(
        'SELECT artifact_id,document_id,title,page_number,original_text_chunk FROM knowledge_artifacts ORDER BY artifact_id')]
    exists = store.connection.execute("SELECT 1 FROM sqlite_master WHERE name='document_lifecycles'").fetchone()
    persisted = {r[0]: json.loads(r[1]) for r in store.connection.execute('SELECT document_id,payload FROM document_lifecycles')} if exists else {}
    fresh = extract_lifecycles(metadata, chunks)
    no_catalog = extract_lifecycles([{**m, 'year': ''} for m in metadata], chunks)
    horizons = []
    for chunk in chunks:
        if chunk['document_id'] != 'DOC002':
            continue
        text = chunk['original_text_chunk']
        for match in re.finditer(r'\b(20\d{2})\s*(?:[-??]|to|through)\s*(20\d{2})\b', text, re.I):
            horizons.append({'artifact_id': chunk['artifact_id'], 'page': chunk['page_number'],
                             'start': match[1], 'end': match[2], 'exact_span': match[0],
                             'context': text[max(0, match.start()-150):match.end()+150]})
    actual = actual_temporal_route(store, query, top_k)
    target = fresh.get('DOC002')
    return {
        'query': query, 'actual_temporal_route': actual,
        'catalog': metadata, 'doc002_source_range_candidates': horizons,
        'doc002_persisted': persisted.get('DOC002'), 'doc002_fresh': target,
        'doc002_without_catalog_year': no_catalog.get('DOC002'),
        'persisted_matches_fresh': persisted.get('DOC002') == target if 'DOC002' in persisted else None,
        'all_fresh_lifecycles': fresh,
        'findings': {
            'doc002_present': target is not None,
            'source_contains_2023_2028': any(h['start'] == '2023' and h['end'] == '2028' for h in horizons),
            'planning_period_field_supported': bool(target and target.get('planning_period')),
            'date_precedence': 'effective_date, revision_date, publication_date',
            'extraction_precedence': 'Explicit source dates precede inferred catalog and title dates.',
            'current_ranking': 'Replacement remains authoritative; an active explicit planning horizon gains preference over publication-only sources. Otherwise existing role/status/relevance ordering remains.',
            'interpretation': 'Range candidates require source review. A planning end year is not a publication date or proof of supersession.',
        },
    }

if __name__ == '__main__':
    args = arguments(Q1)
    with copied_store(args.database) as store, PipelineTracer(args.output) as tracer:
        capture('original_query', args.query)
        report = diagnose(store, args.query, args.top_k)
        capture('q1_diagnostic', report)
    report['trace_path'] = str(tracer.path)
    save_report(args, f'q1-{tracer.trace_id}.json', report)
