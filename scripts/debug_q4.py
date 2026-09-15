"""Inspect actual routing, title scope, and controlled retrieval ablations."""
import json
from debug_retrieval_common import (
    Q4, arguments, copied_store, actual_temporal_route, save_report, PipelineTracer, capture,
)
from retrieval import retrieve_evidence, _normalize
from config import SETTINGS
TARGETS = (
    ('2022', 'Missouri Comprehensive Conservation Strategy'),
    ('2015', 'Missouri State Wildlife Action Plan'),
)

def diagnose(store, query, top_k, vector=None):
    documents = [dict(r) for r in store.connection.execute('SELECT document_id,title,year FROM documents')]
    resolution = []
    for year, title in TARGETS:
        matches = [d for d in documents if _normalize(title) in _normalize(d['title'])]
        exact = [d for d in matches if year in str(d['year']) or year in d['title']]
        resolution.append({'requested_year': year, 'requested_title': title,
                           'title_matches': matches, 'title_and_year_matches': exact})
    ids = [r['title_and_year_matches'][0]['document_id'] for r in resolution if len(r['title_and_year_matches']) == 1]
    actual = actual_temporal_route(store, query, top_k)
    runs = {}
    for mode in ('lexical', 'dense', 'hybrid', 'hybrid_rerank'):
        if mode == 'dense' and vector is None:
            runs[mode] = {'status': 'not_run', 'reason': 'No cached query embedding supplied.'}
            continue
        diagnostics = {}
        result = retrieve_evidence(store, query, top_k=top_k, query_embedding=vector, mode=mode, diagnostics=diagnostics)
        runs[mode] = {'diagnostics': diagnostics, 'chunks': result,
                      'chesapeake_hits': [a for a in result if 'chesapeake' in (a.title+' '+a.original_text_chunk).casefold()],
                      'resolved_target_coverage': sorted({a.document_id for a in result} & set(ids))}
    if len(ids) == 2 and len(set(ids)) == 2:
        diagnostics = {}
        result = retrieve_evidence(store, query, top_k=top_k, query_embedding=vector, document_ids=ids, diagnostics=diagnostics)
        runs['hard_scope_counterfactual'] = {'diagnostics': diagnostics, 'chunks': result,
            'missing_target_ids': sorted(set(ids) - {a.document_id for a in result})}
        runs['per_document_counterfactual'] = {
            doc_id: retrieve_evidence(store, query, top_k=top_k, query_embedding=vector, document_ids=[doc_id])
            for doc_id in ids}

    else:
        runs['hard_scope_counterfactual'] = {'status': 'not_run', 'reason': 'Both exact title/year targets must resolve uniquely. Inspect catalog aliases and year mismatches.'}
    return {'query': query, 'actual_temporal_route': actual, 'target_resolution': resolution,
            'catalog': documents, 'retrieval_ablations': runs,
            'semantic_available': vector is not None,
            'findings': {
                'fusion': f'Equal RRF votes: 1/({SETTINGS.retrieval.rrf_k}+rank), recovered lexical plus dense. Raw scores are not mixed.',
                'ner': 'Retrieval resolves unique catalog titles/years and explicit DOC IDs; comparisons establish a hard target scope.',
                'filter': 'document_ids and recognized comparison targets are enforced; coverage reservation and marginal decay select across targets.',
                'routing': 'Recognized title years are masked before temporal classification. Independent current/as-of requests remain temporal; ordinary comparisons synthesize.',
                'limitation': 'No embedding means lexical-only hybrid votes; this cannot establish the effect of semantic weighting.'}}

if __name__ == '__main__':
    args = arguments(Q4)
    vector = json.loads(args.embedding_json.read_text(encoding='utf-8')) if args.embedding_json else None
    with copied_store(args.database) as store, PipelineTracer(args.output) as tracer:
        capture('original_query', args.query)
        report = diagnose(store, args.query, args.top_k, vector)
        capture('q4_diagnostic', report)
    report['trace_path'] = str(tracer.path)
    save_report(args, f'q4-{tracer.trace_id}.json', report)
