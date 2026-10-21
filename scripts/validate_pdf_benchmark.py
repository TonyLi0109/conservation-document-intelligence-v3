"""Validate a quiescent PDF benchmark report and every retained candidate."""
from collections import Counter
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_pdf_benchmark as run


def validate(root=run.DEFAULT):
    manifest = run.verify(root)
    hashes = {**manifest['available_output_sha256'], **manifest['pool_text_sha256'],
              **manifest['review_report_sha256']}
    for path, expected in hashes.items():
        if run.sha(root / path) != expected:
            raise ValueError(f'Reported file changed: {path}')
    if run.sha(root / 'candidate_pool.json') != manifest['candidate_pool_sha256']:
        raise ValueError('Candidate pool changed')
    pool = run.read(root / 'candidate_pool.json')
    review = run.read(root / 'review_summary.json')
    canonical = {p['page_id'] for p in map(json.loads, (root / 'inputs/corpus.jsonl').read_bytes().splitlines())}
    for qid in run.QIDS:
        expected = Counter()
        for method in run.ALL_METHODS:
            folder = root / 'runs' / method / qid
            path = folder / 'candidate_occurrences.json'
            rows = run.read(path) if path.exists() else []
            expected.update((x['page_id'], method, x['stage'], x['stage_rank']) for x in rows)
            if method != 'cdirp_v3':
                selected = Counter()
                for p in folder.glob('calls/retrieve_*/accepted.json'):
                    selected.update((p.parent.name, x['page_id'], x['score']) for x in run.read(p)['candidates'])
                if selected != Counter((x['stage'], x['page_id'], x['score']) for x in rows):
                    raise ValueError(f'Accepted candidate missing: {method}/{qid}')
        ids = [x['page_id'] for x in pool['queries'][qid]]
        if len(ids) != len(set(ids)) or not set(ids) <= canonical:
            raise ValueError(f'Invalid canonical union: {qid}')
        actual = Counter((x['page_id'], p['method'], p['stage'], p['stage_rank'])
                         for x in pool['queries'][qid] for p in x['provenance'])
        if actual != expected:
            raise ValueError(f'Candidate occurrence missing from union: {qid}')
    rows = review['query_method_results']
    for row in rows:
        path = root / 'runs' / row['method'] / row['query_id'] / 'status.json'
        current = run.read(path)['status'] if path.exists() else 'not_run'
        if row['status'] != current:
            raise ValueError('Run changed during report generation; regenerate after workers stop')
    audit = {'checked_at_utc': run.now(), 'verified_report_and_output_hashes': len(hashes),
             'terminal_outputs': sum(x['terminal_output_available'] for x in rows),
             'successful_answers': sum(x['status'] == 'completed' for x in rows),
             'native_fallback_answers': sum(x['method'] == 'cdirp_v3' and x['terminal_output_available']
                                           and x['status'] == 'failed' for x in rows),
             'pending_questions': [x['method'] + '/' + x['query_id'] for x in rows if not x['terminal_output_available']],
             'all_accepted_retrieval_candidates_retained': True,
             'all_candidate_occurrences_in_union': True, 'canonical_deduplicated_unions': True,
             'pool_status': pool['status']}
    run.save(root / 'available_results_validation.json', audit)
    manifest = run.read(root / 'manifest.json')
    manifest['available_results_validation_sha256'] = run.sha(root / 'available_results_validation.json')
    manifest['result_validator_sha256'] = run.sha(Path(__file__))
    run.save(root / 'manifest.json', manifest)
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == '__main__':
    validate()
