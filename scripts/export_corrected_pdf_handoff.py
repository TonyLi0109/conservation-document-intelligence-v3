"""Export admitted QGCG candidates separately from comprehensive retrieval audit logs."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_pdf_benchmark as run
from scripts.report_revised_benchmark import make_report
from scripts.validate_pdf_benchmark import validate


POLICY = {
    "cdirp_v3": "Native reranked evidence first, then generation/final evidence, then fused_ranking entries in event/stage order; deduplicate by physical page. This is an export ordering, not a native single global relevance score.",
    "excluded_from_admitted_pool": "Raw dense_search, lexical_search, temporal_selection.evidence/decisions, and unselected intermediate candidate lists remain in the audit outputs only. Temporal artifacts are included through reranked_evidence.",
    "other_methods": "All accepted model-selected pages, ordered by maximum retrieval score, then earliest batch/rank, as saved by the runner.",
    "top100": "First min(100, admitted page count) per question/method; no padding. Full admitted union and Top-100 union are exported separately.",
    "scores": "Retrieval-stage scores only; no independent relevance judgments or gold labels.",
    "scope": "Four original methods only. No standalone BM25, Dense, Hybrid RRF experiments or three-model relevance assessment have been performed.",
}


def admitted_rows(method, rows, occurrences):
    if method != "cdirp_v3":
        return rows
    priority = {"reranked_evidence": 0, "generation_evidence_handles": 1, "final_output": 2}
    selected = [x for x in occurrences if x['stage'] in priority or (
        x['stage'] == 'retrieval_ranking' and x['stage_path'].endswith('.fused_ranking'))]
    selected.sort(key=lambda x: priority.get(x['stage'], 3))
    result, seen = [], set()
    for row in selected:
        if row['page_id'] not in seen:
            seen.add(row['page_id'])
            result.append({**row, 'rank': len(result)+1, 'rank_policy': POLICY['cdirp_v3']})
    return result


def export(root):
    complete = make_report(root, runner=run)
    validate(root)
    if not complete:
        raise RuntimeError('Run incomplete: audit report updated; no frozen handoff exported.')
    out = root / 'handoff'
    out.mkdir(exist_ok=True)
    pages = {p['page_id']: p for p in map(json.loads, (root/'inputs/corpus.jsonl').read_bytes().splitlines())}
    questions = run.read(root/'inputs/questions.json')
    assert questions == run.docx_rows(root/'inputs/Revised_9_Questions.docx')
    for name in ['questions.json', 'questions.csv', 'Revised_9_Questions.docx', 'pdf_inventory.json']:
        shutil.copy2(root/'inputs'/name, out/name)
    shutil.copy2(root/'source_validation.json', out/'source_validation.json')
    pools, limited, counts, answers = {}, {}, [], []
    for question in questions:
        qid = question['query_id']
        union, top_union = {}, {}
        for method in run.ALL_METHODS:
            folder = root/'runs'/method/qid
            for filename in ['retrieval_results.json', 'candidate_occurrences.json', 'status.json']:
                dest = out/'audit'/method/qid/filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(folder/filename, dest)
            rows = admitted_rows(method, run.read(folder/'retrieval_results.json'), run.read(folder/'candidate_occurrences.json'))
            assert len({x['page_id'] for x in rows}) == len(rows)
            assert all(x['page_id'] in pages for x in rows)
            run.save(out/'rankings'/method/f'{qid}.json', {'question': question, 'policy': POLICY, 'candidates': rows})
            run.save(out/'top100'/method/f'{qid}.json', {'question': question, 'policy': POLICY, 'candidates': rows[:100]})
            for i, row in enumerate(rows):
                provenance = {**row, 'method': method}
                union.setdefault(row['page_id'], []).append(provenance)
                if i < 100:
                    top_union.setdefault(row['page_id'], []).append(provenance)
            status = run.read(folder/'status.json')
            answers.append({'question': question, 'method': method, 'status': status, 'answer': run.read(folder/'answer.json')})
            counts.append({'question_id': qid, 'method': method, 'admitted': len(rows), 'top100': min(100,len(rows)), 'answer_status': status['status']})
        pools[qid] = [{'page_id': pid, 'provenance': union[pid]} for pid in sorted(union)]
        limited[qid] = [{'page_id': pid, 'provenance': top_union[pid]} for pid in sorted(top_union)]
        for directory, selected in [('pools', union), ('top100_pools', top_union)]:
            dest = out/directory/qid/'pages.jsonl'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(''.join(json.dumps({**pages[pid], 'retrieved_by': selected[pid]}, ensure_ascii=False)+'\n' for pid in sorted(selected)), encoding='utf-8')
    run.save(out/'candidate_pool.json', {'status': 'frozen', 'policy': POLICY, 'questions': questions, 'queries': pools})
    run.save(out/'top100_candidate_pool.json', {'status': 'frozen', 'policy': POLICY, 'questions': questions, 'queries': limited})
    run.save(out/'all_answers.json', answers)
    lines = ['CORRECTED PDF RETRIEVAL HANDOFF', '', 'DOC014: corrected 40-page Habitat Matters, 2025 Canadian NAWMP report.',
             'Source: 36 PDFs, 2036 physical pages. Question texts and source DOCX included.',
             'Use candidate_pool.json with pools/ for the full admitted QGCG pool.',
             'Use top100_candidate_pool.json with top100_pools/ for the per-method Top-100 union.',
             'Do not mix either pool with the old 20261020 corpus or historical qrels.',
             'The experiment-root candidate_pool.json is a comprehensive AUDIT union, not this admitted QGCG pool.',
             'Original stage occurrences and original rankings are included under audit/; full execution traces remain in the experiment runs/ directory.', '', *POLICY.values(), '',
             'Question | Full admitted union | Top-100 union']
    lines += [f"{q} | {len(pools[q])} | {len(limited[q])}" for q in pools]
    (out/'README.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    run.save(out/'handoff_manifest.json', {'run_id': root.name, 'created_at_utc': run.now(), 'status': 'frozen', 'policy': POLICY,
        'method_counts': counts, 'source_validation': run.read(root/'source_validation.json'),
        'files_sha256': {p.relative_to(out).as_posix(): run.sha(p) for p in out.rglob('*') if p.is_file() and p.name!='handoff_manifest.json'}})
    for rel, digest in run.read(out/'handoff_manifest.json')['files_sha256'].items():
        assert run.sha(out/rel) == digest
    with zipfile.ZipFile(root/'corrected_qgcg_handoff.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in out.rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(out))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    export(parser.parse_args().root.resolve())
