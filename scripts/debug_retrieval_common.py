"""Offline RCA helpers. Derived indexes are written only to a temporary DB copy."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_tracer import PipelineTracer, capture, _encode
Q1 = "Which document in the corpus provides the most current guidance on Missouri wetland management?"
Q4 = "How does the 2022 Missouri Comprehensive Conservation Strategy differ from the 2015 Missouri State Wildlife Action Plan in its treatment of aquatic invasive species?"

def arguments(query):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=ROOT / 'data/corpus.db')
    parser.add_argument('--output', type=Path, default=ROOT / 'traces')
    parser.add_argument('--query', default=query)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--embedding-json', type=Path, help='Cached embedding for this exact query and index model; no API calls')
    return parser.parse_args()

@contextmanager
def copied_store(path):
    from database import KnowledgeStore
    path = path.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='v3-rca-') as directory:
        destination = Path(directory) / 'corpus.db'
        source = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
        target = sqlite3.connect(destination)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        with KnowledgeStore(destination) as store:
            yield store

def actual_temporal_route(store, query, top_k):
    from temporal import detect_temporal_intent, select_temporal_evidence, render_temporal_answer
    intent = detect_temporal_intent(query, documents=[dict(r) for r in store.connection.execute('SELECT document_id,title,year FROM documents')])
    capture('temporal_intent', intent)
    if intent.mode == 'none':
        return {'intent': intent.mode, 'note': 'Normal synthesis route; not executed in offline diagnostics.'}
    selection = select_temporal_evidence(query, store, top_k=top_k, intent=intent)
    rendered = render_temporal_answer(selection, store)
    capture('temporal_selection', selection)
    return {'intent': intent.mode, 'selection': selection, 'answer': rendered[0], 'preamble': rendered[1]}

def save_report(args, name, report):
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / name
    path.write_text(json.dumps(report, default=_encode, ensure_ascii=False, indent=2), encoding='utf-8')
    print(path)
