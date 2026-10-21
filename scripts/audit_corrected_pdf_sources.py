"""Check live PDFs, frozen PDF text, revised questions and approved DOC014 identity."""
import argparse
import json
from pathlib import Path
import sys
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_pdf_benchmark as run


def audit(root):
    manifest = run.verify(root)
    assert run.docx_rows(run.ROOT/'Revised_9_Questions.docx') == run.read(root/'inputs/questions.json')
    corpus = {p['page_id']:p for p in map(json.loads,(root/'inputs/corpus.jsonl').read_bytes().splitlines())}
    for item in manifest['pdf_inventory']:
        print('Checking '+item['doc_id'], flush=True)
        assert run.sha(run.ROOT/'PDF'/item['filename']) == item['sha256']
        with pymupdf.open(root/'inputs/PDF'/item['filename']) as pdf:
            assert len(pdf) == item['physical_pages']
            for number, page in enumerate(pdf,1):
                assert corpus[f"{item['doc_id']}_P{number:04d}"]['page_text'] == page.get_text(sort=True)
    assert len(corpus) == 2036
    assert len([pid for pid in corpus if pid.startswith('DOC014_')]) == 40
    assert '2025 Canadian' in corpus['DOC014_P0001']['page_text']
    run.save(root/'source_validation.json', {'checked_at_utc':run.now(), 'all_pdf_hashes_match_current_PDF':True,
        'all_page_texts_match_frozen_PDF':True, 'questions_match_revised_docx_column':True,
        'physical_pages':len(corpus), 'DOC014_pages':40, 'DOC014_first_page':corpus['DOC014_P0001']['page_text']})
    print('Source audit passed.', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    audit(parser.parse_args().root.resolve())
