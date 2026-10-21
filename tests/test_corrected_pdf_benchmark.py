import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_pdf_benchmark as run


class RevisedBenchmarkTests(unittest.TestCase):
    def test_revised_column_and_substantive_changes(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"questions.docx"
            body = ''.join(f'<w:p><w:r><w:t>Q{i}</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>OLD QUESTION</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>REVISED QUESTION</w:t></w:r></w:p></w:tc></w:tr><w:tr><w:tc><w:p><w:r><w:t>Old {i}</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Revised {i}</w:t></w:r></w:p></w:tc></w:tr></w:tbl>' for i in range(1,10))
            xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+body+'</w:body></w:document>'
            with zipfile.ZipFile(path,'w') as z: z.writestr('word/document.xml',xml)
            rows = run.docx_rows(path)
            self.assertEqual([r['query_id'] for r in rows], list(run.QIDS))
            self.assertEqual([r['query'] for r in rows], [f'Revised {i}' for i in range(1,10)])

    def test_batch_windows_cover_long_pages_without_truncation(self):
        text = "abcdefghij" * 3201
        pages = [{"page_id": "DOC001_P0001", "page_text": text}]
        covered = set()
        for batch in run.batches(pages, budget=13000):
            for window in batch:
                self.assertEqual(window["text"], text[window["start"]:window["start"]+12000])
                covered.update(range(window["start"], window["start"]+len(window["text"])))
        self.assertEqual(covered, set(range(len(text))))

    def test_out_of_batch_and_document_citations_rejected(self):
        payload = {"pages": [{"page_id": "DOC001_P0001"}]}
        for pid in ("DOC001", "DOC001_P0002", "https://example.com"):
            with self.assertRaises(ValueError):
                run.validate_output({"candidates": [{"page_id": pid, "score": 4}]}, run.CANDIDATES, payload)
        with self.assertRaises(ValueError):
            run.validate_output({"answer_markdown": "Claim [DOC001_P0002]", "cited_page_ids": ["DOC001_P0001"],
                                 "evidence_gaps": []}, run.ANSWER, payload)

    def test_partial_union_keeps_nonfinal_candidates_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run.save(root / "manifest.json", {"input_sha256": {"questions.csv": "q", "corpus.jsonl": "c"}})
            (root / "inputs").mkdir(exist_ok=True)
            page = {"page_id": "DOC001_P0001", "doc_id": "DOC001", "pdf_page": 1, "page_text": "evidence"}
            (root / "inputs/corpus.jsonl").write_text(json.dumps(page)+"\n")
            for method in ("gpt6_luna", "claude_haiku55"):
                folder = root / "runs" / method / "Q1"
                run.save(folder / "candidate_occurrences.json", [
                    {"page_id": page["page_id"], "method": method, "stage_rank": 1}])
            self.assertFalse(run.report(root))
            pool = run.read(root / "candidate_pool.json")
            self.assertEqual(pool["status"], "partial_not_frozen")
            self.assertEqual(len(pool["queries"]["Q1"]), 1)
            self.assertEqual(len(pool["queries"]["Q1"][0]["provenance"]), 2)

    def test_production_trace_preserves_stage_rank(self):
        path = run.DATA / "scripts/run_revised_cdirp.py"
        spec = importlib.util.spec_from_file_location("revised_cdirp_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        trace = {"events": [{"stage": "dense_search", "sequence": 1, "value": {
            "hits": [{"artifact_id": 10, "score": .8}, {"artifact_id": 11, "score": .7}]}}]}
        rows = module.collect_candidates(trace, {10: "DOC001_P0001", 11: "DOC001_P0002"},
                                         {"DOC001_P0001", "DOC001_P0002"})
        self.assertEqual([r["stage_rank"] for r in rows], [1, 2])
        self.assertEqual([r["page_id"] for r in rows], ["DOC001_P0001", "DOC001_P0002"])

    def test_cdirp_chunks_stay_on_their_physical_page(self):
        spec = importlib.util.spec_from_file_location("canonical_cdirp_test", run.DATA / "scripts/run_revised_cdirp.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pages = [{"doc_id": "DOC001", "pdf_page": 1, "page_text": "first " * 1000},
                 {"doc_id": "DOC001", "pdf_page": 2, "page_text": "second " * 900}]
        chunks = list(module.canonical_chunks(pages, {"DOC001": "Title"}))
        self.assertGreater(len(chunks), 2)
        for chunk in chunks:
            self.assertIn(chunk.original_text_chunk, pages[int(chunk.page_number)-1]["page_text"])
            self.assertFalse("first" in chunk.original_text_chunk and "second" in chunk.original_text_chunk)


if __name__ == "__main__":
    unittest.main()
