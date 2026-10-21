import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from scripts import run_pdf_benchmark as run
from scripts.export_corrected_pdf_handoff import admitted_rows


class AdmissionTests(unittest.TestCase):
    def test_wrong_one_page_doc014_is_rejected(self):
        import pymupdf
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'PDF').mkdir()
            inputs=root/'inputs'
            inputs.mkdir()
            for i in range(1,37):
                doc=pymupdf.open()
                page=doc.new_page()
                page.insert_text((72,72),'Archived web notice' if i==14 else 'Synthetic test evidence')
                doc.save(root/'PDF'/f'DOC{i:03d}_fixture.pdf')
                doc.close()
            with patch.object(run,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'corrected 40-page'):
                    run.extract_pdf_corpus(inputs)

    def test_audit_scans_do_not_become_admitted_candidates(self):
        rows = [
            {'page_id':'DOC014_P0001','stage':'dense_search','stage_path':'events[0].hits[0]'},
            {'page_id':'DOC001_P0002','stage':'temporal_selection','stage_path':'events[1].evidence[0]'},
            {'page_id':'DOC002_P0001','stage':'reranked_evidence','stage_path':'events[2][0]'},
            {'page_id':'DOC001_P0003','stage':'retrieval_ranking','stage_path':'events[3].fused_ranking'},
            {'page_id':'DOC002_P0001','stage':'final_output','stage_path':'events[4].sources[0]'},
        ]
        selected=admitted_rows('cdirp_v3',[],rows)
        self.assertEqual([r['page_id'] for r in selected],['DOC002_P0001','DOC001_P0003'])
        self.assertEqual([r['rank'] for r in selected],[1,2])
        self.assertEqual(len(rows),5)

    def test_baseline_preserves_every_selected_page(self):
        rows=[{'page_id':f'DOC001_P{i:04d}','rank':i} for i in range(1,121)]
        self.assertEqual(admitted_rows('gpt6_luna',rows,[]),rows)


if __name__=='__main__':unittest.main()
