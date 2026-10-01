# Frozen Page Corpus Access

`benchmark/corpus_pages.jsonl` is intentionally excluded from the public Git repository because it contains full extracted page text from the 36-source corpus.

Frozen artifact identity:

| Field | Value |
|---|---|
| Rows | 1,884 |
| Bytes | 4,584,481 |
| SHA-256 | `df1c18ea6c862e577309990d0b537e072210848b17a31078f6bc0f30a21b9e07` |
| Schema | `page_id`, `doc_id`, `pdf_page`, `page_text`, `file_checksum` |

Project members should obtain the exact frozen artifact through an approved private project channel and verify its SHA-256 before running the judging stage.

To reconstruct it from the approved frozen DOC001-DOC036 source directory:

```powershell
python scripts/build_page_corpus.py .\pdfs --output benchmark\corpus_pages.jsonl --expected-documents 36
Get-FileHash -Algorithm SHA256 benchmark\corpus_pages.jsonl
```

A reconstructed file is equivalent to the frozen input only when its checksum matches the value above. Do not commit the page corpus, source PDFs, API credentials, or full prompt traces to the public repository.
