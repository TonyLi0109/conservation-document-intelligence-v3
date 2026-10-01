"""Page-level benchmark extraction is deterministic and fail-closed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_page_corpus import CorpusExtractionError, build_page_corpus, fitz


def make_pdf(path: Path, pages: list[str]) -> None:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_page_dataset_has_stable_ids_order_and_parent_checksums(tmp_path):
    inputs = tmp_path / "pdfs"
    inputs.mkdir()
    make_pdf(inputs / "DOC002.pdf", ["Second document"])
    make_pdf(inputs / "DOC001.pdf", ["First page", "Second page"])
    output = tmp_path / "benchmark" / "corpus_pages.jsonl"

    documents, pages = build_page_corpus(
        inputs, output, expected_documents=2
    )
    rows = read_jsonl(output)

    assert (documents, pages) == (2, 3)
    assert [row["page_id"] for row in rows] == [
        "DOC001_P0001",
        "DOC001_P0002",
        "DOC002_P0001",
    ]
    assert [row["pdf_page"] for row in rows] == [1, 2, 1]
    assert all(type(row["pdf_page"]) is int for row in rows)
    assert "First page" in rows[0]["page_text"]
    expected_hash = hashlib.sha256((inputs / "DOC001.pdf").read_bytes()).hexdigest()
    assert rows[0]["file_checksum"] == rows[1]["file_checksum"] == expected_hash
    assert set(rows[0]) == {
        "page_id", "doc_id", "pdf_page", "page_text", "file_checksum"
    }


def test_blank_pdf_page_is_retained(tmp_path):
    inputs = tmp_path / "pdfs"
    inputs.mkdir()
    make_pdf(inputs / "DOC001.pdf", [""])
    output = tmp_path / "corpus_pages.jsonl"

    build_page_corpus(inputs, output, expected_documents=1)

    assert read_jsonl(output)[0]["page_text"] == ""


def test_descriptive_filenames_and_text_snapshots_are_supported(tmp_path):
    inputs = tmp_path / "sources"
    inputs.mkdir()
    make_pdf(inputs / "DOC001_First_Source.pdf", ["PDF page"])
    text_path = inputs / "DOC002_Web_Snapshot.txt"
    text_path.write_text("Web snapshot\\n\\ncontent", encoding="utf-8")
    output = tmp_path / "corpus_pages.jsonl"

    documents, pages = build_page_corpus(inputs, output, expected_documents=2)
    rows = read_jsonl(output)

    assert (documents, pages) == (2, 2)
    assert rows[1]["page_id"] == "DOC002_P0001"
    assert rows[1]["pdf_page"] == 1
    assert rows[1]["page_text"] == "Web snapshot\\n\\ncontent"
    assert rows[1]["file_checksum"] == hashlib.sha256(
        text_path.read_bytes()
    ).hexdigest()


def test_missing_or_unexpected_document_ids_fail_before_writing(tmp_path):
    inputs = tmp_path / "pdfs"
    inputs.mkdir()
    make_pdf(inputs / "DOC001.pdf", ["Valid"])
    make_pdf(inputs / "DOC003.pdf", ["Unexpected"])
    output = tmp_path / "corpus_pages.jsonl"

    with pytest.raises(CorpusExtractionError, match="missing=DOC002"):
        build_page_corpus(inputs, output, expected_documents=2)

    assert not output.exists()


def test_corrupt_pdf_preserves_previous_atomic_output(tmp_path):
    inputs = tmp_path / "pdfs"
    inputs.mkdir()
    make_pdf(inputs / "DOC001.pdf", ["Valid"])
    (inputs / "DOC002.pdf").write_bytes(b"not a PDF")
    output = tmp_path / "benchmark" / "corpus_pages.jsonl"
    output.parent.mkdir()
    output.write_text("previous-dataset\n", encoding="utf-8")

    with pytest.raises(CorpusExtractionError, match="Could not open"):
        build_page_corpus(inputs, output, expected_documents=2)

    assert output.read_text(encoding="utf-8") == "previous-dataset\n"
    assert not list(output.parent.glob(".corpus_pages.jsonl.*.tmp"))
