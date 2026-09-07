"""Offline fault-boundary and retrieval regressions using temporary source files."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

import main
import pdf_parser
from data_models import DocumentSource, KnowledgeArtifact
from database import KnowledgeStore


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    """Use the real text parser and rebuild pipeline with an offline embedder."""
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source_path = source_dir / "DOC901_Carp.txt"
    source_path.write_text("Invasive carp harvest removed fish from a river.", encoding="utf-8")
    source = DocumentSource(
        "DOC901", "Synthetic carp source", "https://example.org/carp", None,
        source_path.name, "txt", year="2020", agency="Synthetic Agency",
    )
    catalog = {source.document_id: source}
    calls = []

    def embed(texts, **kwargs):
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(main, "load_source_catalog", lambda: catalog)
    monkeypatch.setattr(main, "generate_embeddings", embed)
    with KnowledgeStore(tmp_path / "ingestion.db") as store:
        yield source_dir, source_path, catalog, calls, store


def test_repeated_ingestion_replaces_index_without_duplicate_chunks(source_tree):
    directory, _, _, calls, store = source_tree
    first_count = main.ingest_corpus(str(directory), store)
    first_artifacts = store.retrieve([1.0, 0.0], 10)
    first_hash = store.connection.execute(
        "SELECT file_sha256 FROM document_ingestion_reports"
    ).fetchone()[0]

    assert main.ingest_corpus(str(directory), store) == first_count == 1
    assert store.artifact_count == 1
    assert store.retrieve([1.0, 0.0], 10) == first_artifacts
    assert store.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert store.connection.execute("SELECT COUNT(*) FROM vector_embeddings").fetchone()[0] == 1
    assert store.connection.execute(
        "SELECT file_sha256 FROM document_ingestion_reports"
    ).fetchone()[0] == first_hash
    # Rebuild currently re-embeds; idempotency concerns output, not API caching.
    assert len(calls) == 2


def test_updated_source_replaces_text_metadata_and_version_digest(source_tree):
    directory, source_path, catalog, _, store = source_tree
    main.ingest_corpus(str(directory), store)
    previous_hash = store.connection.execute(
        "SELECT file_sha256 FROM document_ingestion_reports"
    ).fetchone()[0]
    new_text = "Updated monitoring found invasive carp recruitment after harvest."
    source_path.write_text(new_text, encoding="utf-8")
    catalog["DOC901"] = replace(
        catalog["DOC901"], title="Revised synthetic carp source", year="2021",
        resolved_url="https://example.org/carp-revised.txt",
    )

    main.ingest_corpus(str(directory), store)

    evidence = store.retrieve([1.0, 0.0], 10)
    assert len(evidence) == 1 and evidence[0].original_text_chunk == new_text
    assert evidence[0].title == "Revised synthetic carp source"
    assert evidence[0].source_url == "https://example.org/carp-revised.txt"
    metadata = store.connection.execute("SELECT * FROM documents").fetchone()
    assert metadata["year"] == "2021"
    new_hash = store.connection.execute(
        "SELECT file_sha256 FROM document_ingestion_reports"
    ).fetchone()[0]
    assert new_hash == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert new_hash != previous_hash


def test_empty_source_rebuild_preserves_previous_searchable_evidence(source_tree):
    directory, source_path, _, calls, store = source_tree
    main.ingest_corpus(str(directory), store)
    original = store.retrieve([1.0, 0.0], 10)
    original_reports = store.list_ingestion_reports()
    source_path.write_text(" \n\t\x00", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no readable chunks"):
        main.ingest_corpus(str(directory), store)

    assert store.retrieve([1.0, 0.0], 10) == original
    assert store.list_ingestion_reports() == original_reports
    assert store.connection.execute("SELECT COUNT(*) FROM staged_knowledge_artifacts").fetchone()[0] == 0
    assert len(calls) == 1


def test_duplicate_document_files_fail_before_embedding_or_index_replacement(source_tree):
    directory, _, _, calls, store = source_tree
    main.ingest_corpus(str(directory), store)
    original = store.retrieve([1.0, 0.0], 10)
    (directory / "DOC901_Second_Copy.txt").write_text("Another copy.", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate source files for DOC901"):
        main.ingest_corpus(str(directory), store)

    assert len(calls) == 1
    assert store.retrieve([1.0, 0.0], 10) == original


def test_missing_catalog_entry_fails_without_discarding_live_index(source_tree):
    directory, _, _, calls, store = source_tree
    main.ingest_corpus(str(directory), store)
    original = store.retrieve([1.0, 0.0], 10)
    (directory / "DOC902_Unknown.txt").write_text("Unknown source text.", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not in catalog: DOC902"):
        main.ingest_corpus(str(directory), store)

    assert len(calls) == 1
    assert store.retrieve([1.0, 0.0], 10) == original


def test_embedding_failure_after_staging_rolls_back_entire_rebuild(source_tree, monkeypatch):
    directory, source_path, _, _, store = source_tree
    main.ingest_corpus(str(directory), store)
    original = store.retrieve([1.0, 0.0], 10)
    source_path.write_text(" ".join(f"token{i}" for i in range(1800)), encoding="utf-8")
    monkeypatch.setattr(main, "INGESTION_BATCH_SIZE", 1)
    completed_batches = []

    def interrupted_embed(texts, **kwargs):
        completed_batches.append(list(texts))
        if len(completed_batches) == 2:
            raise TimeoutError("Synthetic embedding interruption")
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(main, "generate_embeddings", interrupted_embed)
    with pytest.raises(TimeoutError, match="Synthetic embedding interruption"):
        main.ingest_corpus(str(directory), store)

    assert len(completed_batches) == 2
    assert store.retrieve([1.0, 0.0], 10) == original
    assert store.connection.execute("SELECT COUNT(*) FROM staged_knowledge_artifacts").fetchone()[0] == 0
    assert store.connection.execute("SELECT COUNT(*) FROM staged_vector_embeddings").fetchone()[0] == 0


@pytest.mark.parametrize("contents", [b"", b"%PDF-1.7\ntruncated broken objects\n%%EOF"])
def test_malformed_or_zero_byte_pdf_returns_no_invented_evidence(tmp_path, contents):
    path = tmp_path / "broken.pdf"
    path.write_bytes(contents)
    report = pdf_parser.DocumentParseReport()

    assert list(pdf_parser.parse_pdf_to_artifacts(str(path), "DOC901", "Broken PDF", report)) == []
    assert report.chunk_count == 0 and report.extracted_pages == 0


def test_real_blank_pdf_tracks_empty_page_without_producing_chunks(tmp_path):
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(path)
    report = pdf_parser.DocumentParseReport()

    assert list(pdf_parser.parse_pdf_to_artifacts(str(path), "DOC901", "Blank PDF", report)) == []
    assert report.total_pages == 1
    assert report.empty_pages == [1]
    assert not report.failed_pages and report.chunk_count == 0


def test_long_pdf_preserves_page_attribution_overlap_and_failure_telemetry(tmp_path, monkeypatch):
    path = tmp_path / "long.pdf"
    writer = PdfWriter()
    for _ in range(15):
        writer.add_blank_page(width=72, height=72)
    writer.write(path)
    expected = []
    for page in range(1, 16):
        if page not in (3, 5):
            expected.extend((f"p{page}word{word}", str(page)) for word in range(180))

    def extract(_path, index):
        page = index + 1
        if page == 3:
            return None, False
        if page == 5:
            return "   ", False
        return " ".join(f"p{page}word{word}" for word in range(180)), page == 7

    # Isolate extraction backend faults; exercise real page enumeration/chunking.
    monkeypatch.setattr(pdf_parser, "_extract_page_in_subprocess", extract)
    report = pdf_parser.DocumentParseReport()
    artifacts = list(pdf_parser.parse_pdf_to_artifacts(str(path), "DOC901", "Long PDF", report))
    reconstructed = []
    offset = 0
    for ordinal, artifact in enumerate(artifacts):
        words = artifact.original_text_chunk.split()
        assert artifact.page_number == expected[offset][1]
        assert 0 < len(words) <= pdf_parser.TARGET_WORDS
        if ordinal:
            assert words[:pdf_parser.OVERLAP_WORDS] == reconstructed[-pdf_parser.OVERLAP_WORDS:]
            reconstructed.extend(words[pdf_parser.OVERLAP_WORDS:])
        else:
            reconstructed.extend(words)
        offset += pdf_parser.TARGET_WORDS - pdf_parser.OVERLAP_WORDS
    assert reconstructed == [word for word, _ in expected]
    assert reconstructed[-1] == "p15word179"
    assert report.total_pages == 15 and report.extracted_pages == 13
    assert report.failed_pages == [3] and report.empty_pages == [5]
    assert report.fallback_pages == [7]
    assert report.chunk_count == len(artifacts) > 1


@pytest.mark.parametrize("parser", [pdf_parser.parse_pdf_to_artifacts, pdf_parser.parse_text_to_artifacts])
def test_missing_title_is_rejected_at_ingestion_provenance_boundary(tmp_path, parser):
    with pytest.raises(ValueError, match="title must be a non-empty string"):
        list(parser(str(tmp_path / "unused.pdf"), "DOC901", " "))


@pytest.fixture
def retrieval_store(tmp_path):
    payload = json.loads((Path(__file__).resolve().parents[1] / "evaluation/datasets/fixture_corpus.json").read_text(encoding="utf-8"))
    with KnowledgeStore(tmp_path / "retrieval.db") as store:
        for row in payload["documents"]:
            store.upsert_document_sources([DocumentSource(
                row["document_id"], row["title"], row["source_url"], None,
                row["document_id"] + ".pdf", "pdf",
            )])
            store.ingest_chunk(KnowledgeArtifact(**{
                key: value for key, value in row.items() if key != "embedding"
            }), row["embedding"])
        yield store


def test_exact_title_keyword_lookup_retains_canonical_metadata(retrieval_store):
    matches = retrieval_store.retrieve(
        None, 1, method="keyword", query_text="Synthetic invasive carp acoustic barriers study",
    )
    assert [item.document_id for item in matches] == ["DOC902"]
    assert matches[0].page_number == "2"
    assert matches[0].source_url == "https://example.org/synthetic/carp-acoustics.pdf"
    assert "50%" in matches[0].original_text_chunk


def test_keyword_no_results_and_long_repeated_query_do_not_corrupt_index(retrieval_store):
    assert retrieval_store.retrieve(None, 5, method="keyword", query_text="xylophonicquasarzz") == []
    assert retrieval_store.retrieve_document_matches("xylophonicquasarzz", 5) == []
    query = "invasive carp harvest"
    short = retrieval_store.retrieve(None, 5, method="keyword", query_text=query)
    long = retrieval_store.retrieve(None, 5, method="keyword", query_text=(query + " ") * 1000)
    assert {item.document_id for item in short} == {item.document_id for item in long}
    assert retrieval_store.artifact_count == 10


def test_document_lookup_preserves_diversity_with_duplicate_chunks(retrieval_store):
    first = retrieval_store.retrieve([1.0, 0.0, 0.0], 1)[0]
    for _ in range(6):
        retrieval_store.ingest_chunk(first, [1.0, 0.0, 0.0])

    matches = retrieval_store.retrieve_document_matches("invasive carp", 5)
    ids = [item.document_id for item in matches]
    assert len(ids) == len(set(ids))
    assert {"DOC901", "DOC902", "DOC908"} <= set(ids)
    assert first.original_text_chunk == next(item.original_text_chunk for item in matches if item.document_id == "DOC901")


def test_quantitative_retrieval_retains_contrasting_relevant_sources(retrieval_store):
    matches = retrieval_store.retrieve(None, 5, method="keyword", query_text="invasive carp commercial harvest abundance")
    evidence = {item.document_id: item for item in matches}
    assert "DOC901" in evidence and "DOC908" in evidence
    assert "60%" in evidence["DOC901"].original_text_chunk
    assert "no statistically significant change" in evidence["DOC908"].original_text_chunk
    assert evidence["DOC901"].source_url != evidence["DOC908"].source_url


def test_invalid_query_vector_fails_without_poisoning_semantic_cache(retrieval_store):
    expected = retrieval_store.retrieve([1.0, 0.0, 0.0], 3)
    with pytest.raises(ValueError, match="dimension"):
        retrieval_store.retrieve([1.0, 0.0], 3)
    with pytest.raises(ValueError, match="finite"):
        retrieval_store.retrieve([float("nan"), 1.0, 0.0], 3)
    assert retrieval_store.retrieve([1.0, 0.0, 0.0], 3) == expected
    assert {item.document_id for item in expected} == {"DOC901", "DOC902", "DOC908"}
