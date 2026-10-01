"""Tests for conventional page-level retrieval baselines."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_conventional_retrievers import (
    Hit,
    Page,
    Query,
    RetrievalInputError,
    conventional_outputs_complete,
    load_pages,
    load_queries,
    rank_scores,
    reciprocal_rank_fusion,
    run_dense,
    write_run,
)


class FakeEncoder:
    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        vectors = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "alpha question": [1.0, 0.0],
        }
        return np.asarray([vectors[text] for text in sentences], dtype=np.float32)


def test_load_inputs_and_reject_duplicate_page_ids(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"page_id":"DOC001_P0001","page_text":"alpha"}\n'
        '{"page_id":"DOC002_P0001","page_text":"beta"}\n',
        encoding="utf-8",
    )
    questions = tmp_path / "questions.csv"
    with questions.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("query_id", "query"))
        writer.writeheader()
        writer.writerow({"query_id": "Q1", "query": "alpha question"})

    assert load_pages(corpus) == [
        Page("DOC001_P0001", "alpha"),
        Page("DOC002_P0001", "beta"),
    ]
    assert load_queries(questions) == [Query("Q1", "alpha question")]

    corpus.write_text(
        '{"page_id":"DOC001_P0001","page_text":"a"}\n'
        '{"page_id":"DOC001_P0001","page_text":"b"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RetrievalInputError, match="Duplicate page_id"):
        load_pages(corpus)


def test_score_ranking_uses_page_id_as_deterministic_tiebreaker() -> None:
    hits = rank_scores(
        ["DOC002_P0001", "DOC001_P0001", "DOC003_P0001"],
        np.asarray([0.5, 0.5, 0.1]),
    )
    assert [hit.page_id for hit in hits] == [
        "DOC001_P0001",
        "DOC002_P0001",
        "DOC003_P0001",
    ]


def test_dense_cosine_ranking_encodes_pages_once() -> None:
    pages = [Page("P1", "alpha"), Page("P2", "beta")]
    rankings = run_dense(
        pages,
        [Query("Q1", "alpha question")],
        model=FakeEncoder(),
        batch_size=8,
    )
    assert [hit.page_id for hit in rankings["Q1"]] == ["P1", "P2"]
    assert rankings["Q1"][0].score == pytest.approx(1.0)


def test_rrf_uses_complete_one_based_ranks_and_stable_ties() -> None:
    bm25 = {
        "Q1": [Hit("P1", 10.0), Hit("P2", 9.0), Hit("P3", 8.0)]
    }
    dense = {
        "Q1": [Hit("P2", 0.9), Hit("P3", 0.8), Hit("P1", 0.7)]
    }
    fused = reciprocal_rank_fusion(bm25, dense, constant=60)["Q1"]

    assert [hit.page_id for hit in fused] == ["P2", "P1", "P3"]
    expected_p2 = 1.0 / 62 + 1.0 / 61
    assert fused[0].score == pytest.approx(expected_p2)


def test_write_run_emits_requested_schema_and_top_k(tmp_path: Path) -> None:
    queries = [Query("Q1", "question")]
    rankings = {
        "Q1": [Hit(f"DOC001_P{number:04d}", 1.0 / number) for number in range(1, 26)]
    }
    output = tmp_path / "candidate_runs" / "bm25.jsonl"
    count = write_run(output, queries, rankings, system="bm25", top_k=20)

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert count == 20
    assert len(records) == 20
    assert records[0] == {
        "query_id": "Q1",
        "page_id": "DOC001_P0001",
        "rank": 1,
        "score": 1.0,
        "system": "bm25",
    }
    assert records[-1]["rank"] == 20
    assert set(records[0]) == {"query_id", "page_id", "rank", "score", "system"}

def test_resume_validation_requires_three_complete_runs(tmp_path: Path) -> None:
    pages = [
        Page(f"DOC001_P{number:04d}", f"page {number}")
        for number in range(1, 26)
    ]
    queries = [Query("Q1", "question")]
    rankings = {
        "Q1": [
            Hit(page.page_id, 1.0 / rank)
            for rank, page in enumerate(pages, start=1)
        ]
    }
    output_dir = tmp_path / "candidate_runs"
    for system in ("bm25", "dense", "hybrid"):
        write_run(
            output_dir / f"{system}.jsonl",
            queries,
            rankings,
            system=system,
        )

    assert conventional_outputs_complete(
        output_dir,
        pages=pages,
        queries=queries,
    )

    with (output_dir / "hybrid.jsonl").open("a", encoding="utf-8") as output:
        output.write('{"malformed":"extra record"}\n')
    assert not conventional_outputs_complete(
        output_dir,
        pages=pages,
        queries=queries,
    )
