"""Tests for Milestone C candidate pool construction and diagnostics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_candidate_pool import (
    CandidatePoolError,
    build_candidate_pool,
    build_pool_records,
    calculate_overlap_matrix,
    jaccard,
    load_candidate_runs,
)


def write_run(path: Path, system: str, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for query_id, page_id, rank in rows:
            output.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "page_id": page_id,
                        "rank": rank,
                        "system": system,
                    }
                )
                + "\n"
            )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_deduplicates_by_query_and_page_and_tracks_all_provenance(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs"
    write_run(
        run_dir / "bm25.jsonl",
        "bm25",
        [
            ("Q1", "DOC036_P0106", 4),
            ("Q1", "DOC001_P0001", 2),
            # Repeated within one run keeps the best rank and does not increase
            # selection_count because provenance counts distinct systems.
            ("Q1", "DOC036_P0106", 3),
        ],
    )
    write_run(
        run_dir / "dense.jsonl",
        "dense",
        [
            ("Q1", "DOC036_P0106", 7),
            ("Q2", "DOC036_P0106", 1),
        ],
    )
    write_run(
        run_dir / "hybrid.jsonl",
        "hybrid",
        [("Q1", "DOC036_P0106", 2)],
    )

    provenance, selections, available, missing = load_candidate_runs(run_dir)
    records = build_pool_records(provenance)

    assert available == ["bm25", "dense", "hybrid"]
    assert missing == ["llm_a", "llm_b", "llm_c", "copilot", "gpt_6_luna", "claude_haiku", "gemini_flash"]
    assert len(records) == 3
    assert records[1] == {
        "query_id": "Q1",
        "page_id": "DOC036_P0106",
        "selection_count": 3,
        "provenance": {"bm25": 3, "dense": 7, "hybrid": 2},
    }
    assert ("Q1", "DOC036_P0106") in selections["bm25"]
    assert ("Q2", "DOC036_P0106") in selections["dense"]


def test_jaccard_uses_query_page_pairs_and_handles_empty_sets() -> None:
    left = {("Q1", "P1"), ("Q1", "P2")}
    right = {("Q1", "P2"), ("Q2", "P1")}

    assert jaccard(left, right) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 0.0

    matrix = calculate_overlap_matrix(
        {"a": left, "b": right, "missing": set()},
        systems=("a", "b", "missing"),
    )
    assert matrix["a"]["a"] == 1.0
    assert matrix["a"]["b"] == pytest.approx(1 / 3)
    assert matrix["b"]["a"] == pytest.approx(1 / 3)
    assert matrix["missing"]["missing"] == 0.0


def test_build_outputs_pool_overlap_and_stats_with_missing_runs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs"
    write_run(
        run_dir / "bm25.jsonl",
        "bm25",
        [
            ("Q1", "DOC001_P0001", 1),
            ("Q1", "DOC001_P0002", 2),
        ],
    )
    write_run(
        run_dir / "dense.jsonl",
        "dense",
        [
            ("Q1", "DOC001_P0002", 1),
            ("Q2", "DOC001_P0001", 1),
        ],
    )
    pool = tmp_path / "pools" / "candidate_pool.jsonl"
    overlap = tmp_path / "analysis" / "pool_overlap.csv"
    stats_path = tmp_path / "analysis" / "pool_stats.json"

    stats = build_candidate_pool(
        run_dir=run_dir,
        pool_path=pool,
        overlap_path=overlap,
        stats_path=stats_path,
    )

    rows = read_jsonl(pool)
    assert len(rows) == 3
    assert stats["total_unique_candidate_pages_across_queries"] == 3
    assert stats["total_distinct_page_ids"] == 2
    assert stats["selection_count_distribution"]["1"] == 2
    assert stats["selection_count_distribution"]["2"] == 1
    assert stats["exclusive_pages_by_system"]["bm25"] == 1
    assert stats["exclusive_pages_by_system"]["dense"] == 1
    assert stats["systems_missing"] == ["hybrid", "llm_a", "llm_b", "llm_c", "copilot", "gpt_6_luna", "claude_haiku", "gemini_flash"]

    with overlap.open(encoding="utf-8", newline="") as source:
        matrix_rows = list(csv.reader(source))
    assert matrix_rows[0] == [
        "system", "bm25", "dense", "hybrid", "llm_a", "llm_b", "llm_c", "copilot", "gpt_6_luna", "claude_haiku", "gemini_flash"
    ]
    assert len(matrix_rows) == 11
    assert float(matrix_rows[1][2]) == pytest.approx(1 / 3, abs=1e-6)

    disk_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert disk_stats == stats


def test_available_run_with_wrong_system_fails_without_outputs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs"
    write_run(
        run_dir / "bm25.jsonl",
        "dense",
        [("Q1", "DOC001_P0001", 1)],
    )
    pool = tmp_path / "candidate_pool.jsonl"

    with pytest.raises(CandidatePoolError, match="does not match file"):
        build_candidate_pool(
            run_dir=run_dir,
            pool_path=pool,
            overlap_path=tmp_path / "overlap.csv",
            stats_path=tmp_path / "stats.json",
        )

    assert not pool.exists()
