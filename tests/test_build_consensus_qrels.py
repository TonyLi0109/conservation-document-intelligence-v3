"""Tests for Milestone E consensus qrels aggregation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_consensus_qrels import (
    FULL_COLUMNS,
    PoolCandidate,
    QrelsError,
    aggregate_grades,
    build_consensus_qrels,
    render_trec_qrels,
)


def write_pool(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for query_id, page_id, selection_count in rows:
            output.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "page_id": page_id,
                        "selection_count": selection_count,
                        "provenance": {"bm25": 1},
                    }
                )
                + "\n"
            )


def write_judge(
    directory: Path,
    judge: str,
    rows: list[tuple[str, str, int]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{judge}.jsonl").open("w", encoding="utf-8") as output:
        for query_id, page_id, grade in rows:
            output.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "page_id": page_id,
                        "grade": grade,
                        "rationale": "Evidence-based rationale.",
                    }
                )
                + "\n"
            )


def test_aggregation_math_core_relevant_and_rounding() -> None:
    candidate = PoolCandidate("Q1", "DOC036_P0106", 3)
    record = aggregate_grades(candidate, (4, 4, 3))

    assert record.mean_grade == pytest.approx(11 / 3)
    assert record.grade_range == 1
    assert record.variance == pytest.approx(2 / 9)
    assert record.final_grade == 4
    assert record.relevant is True
    assert record.core is True
    assert record.selection_count == 3


@pytest.mark.parametrize(
    ("grades", "relevant", "core", "final_grade"),
    [
        ((4, 2, 2), True, False, 3),
        ((3, 2, 2), False, False, 2),
        ((4, 4, 2), True, False, 3),
        ((4, 4, 4), True, True, 4),
        ((1, 1, 4), False, False, 2),
    ],
)
def test_consensus_threshold_logic(
    grades: tuple[int, int, int],
    relevant: bool,
    core: bool,
    final_grade: int,
) -> None:
    record = aggregate_grades(
        PoolCandidate("Q1", "DOC001_P0001", 1),
        grades,
    )
    assert record.relevant is relevant
    assert record.core is core
    assert record.final_grade == final_grade


def test_outputs_have_exact_csv_and_trec_schemas(tmp_path: Path) -> None:
    pool = tmp_path / "candidate_pool.jsonl"
    judgment_dir = tmp_path / "judgments"
    rows = [
        ("Q1", "DOC001_P0001", 2),
        ("Q1", "DOC001_P0002", 1),
    ]
    write_pool(pool, rows)
    write_judge(
        judgment_dir,
        "judge_a",
        [("Q1", "DOC001_P0001", 4), ("Q1", "DOC001_P0002", 0)],
    )
    write_judge(
        judgment_dir,
        "judge_b",
        [("Q1", "DOC001_P0001", 3), ("Q1", "DOC001_P0002", 1)],
    )
    write_judge(
        judgment_dir,
        "judge_c",
        [("Q1", "DOC001_P0001", 2), ("Q1", "DOC001_P0002", 0)],
    )
    full = tmp_path / "qrels" / "full.csv"
    trec = tmp_path / "qrels" / "qrels.tsv"

    stats = build_consensus_qrels(
        pool_path=pool,
        judgment_dir=judgment_dir,
        full_output=full,
        trec_output=trec,
    )

    with full.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        output_rows = list(reader)
        assert tuple(reader.fieldnames or ()) == FULL_COLUMNS
    assert len(output_rows) == 2
    assert output_rows[0]["mean_grade"] == "3.000000"
    assert output_rows[0]["final_grade"] == "3"
    assert output_rows[0]["relevant"] == "true"
    assert output_rows[0]["core"] == "false"
    assert output_rows[0]["variance"] == "0.666667"
    assert trec.read_text(encoding="utf-8").splitlines() == [
        "Q1\t0\tDOC001_P0001\t3",
        "Q1\t0\tDOC001_P0002\t0",
    ]
    assert stats["consensus_count"] == 2
    assert stats["relevant_count"] == 1


def test_relevant_only_trec_filter() -> None:
    relevant = aggregate_grades(
        PoolCandidate("Q1", "DOC001_P0001", 1),
        (4, 3, 2),
    )
    irrelevant = aggregate_grades(
        PoolCandidate("Q1", "DOC001_P0002", 1),
        (1, 1, 0),
    )

    assert render_trec_qrels(
        [relevant, irrelevant],
        relevant_only=True,
    ) == "Q1\t0\tDOC001_P0001\t3\n"


def test_missing_and_incomplete_judges_warn_and_skip(tmp_path: Path, caplog) -> None:
    pool = tmp_path / "candidate_pool.jsonl"
    judgment_dir = tmp_path / "judgments"
    write_pool(
        pool,
        [
            ("Q1", "DOC001_P0001", 1),
            ("Q1", "DOC001_P0002", 1),
        ],
    )
    write_judge(
        judgment_dir,
        "judge_a",
        [("Q1", "DOC001_P0001", 4), ("Q1", "DOC001_P0002", 2)],
    )
    write_judge(
        judgment_dir,
        "judge_b",
        [("Q1", "DOC001_P0001", 4)],
    )
    full = tmp_path / "full.csv"
    trec = tmp_path / "qrels.tsv"

    stats = build_consensus_qrels(
        pool_path=pool,
        judgment_dir=judgment_dir,
        full_output=full,
        trec_output=trec,
    )

    assert stats["judges_missing"] == ["judge_c"]
    assert stats["consensus_count"] == 0
    assert stats["incomplete_count"] == 2
    assert full.read_text(encoding="utf-8").splitlines() == [
        ",".join(FULL_COLUMNS)
    ]
    assert trec.read_text(encoding="utf-8") == ""
    assert "Judgment file missing" in caplog.text
    assert "without a complete three-judge grade tuple" in caplog.text


def test_out_of_pool_judgment_is_rejected_before_outputs(tmp_path: Path) -> None:
    pool = tmp_path / "candidate_pool.jsonl"
    judgments = tmp_path / "judgments"
    write_pool(pool, [("Q1", "DOC001_P0001", 1)])
    write_judge(judgments, "judge_a", [("Q1", "DOC999_P0001", 3)])
    full = tmp_path / "full.csv"

    with pytest.raises(QrelsError, match="not in candidate pool"):
        build_consensus_qrels(
            pool_path=pool,
            judgment_dir=judgments,
            full_output=full,
            trec_output=tmp_path / "qrels.tsv",
        )

    assert not full.exists()
