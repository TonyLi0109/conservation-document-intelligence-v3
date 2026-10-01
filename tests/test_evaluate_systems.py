"""Tests for Milestone H retrieval-system evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.evaluate_systems import (
    OUTPUT_COLUMNS,
    Qrel,
    SYSTEMS,
    dcg,
    evaluate_all,
    evaluate_query,
    load_core_groups,
    load_qrels,
    load_run,
    render_summary,
    write_csv_atomic,
)


def test_metrics_use_mean_thresholds_and_graded_ndcg() -> None:
    qrels = {
        "P1": Qrel(4.0, 4),
        "P2": Qrel(3.0, 3),
        "P3": Qrel(2.0, 2),
    }
    result = evaluate_query(
        "bm25", "Q1", ["P2", "UNKNOWN", "P1"], qrels,
        [frozenset({"P1", "P4"}), frozenset({"P2"})],
    )

    assert result.recall_at_20_relevant == 1.0
    assert result.recall_at_20_core == 1.0
    assert result.core_group_recall_at_20 == 1.0
    assert result.core_groups_retrieved_at_20 == 2
    assert result.judged_retrieved_at_20 == 2
    assert result.judged_coverage_at_20 == pytest.approx(2 / 3)
    assert result.first_relevant_rank == 1
    assert result.reciprocal_rank == 1.0
    assert result.ndcg_at_20 == pytest.approx(dcg([3, 0, 4]) / dcg([4, 3, 2]))


def test_empty_reference_tier_is_zero_for_nine_query_macro_policy() -> None:
    result = evaluate_query("dense", "Q4", ["P1"], {"P1": Qrel(2.0, 2)})
    assert result.recall_at_20_relevant == 0.0
    assert result.recall_at_20_core == 0.0
    assert result.reciprocal_rank == 0.0


def test_macro_average_includes_every_qrels_query() -> None:
    qrels = {
        "Q1": {"P1": Qrel(3.0, 3)},
        "Q2": {"P2": Qrel(1.0, 1)},
    }
    runs = {system: {"Q1": ["P1"]} for system in SYSTEMS}
    details, summaries = evaluate_all(qrels, runs)

    assert len(details) == 2 * len(SYSTEMS)
    assert summaries["bm25"]["recall_relevant"] == 0.5
    assert summaries["bm25"]["mrr"] == 0.5


def test_loaders_and_atomic_output_schema(tmp_path: Path) -> None:
    qrels_path = tmp_path / "qrels.csv"
    qrels_path.write_text(
        "query_id,page_id,mean_grade,final_grade\nQ1,P1,3.0,3\n",
        encoding="utf-8",
    )
    run_path = tmp_path / "bm25.jsonl"
    records = [
        {"query_id": "Q1", "page_id": "P2", "rank": 2, "system": "bm25"},
        {"query_id": "Q1", "page_id": "P1", "rank": 1, "system": "bm25"},
    ]
    run_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    qrels = load_qrels(qrels_path)
    run = load_run(run_path, "bm25")
    row = evaluate_query("bm25", "Q1", run["Q1"], qrels["Q1"])
    output = tmp_path / "analysis" / "evaluation.csv"
    write_csv_atomic(output, [row])

    with output.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        assert tuple(reader.fieldnames or ()) == OUTPUT_COLUMNS
        written = list(reader)
    assert run["Q1"] == ["P1", "P2"]
    assert written[0]["first_relevant_rank"] == "1"


def test_summary_is_markdown_table() -> None:
    summary = {
        system: {
            "judged_coverage": 0.5,
            "recall_relevant": 0.1,
            "recall_core": 0.2,
            "recall_core_groups": 0.25,
            "ndcg": 0.3,
            "mrr": 0.4,
        }
        for system in SYSTEMS
    }
    rendered = render_summary(summary)
    assert "| System | Judged coverage | Relevant Recall@20 |" in rendered
    assert "| gemini_flash | 0.5000 | 0.1000 | 0.2000 | 0.2500 | 0.3000 | 0.4000 |" in rendered


def test_rich_qrels_use_exact_consensus_flags(tmp_path: Path) -> None:
    path = tmp_path / "qrels.csv"
    path.write_text(
        "query_id,page_id,mean_grade,final_grade,relevant,core\n"
        "Q1,P1,3.6,4,True,False\n",
        encoding="utf-8",
    )
    qrel = load_qrels(path)["Q1"]["P1"]
    assert qrel.relevant is True
    assert qrel.core is False


def test_load_core_groups_and_group_alternative_coverage(tmp_path: Path) -> None:
    path = tmp_path / "groups.json"
    path.write_text(
        json.dumps({
            "questions": [{
                "question_id": "Q1",
                "groups": [
                    {"group_id": "Q1_G01", "member_page_ids": ["P1", "P2"]},
                    {"group_id": "Q1_G02", "member_page_ids": ["P3"]},
                ],
            }]
        }),
        encoding="utf-8",
    )
    groups = load_core_groups(path)
    assert groups["Q1"] == [frozenset({"P1", "P2"}), frozenset({"P3"})]

    result = evaluate_query(
        "bm25", "Q1", ["P2"], {"P2": Qrel(4.0, 4)}, groups["Q1"]
    )
    assert result.core_groups_retrieved_at_20 == 1
    assert result.core_group_recall_at_20 == 0.5
