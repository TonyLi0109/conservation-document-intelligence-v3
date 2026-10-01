"""Focused tests for Professor Shang Section 7 diagnostics."""
from __future__ import annotations

from scripts.analyze_core_group_failures import (
    adjacent_hits, available_depth_diagnostics, classify_failure,
    complementarity_rows, rank_of_group, token_overlap,
)
from scripts.analyze_judge_stability import jaccard, quadratic_weighted_kappa, same_group_pairs


def test_group_rank_and_adjacent_page_detection() -> None:
    ranking = ["DOC001_P0008", "DOC001_P0011", "DOC002_P0010"]
    assert rank_of_group(ranking, ["DOC001_P0011", "DOC001_P0012"]) == 2
    assert adjacent_hits("DOC001_P0010", ranking) == [
        ("DOC001_P0008", 1, -2), ("DOC001_P0011", 2, 1)
    ]


def test_failure_taxonomy_prefers_observed_deep_ranking_failure() -> None:
    group = {
        "query_id": "Q3", "members": ["DOC001_P0050"],
        "confidence": "high",
    }
    reason, basis = classify_failure(
        group=group, query="wetland status", ranking20=["DOC001_P0001"],
        ranking100=["DOC001_P0001"] * 20 + ["DOC001_P0050"],
        corpus={"DOC001_P0050": "wetland status evidence"},
        covered_groups_for_query=1,
    )
    assert reason == "Ranking failure"
    assert "rank 21" in basis


def test_overlap_and_group_pair_stability_primitives() -> None:
    assert jaccard({1, 2}, {2, 3}) == 1 / 3
    assert quadratic_weighted_kappa([0, 1, 4], [0, 1, 4]) == 1.0
    assert same_group_pairs([
        {"member_page_ids": ["P1", "P2", "P3"]},
        {"member_page_ids": ["P4"]},
    ]) == {("P1", "P2"), ("P1", "P3"), ("P2", "P3")}
    assert token_overlap("Missouri wetland planning", ["wetland plan in Missouri"]) > 0.5



def test_extended_complementarity_includes_sparse_copilot() -> None:
    groups = [
        {"query_id": "Q1", "group_id": "Q1_G01", "members": ["DOC001_P0001"]},
        {"query_id": "Q1", "group_id": "Q1_G02", "members": ["DOC002_P0001"]},
    ]
    runs = {
        "bm25": {"Q1": ["DOC001_P0001"]},
        "copilot": {"Q1": ["DOC002_P0001"]},
    }
    rows = complementarity_rows(("bm25", "copilot"), groups, runs)
    assert rows[-1]["system_added"] == "copilot"
    assert rows[-1]["new_core_groups_contributed"] == 1
    assert rows[-1]["covered_core_groups"] == 2


def test_sparse_external_run_reports_actual_available_depth() -> None:
    groups = {
        "Q1": [
            {"query_id": "Q1", "group_id": "Q1_G01", "members": ["DOC001_P0001"]},
            {"query_id": "Q1", "group_id": "Q1_G02", "members": ["DOC002_P0001"]},
        ]
    }
    runs = {"copilot": {"Q1": ["DOC001_P0001"]}}
    rows = available_depth_diagnostics("copilot", groups, runs)
    assert rows[0]["candidate_count"] == 1
    assert rows[0]["requested_top20_complete"] is False
    assert rows[0]["covered_core_groups_at_available_depth"] == 1
    assert rows[-1]["query_id"] == "ALL"
    assert "no Copilot Top-100 run" in rows[-1]["limitation"]
