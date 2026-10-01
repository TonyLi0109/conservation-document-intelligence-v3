"""Tests for provisional-core selection and consolidation validation."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.consolidate_core_groups import ConsolidationError, normalize, provisional


def test_provisional_uses_exact_consensus_core_rule(tmp_path: Path) -> None:
    path = tmp_path / "qrels.csv"
    rows = [
        {
            "query_id": "Q1", "page_id": "DOC001_P0001",
            "r1": "4", "r2": "4", "r3": "3", "mean_grade": "3.666667",
        },
        {
            "query_id": "Q1", "page_id": "DOC001_P0002",
            "r1": "4", "r2": "3", "r3": "4", "mean_grade": "3.666667",
        },
        {
            "query_id": "Q1", "page_id": "DOC001_P0003",
            "r1": "4", "r2": "3", "r3": "3", "mean_grade": "3.333333",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    assert provisional(path) == {
        "Q1": ["DOC001_P0001", "DOC001_P0002"]
    }


def test_normalize_assigns_stable_ids_and_preserves_unassigned_pages() -> None:
    groups, unassigned = normalize(
        "Q4",
        [{
            "group_id": "model-id-is-ignored",
            "facet_description": "2015 framework",
            "member_page_ids": ["DOC001_P0003"],
            "why_this_facet_is_core": "Required comparison side.",
            "redundancy_confidence": "high",
        }],
        {"DOC001_P0003", "DOC036_P0010"},
    )

    assert groups[0]["group_id"] == "Q4_G01"
    assert unassigned == ["DOC036_P0010"]
    assert groups[0]["manual_review_required"] is False


def test_normalize_rejects_cross_group_duplicate_membership() -> None:
    raw = [
        {
            "facet_description": "A",
            "member_page_ids": ["DOC001_P0001"],
            "why_this_facet_is_core": "A is required.",
            "redundancy_confidence": "high",
        },
        {
            "facet_description": "B",
            "member_page_ids": ["DOC001_P0001"],
            "why_this_facet_is_core": "B is required.",
            "redundancy_confidence": "medium",
        },
    ]
    with pytest.raises(ConsolidationError, match="Duplicate assignment"):
        normalize("Q1", raw, {"DOC001_P0001"})



def test_consolidate_retry_includes_validation_error(monkeypatch) -> None:
    import scripts.consolidate_core_groups as module

    responses = [
        '{"groups": ['
        '{"facet_description":"A","member_page_ids":["DOC001_P0001"],'
        '"why_this_facet_is_core":"A","redundancy_confidence":"high"},'
        '{"facet_description":"B","member_page_ids":["DOC001_P0001"],'
        '"why_this_facet_is_core":"B","redundancy_confidence":"high"}'
        ']}',
        '{"groups": ['
        '{"facet_description":"A","member_page_ids":["DOC001_P0001"],'
        '"why_this_facet_is_core":"A","redundancy_confidence":"high"}'
        ']}',
    ]
    prompts: list[str] = []

    def fake_call(model: str, prompt: str, timeout: float) -> str:
        prompts.append(prompt)
        return responses.pop(0)

    monkeypatch.setattr(module, "call", fake_call)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    result = module.consolidate(
        "Q1",
        "Question",
        ["DOC001_P0001"],
        {"DOC001": "Title"},
        {("Q1", "DOC001_P0001"): {
            "core_facet_labels": ["A"],
            "rationales": ["Reason"],
        }},
        "openai/gpt-5.6-sol",
        30,
        2,
    )

    assert result["group_count"] == 1
    assert len(prompts) == 2
    assert "Duplicate assignment DOC001_P0001" in prompts[1]
    assert "AT MOST ONE group" in prompts[1]
