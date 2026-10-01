"""Tests for blind, independent page-level LLM judging."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

from scripts.run_conventional_retrievers import Query
from scripts.run_llm_judges import (
    Candidate,
    JudgeError,
    LiteLLMAsyncClient,
    MockJudgeClient,
    PageEvidence,
    build_blind_prompt,
    judge_candidate,
    load_candidates,
    load_document_titles,
    parse_judgment,
    provider_retry_delay,
    run_judge,
)


def evidence() -> PageEvidence:
    return PageEvidence(
        page_id="DOC036_P0106",
        doc_id="DOC036",
        pdf_page=106,
        document_title="2022 Missouri Comprehensive Conservation Strategy",
        page_text="The page contains direct Missouri conservation evidence.",
    )


def test_candidate_loader_discards_origin_metadata_and_deduplicates_strictly(
    tmp_path: Path,
) -> None:
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps(
            {
                "query_id": "Q1",
                "page_id": "DOC036_P0106",
                "selection_count": 4,
                "provenance": {"bm25": 1, "llm_a": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    candidates = load_candidates(pool)

    assert candidates == [Candidate("Q1", "DOC036_P0106")]
    assert not hasattr(candidates[0], "provenance")
    assert not hasattr(candidates[0], "rank")
    assert not hasattr(candidates[0], "selection_count")

    with pool.open("a", encoding="utf-8") as output:
        output.write(
            '{"query_id":"Q1","page_id":"DOC036_P0106","selection_count":1}\n'
        )
    with pytest.raises(JudgeError, match="duplicate candidate"):
        load_candidates(pool)


def test_catalog_loader_accepts_repository_document_id_header(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "source_catalog.csv"
    catalog.write_text(
        "document_id,title,year\n"
        "DOC001,Missouri State Wildlife Action Plan,2015\n",
        encoding="utf-8",
    )

    assert load_document_titles(catalog) == {
        "DOC001": "Missouri State Wildlife Action Plan"
    }


def test_blind_prompt_contains_only_allowed_inputs() -> None:
    prompt = build_blind_prompt(
        Query("Q1", "Which evidence is current?"),
        evidence(),
    )

    assert "Question ID: Q1" in prompt
    assert "Which evidence is current?" in prompt
    assert "DOC036_P0106" in prompt
    assert "2022 Missouri Comprehensive Conservation Strategy" in prompt
    assert "direct Missouri conservation evidence" in prompt
    assert "provenance" not in prompt.casefold()
    assert "selection_count" not in prompt
    assert "retrieval rank" not in prompt.casefold()
    assert "bm25" not in prompt.casefold()


def test_parse_judgment_accepts_fenced_fallback_and_rejects_bad_grade() -> None:
    candidate = Candidate("Q1", "DOC036_P0106")
    parsed = parse_judgment(
        '{"question_id":"Q1","page_id":"DOC036_P0106",'
        '"grade":4,"core_facet":"current planning horizon",'
        '"redundancy_group_hint":null,'
        '"rationale":"Direct decisive evidence."}',
        candidate,
    )
    assert parsed.grade == 4
    assert parsed.core_facet == "current planning horizon"

    with pytest.raises(JudgeError, match="integer from 0 through 4"):
        parse_judgment(
            '{"question_id":"Q1","page_id":"DOC036_P0106",'
            '"grade":5,"core_facet":null,"redundancy_group_hint":null,'
            '"rationale":"Invalid."}',
            candidate,
        )

class FlakyClient:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self.max_tokens: list[int] = []

    async def complete(self, **kwargs: object) -> str:
        self.calls += 1
        self.prompts.append(str(kwargs["user_prompt"]))
        self.max_tokens.append(int(kwargs["max_tokens"]))
        if self.calls == 1:
            return "not json"
        prompt = str(kwargs["user_prompt"])
        query_id = re.search(r"Question ID: (.+)", prompt).group(1)
        page_id = re.search(r"Page ID: (.+)", prompt).group(1)
        return json.dumps(
            {
                "question_id": query_id,
                "page_id": page_id,
                "grade": 3,
                "core_facet": None,
                "redundancy_group_hint": None,
                "rationale": "Strong explicit support.",
            }
        )


def test_async_judge_retries_malformed_response(monkeypatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    client = FlakyClient()
    result = asyncio.run(
        judge_candidate(
            client=client,
            semaphore=asyncio.Semaphore(1),
            model="test/model",
            judge_name="judge_a",
            candidate=Candidate("Q1", "DOC036_P0106"),
            query=Query("Q1", "Question"),
            page=evidence(),
            timeout=1,
            retries=2,
        )
    )

    assert client.calls == 2
    assert "FORMAT CORRECTION" in client.prompts[1]
    assert client.max_tokens == [2048, 2048]
    assert result.grade == 3


def test_mock_run_writes_exact_judgment_schema(tmp_path: Path) -> None:
    output = tmp_path / "judge_a.jsonl"
    count = asyncio.run(
        run_judge(
            client=MockJudgeClient(),
            judge_name="judge_a",
            model="mock/model",
            candidates=[
                Candidate("Q1", "DOC036_P0106"),
                Candidate("Q2", "DOC001_P0001"),
            ],
            queries={
                "Q1": Query("Q1", "First question"),
                "Q2": Query("Q2", "Second question"),
            },
            pages={
                "DOC036_P0106": evidence(),
                "DOC001_P0001": PageEvidence(
                    "DOC001_P0001", "DOC001", 1, "Plan", "Page text"
                ),
            },
            output_path=output,
            concurrency=2,
            timeout=1,
            retries=1,
        )
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert count == 2
    assert [row["question_id"] for row in rows] == ["Q1", "Q2"]
    assert all(set(row) == {"question_id", "page_id", "grade", "core_facet", "redundancy_group_hint", "rationale"} for row in rows)
    assert all(row["grade"] == 2 for row in rows)

class SelectiveFailureClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, **kwargs: object) -> str:
        prompt = str(kwargs["user_prompt"])
        query_id = re.search(r"Question ID: (.+)", prompt).group(1)
        page_id = re.search(r"Page ID: (.+)", prompt).group(1)
        self.calls.append(page_id)
        if page_id == "DOC001_P0001":
            return "not json"
        return json.dumps(
            {
                "question_id": query_id,
                "page_id": page_id,
                "grade": 3,
                "core_facet": None,
                "redundancy_group_hint": None,
                "rationale": "Strong explicit support.",
            }
        )


class CountingJudgeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, **kwargs: object) -> str:
        prompt = str(kwargs["user_prompt"])
        query_id = re.search(r"Question ID: (.+)", prompt).group(1)
        page_id = re.search(r"Page ID: (.+)", prompt).group(1)
        self.calls.append(page_id)
        return json.dumps(
            {
                "question_id": query_id,
                "page_id": page_id,
                "grade": 2,
                "core_facet": None,
                "redundancy_group_hint": None,
                "rationale": "Supporting evidence.",
            }
        )


def test_parser_rejects_extra_keys_and_enforces_null_facets() -> None:
    candidate = Candidate("Q1", "DOC036_P0106")
    with pytest.raises(JudgeError, match="unexpected fields"):
        parse_judgment(
            json.dumps(
                {
                    "question_id": "Q1",
                    "page_id": "DOC036_P0106",
                    "grade": 1,
                    "core_facet": None,
                    "redundancy_group_hint": None,
                    "rationale": "Marginal support.",
                    "1": "extraneous model annotation",
                }
            ),
            candidate,
        )

    with pytest.raises(JudgeError, match="must be null"):
        parse_judgment(
            json.dumps(
                {
                    "question_id": "Q1",
                    "page_id": "DOC036_P0106",
                    "grade": 3,
                    "core_facet": "invalid facet",
                    "redundancy_group_hint": None,
                    "rationale": "Strong support.",
                }
            ),
            candidate,
        )

def test_judge_resume_retries_only_unfinished_candidate(tmp_path: Path) -> None:
    candidates = [
        Candidate("Q1", "DOC036_P0106"),
        Candidate("Q2", "DOC001_P0001"),
    ]
    queries = {
        "Q1": Query("Q1", "First question"),
        "Q2": Query("Q2", "Second question"),
    }
    pages = {
        "DOC036_P0106": evidence(),
        "DOC001_P0001": PageEvidence(
            "DOC001_P0001", "DOC001", 1, "Plan", "Page text"
        ),
    }
    output = tmp_path / "judge_b.jsonl"
    interrupted = SelectiveFailureClient()

    with pytest.raises(JudgeError, match="1/2 are checkpointed"):
        asyncio.run(
            run_judge(
                client=interrupted,
                judge_name="judge_b",
                model="mock/model",
                candidates=candidates,
                queries=queries,
                pages=pages,
                output_path=output,
                concurrency=1,
                timeout=1,
                retries=1,
            )
        )

    resumed = CountingJudgeClient()
    count = asyncio.run(
        run_judge(
            client=resumed,
            judge_name="judge_b",
            model="mock/model",
            candidates=candidates,
            queries=queries,
            pages=pages,
            output_path=output,
            concurrency=1,
            timeout=1,
            retries=1,
            resume=True,
        )
    )

    assert count == 2
    assert resumed.calls == ["DOC001_P0001"]
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["page_id"] for row in rows] == ["DOC036_P0106", "DOC001_P0001"]

def test_parse_judgment_accepts_single_object_array() -> None:
    candidate = Candidate("Q1", "DOC036_P0106")
    judgment = parse_judgment(
        '[{"question_id":"Q1","page_id":"DOC036_P0106",'
        '"grade":3,"core_facet":null,"redundancy_group_hint":null,'
        '"rationale":"Strong support."}]',
        candidate,
    )

    assert judgment.grade == 3
    assert judgment.rationale == "Strong support."

def test_gemini_client_uses_lowest_reasoning_and_json_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"query_id":"Q1","page_id":"DOC036_P0106",'
                        '"grade":2,"rationale":"Support."}'
                    )
                )
            ]
        )

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(acompletion=fake_acompletion),
    )
    result = asyncio.run(
        LiteLLMAsyncClient().complete(
            model="gemini/gemini-3.1-pro-preview",
            system_prompt="system",
            user_prompt="user",
            timeout=1,
            max_tokens=2048,
        )
    )

    assert result.startswith('{"query_id"')
    assert captured["reasoning_effort"] == "none"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["max_tokens"] == 2048

def test_provider_retry_delay_honors_short_waits_and_rejects_daily_waits() -> None:
    short = Exception('"retryDelay": "4s"')
    daily = Exception("Please retry in 23h28m8.3s")

    assert provider_retry_delay(short) == pytest.approx(5.0)
    assert provider_retry_delay(daily) is None
