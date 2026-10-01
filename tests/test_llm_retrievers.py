"""Tests for frontier LLM candidate generation."""
from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from scripts.run_conventional_retrievers import Query
from scripts.run_llm_retrievers import (
    CANDIDATE_RETRIEVAL_PROMPT,
    CorpusPage,
    LLMRetrievalError,
    MockLLMClient,
    build_user_prompt,
    configure_top_k,
    generate_query_candidates,
    pack_pages,
    parse_page_ids,
    request_candidates,
    recover_rankings_from_trace,
    run_system,
    write_run,
)


def pages(count: int) -> list[CorpusPage]:
    return [
        CorpusPage(
            page_id=f"DOC001_P{index:04d}",
            doc_id="DOC001",
            pdf_page=index,
            page_text=("relevant text " * 30) + str(index),
        )
        for index in range(1, count + 1)
    ]


def test_parser_handles_json_fences_objects_and_plain_text() -> None:
    allowed = {"DOC001_P0001", "DOC001_P0002"}
    assert parse_page_ids(
        '```json\n["DOC001_P0002", "DOC001_P0001"]\n```',
        allowed_ids=allowed,
        limit=20,
    ) == ["DOC001_P0002", "DOC001_P0001"]
    assert parse_page_ids(
        '{"results":[{"page_id":"DOC001_P0001"}]}',
        allowed_ids=allowed,
        limit=20,
    ) == ["DOC001_P0001"]
    assert parse_page_ids(
        "1. DOC001_P0002\n2. DOC999_P0001",
        allowed_ids=allowed,
        limit=20,
    ) == ["DOC001_P0002"]


def test_standard_prompt_replaces_query_and_corpus_context_exactly() -> None:
    source = pages(1)
    prompt = build_user_prompt(
        Query("Q1", "Which Missouri evidence is direct?"),
        source,
        max_page_chars=1_000,
    )

    expected = CANDIDATE_RETRIEVAL_PROMPT.replace(
        "{query}", "Which Missouri evidence is direct?"
    ).replace(
        "{corpus_context}",
        json.dumps(
            {
                "page_id": "DOC001_P0001",
                "doc_id": "DOC001",
                "pdf_page": 1,
                "page_text": source[0].page_text,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    assert prompt == expected
    assert "{query}" not in prompt
    assert "{corpus_context}" not in prompt
    assert "Question ID:" not in prompt
    assert "Return at most" not in prompt
    assert "You MUST output EXACTLY 20 candidate pages" in prompt
    assert "Under no circumstances should you output fewer than 20" in prompt


def test_pack_pages_preserves_order_and_all_pages() -> None:
    source = pages(12)
    batches = pack_pages(source, prompt_budget=2_000, max_page_chars=400)
    flattened = [page.page_id for batch in batches for page in batch]

    assert flattened == [page.page_id for page in source]
    assert len(batches) > 1


def test_hierarchical_mock_returns_valid_unique_top_20() -> None:
    source = pages(45)
    result = generate_query_candidates(
        client=MockLLMClient(),
        model="mock/model",
        system="llm_a",
        query=Query("Q1", "What is relevant?"),
        corpus=source,
        prompt_budget=4_000,
        max_page_chars=300,
        timeout=1,
        retries=1,
        trace_path=None,
    )

    assert len(result) == 20
    assert len(result) == len(set(result))
    assert set(result) <= {page.page_id for page in source}
    assert all(re.fullmatch(r"DOC001_P\d{4}", page_id) for page_id in result)


def test_writer_emits_exact_schema(tmp_path: Path) -> None:
    output = tmp_path / "llm_a.jsonl"
    ranking = [f"DOC001_P{index:04d}" for index in range(20, 0, -1)]
    count = write_run(
        output,
        [Query("Q1", "question")],
        {"Q1": ranking},
        system="llm_a",
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert count == 20
    assert [row["page_id"] for row in rows] == ranking
    assert [row["rank"] for row in rows] == list(range(1, 21))
    assert all(row["query_id"] == "Q1" and row["system"] == "llm_a" for row in rows)
class IncrementalRepairClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, **_kwargs) -> str:
        self.calls += 1
        selected = range(1, 16) if self.calls == 1 else range(16, 21)
        return json.dumps(
            [{"page_id": f"DOC001_P{index:04d}"} for index in selected]
        )


def test_short_response_is_repaired_with_additional_unique_pages() -> None:
    client = IncrementalRepairClient()
    result = request_candidates(
        client=client,
        model="mock/model",
        system="llm_a",
        query=Query("Q1", "What is relevant?"),
        pages=pages(25),
        target=20,
        max_page_chars=300,
        timeout=1,
        retries=2,
        trace_path=None,
        pass_number=1,
        batch_number=1,
        final_selection=True,
    )

    assert result == [f"DOC001_P{index:04d}" for index in range(1, 21)]
    assert client.calls == 2

class EmptyResponseClient:
    def complete(self, **_kwargs) -> str:
        return "[]"


def test_short_response_is_retried_then_rejected(caplog: pytest.LogCaptureFixture) -> None:
    kwargs = {
        "client": EmptyResponseClient(),
        "model": "mock/model",
        "system": "llm_a",
        "query": Query("Q1", "What is relevant?"),
        "pages": pages(2),
        "target": 1,
        "max_page_chars": 300,
        "timeout": 1,
        "retries": 1,
        "trace_path": None,
        "pass_number": 1,
        "batch_number": 1,
    }

    with pytest.raises(LLMRetrievalError, match="expected exactly 1"):
        request_candidates(**kwargs, allow_empty=True)
    assert "returned only 0 candidates" in caplog.text
class FailAfterOneClient:
    def __init__(self) -> None:
        self.calls = 0
        self.delegate = MockLLMClient()

    def complete(self, **kwargs) -> str:
        self.calls += 1
        if self.calls > 1:
            raise LLMRetrievalError("simulated interruption")
        return self.delegate.complete(**kwargs)


class CountingSuccessClient:
    def __init__(self) -> None:
        self.calls = 0
        self.delegate = MockLLMClient()

    def complete(self, **kwargs) -> str:
        self.calls += 1
        return self.delegate.complete(**kwargs)


def test_trace_recovery_requires_transition_or_completion_marker(tmp_path: Path) -> None:
    trace = tmp_path / "llm_a.jsonl"
    records = [
        {
            "system": "llm_a",
            "model": "mock/model",
            "query_id": "Q1",
            "response": "[]",
            "parsed_page_ids": ["DOC001_P0001"],
            "error": None,
        },
        {
            "system": "llm_a",
            "model": "mock/model",
            "query_id": "Q2",
            "response": "[]",
            "parsed_page_ids": ["DOC001_P0002"],
            "error": None,
        },
        {
            "event": "query_complete",
            "system": "llm_a",
            "model": "mock/model",
            "query_id": "Q2",
            "ranked_page_ids": ["DOC001_P0002"],
            "error": None,
        },
        {
            "system": "llm_a",
            "model": "mock/model",
            "query_id": "Q3",
            "response": "[]",
            "parsed_page_ids": ["DOC001_P0003"],
            "error": None,
        },
    ]
    trace.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    recovered = recover_rankings_from_trace(
        trace,
        model="mock/model",
        system="llm_a",
        query_ids={"Q1", "Q2", "Q3"},
        allowed_ids={"DOC001_P0001", "DOC001_P0002", "DOC001_P0003"},
    )

    # Legacy traces with fewer than the required candidate count are stale.
    assert recovered == {}


def test_resume_checkpoint_skips_completed_paid_query(tmp_path: Path) -> None:
    source = pages(20)
    queries = [Query("Q1", "first"), Query("Q2", "second")]
    output_dir = tmp_path / "runs"
    trace_dir = tmp_path / "traces"
    interrupted = FailAfterOneClient()

    with pytest.raises(LLMRetrievalError, match="simulated interruption"):
        run_system(
            client=interrupted,
            model="mock/model",
            system="llm_a",
            queries=queries,
            corpus=source,
            output_dir=output_dir,
            trace_dir=trace_dir,
            prompt_budget=4_000,
            max_page_chars=300,
            timeout=1,
            retries=1,
        )

    resumed = CountingSuccessClient()
    count = run_system(
        client=resumed,
        model="mock/model",
        system="llm_a",
        queries=queries,
        corpus=source,
        output_dir=output_dir,
        trace_dir=trace_dir,
        prompt_budget=4_000,
        max_page_chars=300,
        timeout=1,
        retries=1,
        resume=True,
    )

    assert interrupted.calls == 2
    assert resumed.calls == 1
    assert count == 40
    rows = [
        json.loads(line)
        for line in (output_dir / "llm_a.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["query_id"] for row in rows] == (["Q1"] * 20 + ["Q2"] * 20)





def test_configurable_top_k_regenerates_prompt_and_output(tmp_path: Path) -> None:
    import scripts.run_llm_retrievers as module
    try:
        configure_top_k(100)
        assert module.TOP_K == 100
        assert "EXACTLY 100 candidate pages" in module.CANDIDATE_RETRIEVAL_PROMPT
        ranking = generate_query_candidates(
            client=MockLLMClient(), model="mock/model", system="llm_a",
            query=Query("Q1", "question"), corpus=pages(120),
            prompt_budget=400_000, max_page_chars=300,
            timeout=1, retries=1, trace_path=None,
        )
        assert len(ranking) == 100
        output = tmp_path / "llm_a.jsonl"
        assert write_run(output, [Query("Q1", "question")], {"Q1": ranking}, system="llm_a") == 100
    finally:
        configure_top_k(20)
