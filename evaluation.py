"""Reproducible retrieval and grounded-generation evaluation for V3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import SETTINGS, V3_ROOT
from database import KnowledgeStore
from main import ask_chatbot_with_context, search_corpus
from validator import INSUFFICIENT_MESSAGE, VALIDATION_FAILED_MESSAGE


DEFAULT_CASES_PATH = V3_ROOT / "evaluation" / "cases.json"
DEFAULT_RESULTS_PATH = V3_ROOT / "data" / "evaluation_results.json"


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_document_ids: list[str]
    acceptable_statuses: list[str]


def load_evaluation_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[EvaluationCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("evaluation cases must be a non-empty JSON array")
    cases = [EvaluationCase(**item) for item in payload]
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("evaluation case IDs must be unique")
    return cases


def _classify_answer(markdown: str) -> str:
    if markdown == VALIDATION_FAILED_MESSAGE:
        return "validation_failed"
    if markdown.startswith(INSUFFICIENT_MESSAGE):
        return "insufficient_evidence"
    if "**Unsupported facets**" in markdown:
        return "partially_answered"
    return "answered"


def run_evaluation(
    store: KnowledgeStore,
    *,
    cases_path: str | Path = DEFAULT_CASES_PATH,
    top_k: int = SETTINGS.retrieval.top_k,
    include_generation: bool = True,
    model: str | None = None,
) -> dict[str, object]:
    """Run the fixed V3 suite and persist a transparent, inspectable report."""

    rows: list[dict[str, object]] = []
    for case in load_evaluation_cases(cases_path):
        retrieved = search_corpus(case.question, store, top_k=top_k)
        retrieved_ids = list(dict.fromkeys(item.document_id for item in retrieved))
        expected = set(case.expected_document_ids)
        retrieval_hit = not expected or bool(expected.intersection(retrieved_ids))
        status = "not_run"
        cited_ids: list[str] = []
        error = ""
        if include_generation:
            try:
                answer, _, sources = ask_chatbot_with_context(
                    case.question, store, top_k=top_k, model=model
                )
                status = _classify_answer(answer)
                cited_ids = list(dict.fromkeys(item.document_id for item in sources))
            except Exception as exception:
                status = "system_error"
                error = f"{type(exception).__name__}: {exception}"
        rows.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "expected_document_ids": case.expected_document_ids,
                "retrieved_document_ids": retrieved_ids,
                "retrieval_hit": retrieval_hit,
                "status": status,
                "acceptable_statuses": case.acceptable_statuses,
                "status_pass": status in case.acceptable_statuses if include_generation else None,
                "cited_document_ids": cited_ids,
                "error": error,
            }
        )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model or SETTINGS.models.llm_model,
        "top_k": top_k,
        "case_count": len(rows),
        "retrieval_recall_at_k": sum(bool(row["retrieval_hit"]) for row in rows) / len(rows),
        "status_accuracy": (
            sum(bool(row["status_pass"]) for row in rows) / len(rows)
            if include_generation else None
        ),
        "cases": rows,
    }
    DEFAULT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_RESULTS_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
