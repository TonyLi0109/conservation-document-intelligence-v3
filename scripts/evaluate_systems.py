"""Evaluate page-level retrieval runs against the consensus qrels.

The evaluator computes each metric independently for every query and then
macro-averages over the complete qrels query set. Pages absent from the qrels
are treated as unjudged (grade zero). A query with no relevant/core reference
pages receives recall zero so every system is averaged over the same queries.

Example:
    python scripts/evaluate_systems.py
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QRELS = REPOSITORY_ROOT / "benchmark" / "qrels" / "consensus_qrels_full.csv"
DEFAULT_CORE_GROUPS = REPOSITORY_ROOT / "benchmark" / "qrels" / "core_groups.json"
DEFAULT_RUN_DIR = REPOSITORY_ROOT / "benchmark" / "candidate_runs"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "benchmark" / "analysis" / "system_evaluation.csv"
SYSTEMS = ("bm25", "dense", "hybrid", "llm_a", "llm_b", "llm_c", "copilot", "gpt_6_luna", "claude_haiku", "gemini_flash")
OUTPUT_COLUMNS = (
    "system",
    "query_id",
    "retrieved_at_20",
    "judged_retrieved_at_20",
    "judged_coverage_at_20",
    "relevant_reference_count",
    "relevant_retrieved_at_20",
    "recall_at_20_relevant",
    "core_reference_count",
    "core_retrieved_at_20",
    "recall_at_20_core",
    "core_group_reference_count",
    "core_groups_retrieved_at_20",
    "core_group_recall_at_20",
    "ndcg_at_20",
    "first_relevant_rank",
    "reciprocal_rank",
)


class EvaluationError(ValueError):
    """Raised when an input record is missing or malformed."""


@dataclass(frozen=True)
class Qrel:
    mean_grade: float
    final_grade: int
    relevant_flag: bool | None = None
    core_flag: bool | None = None

    @property
    def relevant(self) -> bool:
        return self.relevant_flag if self.relevant_flag is not None else self.mean_grade >= 2.5

    @property
    def core(self) -> bool:
        return self.core_flag if self.core_flag is not None else self.mean_grade >= 3.5


@dataclass(frozen=True)
class QueryMetrics:
    system: str
    query_id: str
    retrieved_at_20: int
    judged_retrieved_at_20: int
    judged_coverage_at_20: float
    relevant_reference_count: int
    relevant_retrieved_at_20: int
    recall_at_20_relevant: float
    core_reference_count: int
    core_retrieved_at_20: int
    recall_at_20_core: float
    core_group_reference_count: int
    core_groups_retrieved_at_20: int
    core_group_recall_at_20: float
    ndcg_at_20: float
    first_relevant_rank: int | None
    reciprocal_rank: float

    def csv_row(self) -> dict[str, object]:
        return {
            "system": self.system,
            "query_id": self.query_id,
            "retrieved_at_20": self.retrieved_at_20,
            "judged_retrieved_at_20": self.judged_retrieved_at_20,
            "judged_coverage_at_20": f"{self.judged_coverage_at_20:.6f}",
            "relevant_reference_count": self.relevant_reference_count,
            "relevant_retrieved_at_20": self.relevant_retrieved_at_20,
            "recall_at_20_relevant": f"{self.recall_at_20_relevant:.6f}",
            "core_reference_count": self.core_reference_count,
            "core_retrieved_at_20": self.core_retrieved_at_20,
            "recall_at_20_core": f"{self.recall_at_20_core:.6f}",
            "core_group_reference_count": self.core_group_reference_count,
            "core_groups_retrieved_at_20": self.core_groups_retrieved_at_20,
            "core_group_recall_at_20": f"{self.core_group_recall_at_20:.6f}",
            "ndcg_at_20": f"{self.ndcg_at_20:.6f}",
            "first_relevant_rank": self.first_relevant_rank or "",
            "reciprocal_rank": f"{self.reciprocal_rank:.6f}",
        }


def load_qrels(path: Path) -> dict[str, dict[str, Qrel]]:
    """Load rich CSV qrels, or a TREC TSV using grade-based tier fallback.

    The rich CSV is canonical because it retains mean grades. When only TREC
    qrels are supplied, final grades 3/4 represent relevant evidence and grade
    4 represents core evidence; synthetic means preserve those same cutoffs.
    """
    if not path.is_file():
        raise EvaluationError(f"Qrels file does not exist: {path}")
    qrels: dict[str, dict[str, Qrel]] = {}
    if path.suffix.casefold() == ".tsv":
        with path.open("r", encoding="utf-8", newline="") as source:
            for line_number, row in enumerate(csv.reader(source, delimiter="\t"), start=1):
                if not row:
                    continue
                if len(row) != 4:
                    raise EvaluationError(f"{path}:{line_number}: expected four TSV columns")
                query_id, _, page_id, raw_grade = row
                try:
                    final_grade = int(raw_grade)
                except ValueError as exc:
                    raise EvaluationError(f"{path}:{line_number}: invalid final_grade") from exc
                _store_qrel(qrels, query_id, page_id, Qrel(float(final_grade), final_grade), path, line_number)
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            required = {"query_id", "page_id", "mean_grade", "final_grade"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise EvaluationError(f"{path}: missing columns {sorted(missing)}")
            for line_number, row in enumerate(reader, start=2):
                try:
                    qrel = Qrel(
                        float(row["mean_grade"]),
                        int(row["final_grade"]),
                        _parse_optional_bool(row.get("relevant"), path, line_number, "relevant"),
                        _parse_optional_bool(row.get("core"), path, line_number, "core"),
                    )
                except (TypeError, ValueError) as exc:
                    raise EvaluationError(f"{path}:{line_number}: invalid grade") from exc
                if not 0 <= qrel.final_grade <= 4:
                    raise EvaluationError(f"{path}:{line_number}: final_grade must be 0..4")
                _store_qrel(qrels, row["query_id"], row["page_id"], qrel, path, line_number)
    if not qrels:
        raise EvaluationError(f"No qrels found in {path}")
    return qrels


def _parse_optional_bool(
    value: str | None, path: Path, line_number: int, field: str
) -> bool | None:
    """Parse rich-qrels boolean columns while retaining legacy compatibility."""
    if value is None or not value.strip():
        return None
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise EvaluationError(f"{path}:{line_number}: invalid {field} boolean")


def load_core_groups(path: Path) -> dict[str, list[frozenset[str]]]:
    """Load the per-question core groups produced by consolidation."""
    if not path.is_file():
        raise EvaluationError(f"Core-groups file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Malformed core-groups JSON: {path}") from exc
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions, list):
        raise EvaluationError(f"{path}: expected a questions array")
    result: dict[str, list[frozenset[str]]] = {}
    seen_group_ids: set[str] = set()
    for question in questions:
        if not isinstance(question, dict):
            raise EvaluationError(f"{path}: question entry is not an object")
        query_id = question.get("question_id")
        groups = question.get("groups")
        if not isinstance(query_id, str) or not query_id or not isinstance(groups, list):
            raise EvaluationError(f"{path}: invalid question_id/groups")
        if query_id in result:
            raise EvaluationError(f"{path}: duplicate question_id {query_id}")
        parsed: list[frozenset[str]] = []
        for group in groups:
            if not isinstance(group, dict):
                raise EvaluationError(f"{path}: group is not an object")
            group_id = group.get("group_id")
            members = group.get("member_page_ids")
            if not isinstance(group_id, str) or not group_id or group_id in seen_group_ids:
                raise EvaluationError(f"{path}: missing or duplicate group_id {group_id!r}")
            if not isinstance(members, list) or not members or any(
                not isinstance(member, str) or not member for member in members
            ):
                raise EvaluationError(f"{path}: group {group_id} has invalid member_page_ids")
            member_set = frozenset(members)
            if len(member_set) != len(members):
                raise EvaluationError(f"{path}: group {group_id} repeats a page")
            seen_group_ids.add(group_id)
            parsed.append(member_set)
        result[query_id] = parsed
    return result


def _store_qrel(
    qrels: dict[str, dict[str, Qrel]], query_id: str, page_id: str,
    qrel: Qrel, path: Path, line_number: int,
) -> None:
    if not query_id or not page_id:
        raise EvaluationError(f"{path}:{line_number}: query_id/page_id cannot be empty")
    pages = qrels.setdefault(query_id, {})
    if page_id in pages:
        raise EvaluationError(f"{path}:{line_number}: duplicate {query_id}/{page_id}")
    pages[page_id] = qrel


def load_run(path: Path, expected_system: str) -> dict[str, list[str]]:
    """Load and rank one JSONL run, retaining at most 20 unique pages/query."""
    if not path.is_file():
        raise EvaluationError(f"Run file does not exist: {path}")
    ranked: dict[str, list[tuple[int, str]]] = {}
    seen_pages: set[tuple[str, str]] = set()
    seen_ranks: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                query_id = str(row["query_id"]).strip()
                page_id = str(row["page_id"]).strip()
                rank = int(row["rank"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise EvaluationError(f"{path}:{line_number}: malformed run record") from exc
            if row.get("system") not in (None, expected_system):
                raise EvaluationError(
                    f"{path}:{line_number}: system {row.get('system')!r} does not match {expected_system!r}"
                )
            if not query_id or not page_id or rank < 1:
                raise EvaluationError(f"{path}:{line_number}: invalid query_id/page_id/rank")
            if (query_id, page_id) in seen_pages:
                raise EvaluationError(f"{path}:{line_number}: duplicate {query_id}/{page_id}")
            if (query_id, rank) in seen_ranks:
                raise EvaluationError(f"{path}:{line_number}: duplicate rank {rank} for {query_id}")
            seen_pages.add((query_id, page_id))
            seen_ranks.add((query_id, rank))
            ranked.setdefault(query_id, []).append((rank, page_id))
    return {
        query_id: [page_id for _, page_id in sorted(hits)[:20]]
        for query_id, hits in ranked.items()
    }


def dcg(grades: Iterable[int]) -> float:
    """Compute DCG with final_grade as the direct (linear) relevance gain."""
    return sum(grade / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def evaluate_query(
    system: str,
    query_id: str,
    ranking: Sequence[str],
    qrels: Mapping[str, Qrel],
    core_groups: Sequence[frozenset[str]] = (),
    cutoff: int = 20,
) -> QueryMetrics:
    retrieved = list(ranking[:cutoff])
    relevant = {page_id for page_id, qrel in qrels.items() if qrel.relevant}
    core = {page_id for page_id, qrel in qrels.items() if qrel.core}
    relevant_hits = sum(page_id in relevant for page_id in retrieved)
    core_hits = sum(page_id in core for page_id in retrieved)
    retrieved_set = set(retrieved)
    core_group_hits = sum(bool(group & retrieved_set) for group in core_groups)
    judged_hits = sum(page_id in qrels for page_id in retrieved)
    grades = [qrels.get(page_id, Qrel(0.0, 0)).final_grade for page_id in retrieved]
    ideal_grades = sorted((qrel.final_grade for qrel in qrels.values()), reverse=True)[:cutoff]
    ideal_dcg = dcg(ideal_grades)
    first_relevant_rank = next(
        (rank for rank, page_id in enumerate(retrieved, start=1) if page_id in relevant),
        None,
    )
    return QueryMetrics(
        system=system,
        query_id=query_id,
        retrieved_at_20=len(retrieved),
        judged_retrieved_at_20=judged_hits,
        judged_coverage_at_20=judged_hits / len(retrieved) if retrieved else 0.0,
        relevant_reference_count=len(relevant),
        relevant_retrieved_at_20=relevant_hits,
        recall_at_20_relevant=relevant_hits / len(relevant) if relevant else 0.0,
        core_reference_count=len(core),
        core_retrieved_at_20=core_hits,
        recall_at_20_core=core_hits / len(core) if core else 0.0,
        core_group_reference_count=len(core_groups),
        core_groups_retrieved_at_20=core_group_hits,
        core_group_recall_at_20=(core_group_hits / len(core_groups) if core_groups else 0.0),
        ndcg_at_20=dcg(grades) / ideal_dcg if ideal_dcg else 0.0,
        first_relevant_rank=first_relevant_rank,
        reciprocal_rank=1.0 / first_relevant_rank if first_relevant_rank else 0.0,
    )


def evaluate_all(
    qrels: Mapping[str, Mapping[str, Qrel]],
    runs: Mapping[str, Mapping[str, Sequence[str]]],
    core_groups: Mapping[str, Sequence[frozenset[str]]] | None = None,
) -> tuple[list[QueryMetrics], dict[str, dict[str, float]]]:
    """Evaluate every system over every qrels query and return macro means."""
    details: list[QueryMetrics] = []
    summaries: dict[str, dict[str, float]] = {}
    query_ids = sorted(qrels, key=_natural_key)
    for system in SYSTEMS:
        run = runs[system]
        system_rows = [
            evaluate_query(
                system,
                query_id,
                run.get(query_id, ()),
                qrels[query_id],
                (core_groups or {}).get(query_id, ()),
            )
            for query_id in query_ids
        ]
        details.extend(system_rows)
        count = len(system_rows)
        summaries[system] = {
            "recall_relevant": sum(row.recall_at_20_relevant for row in system_rows) / count,
            "recall_core": sum(row.recall_at_20_core for row in system_rows) / count,
            "recall_core_groups": sum(row.core_group_recall_at_20 for row in system_rows) / count,
            "ndcg": sum(row.ndcg_at_20 for row in system_rows) / count,
            "mrr": sum(row.reciprocal_rank for row in system_rows) / count,
            "judged_coverage": sum(row.judged_coverage_at_20 for row in system_rows) / count,
        }
    return details, summaries


def _natural_key(value: str) -> tuple[tuple[int, object], ...]:
    import re
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def write_csv_atomic(path: Path, rows: Sequence[QueryMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(row.csv_row() for row in rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def render_summary(summaries: Mapping[str, Mapping[str, float]]) -> str:
    header = (
        "| System | Judged coverage | Relevant Recall@20 | Page-level CoreRecall@20 | "
        "CoreGroupRecall@20 | nDCG@20 | MRR |"
    )
    separator = "|---|---:|---:|---:|---:|---:|---:|"
    rows = [
        f"| {system} | {summaries[system]['judged_coverage']:.4f} | "
        f"{summaries[system]['recall_relevant']:.4f} | "
        f"{summaries[system]['recall_core']:.4f} | "
        f"{summaries[system]['recall_core_groups']:.4f} | "
        f"{summaries[system]['ndcg']:.4f} | "
        f"{summaries[system]['mrr']:.4f} |"
        for system in SYSTEMS
    ]
    return "\n".join((header, separator, *rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--core-groups", type=Path, default=DEFAULT_CORE_GROUPS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        qrels = load_qrels(args.qrels)
        core_groups = load_core_groups(args.core_groups)
        runs = {
            system: load_run(args.run_dir / f"{system}.jsonl", system)
            for system in SYSTEMS
        }
        details, summaries = evaluate_all(qrels, runs, core_groups)
        write_csv_atomic(args.output, details)
    except (EvaluationError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Evaluated {len(SYSTEMS)} systems across {len(qrels)} queries.\n")
    print(render_summary(summaries))
    incomplete = [
        (system, values["judged_coverage"])
        for system, values in summaries.items()
        if values["judged_coverage"] < 1.0
    ]
    if incomplete:
        labels = ", ".join(f"{system}={coverage:.2%}" for system, coverage in incomplete)
        print(
            "\nWARNING: Incomplete qrels coverage; scores for these systems are provisional: "
            + labels
        )
    print(f"\nDetailed results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
