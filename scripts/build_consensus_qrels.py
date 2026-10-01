"""Aggregate three independent relevance judgments into consensus qrels.

Only query-page pairs with all three judge grades are exported. Missing judge
files and incomplete pairs are warned about and skipped rather than silently
substituting a two-judge consensus.

Example:
    python scripts/build_consensus_qrels.py
    python scripts/build_consensus_qrels.py --trec-relevant-only
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable


LOGGER = logging.getLogger("build_consensus_qrels")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = REPOSITORY_ROOT / "benchmark" / "pools" / "candidate_pool.jsonl"
DEFAULT_JUDGMENT_DIR = REPOSITORY_ROOT / "benchmark" / "judgments"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "benchmark" / "qrels"
JUDGES = ("judge_a", "judge_b", "judge_c")
FULL_COLUMNS = (
    "query_id",
    "page_id",
    "r1",
    "r2",
    "r3",
    "mean_grade",
    "final_grade",
    "core",
    "relevant",
    "range",
    "variance",
    "selection_count",
)
PAGE_ID_PATTERN = re.compile(r"^DOC\d{3}_P\d{4}$")
CandidateKey = tuple[str, str]


class QrelsError(ValueError):
    """Raised when an available pool or judgment record is malformed."""


@dataclass(frozen=True)
class PoolCandidate:
    query_id: str
    page_id: str
    selection_count: int

    @property
    def key(self) -> CandidateKey:
        return (self.query_id, self.page_id)


@dataclass(frozen=True)
class ConsensusRecord:
    query_id: str
    page_id: str
    r1: int
    r2: int
    r3: int
    mean_grade: float
    final_grade: int
    core: bool
    relevant: bool
    grade_range: int
    variance: float
    selection_count: int

    def csv_row(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "page_id": self.page_id,
            "r1": self.r1,
            "r2": self.r2,
            "r3": self.r3,
            "mean_grade": f"{self.mean_grade:.6f}",
            "final_grade": self.final_grade,
            "core": str(self.core).lower(),
            "relevant": str(self.relevant).lower(),
            "range": self.grade_range,
            "variance": f"{self.variance:.6f}",
            "selection_count": self.selection_count,
        }


def natural_key(value: str) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in parts
        if part
    )


def load_candidate_pool(path: Path) -> list[PoolCandidate]:
    """Load canonical candidates and selection counts in deterministic order."""

    if not path.is_file():
        raise QrelsError(f"Candidate pool does not exist: {path}")
    candidates: list[PoolCandidate] = []
    seen: set[CandidateKey] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            label = f"{path}:{line_number}"
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise QrelsError(f"{label}: invalid JSON: {error}") from error
            if not isinstance(item, dict):
                raise QrelsError(f"{label}: record must be an object")
            query_id = item.get("query_id")
            page_id = item.get("page_id")
            selection_count = item.get("selection_count")
            if not isinstance(query_id, str) or not query_id.strip():
                raise QrelsError(f"{label}: invalid query_id")
            if not isinstance(page_id, str) or not PAGE_ID_PATTERN.fullmatch(page_id):
                raise QrelsError(f"{label}: invalid page_id")
            if (
                type(selection_count) is not int
                or not 1 <= selection_count <= 6
            ):
                raise QrelsError(f"{label}: selection_count must be an integer 1..6")
            key = (query_id.strip(), page_id)
            if key in seen:
                raise QrelsError(f"{label}: duplicate candidate {key}")
            seen.add(key)
            candidates.append(
                PoolCandidate(key[0], key[1], selection_count)
            )
    if not candidates:
        LOGGER.warning("Candidate pool contains no records: %s", path)
    return sorted(
        candidates,
        key=lambda candidate: (
            natural_key(candidate.query_id),
            candidate.page_id,
        ),
    )


def load_judgments(
    judgment_dir: Path,
    candidate_keys: set[CandidateKey],
    *,
    judges: Iterable[str] = JUDGES,
) -> tuple[dict[str, dict[CandidateKey, int]], list[str], list[str]]:
    """Load available judge grades and reject duplicates or out-of-pool records."""

    judge_order = tuple(judges)
    grades: dict[str, dict[CandidateKey, int]] = {
        judge: {} for judge in judge_order
    }
    available: list[str] = []
    missing: list[str] = []
    for judge in judge_order:
        path = judgment_dir / f"{judge}.jsonl"
        if not path.is_file():
            LOGGER.warning("Judgment file missing; continuing without %s: %s", judge, path)
            missing.append(judge)
            continue
        available.append(judge)
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                label = f"{path}:{line_number}"
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise QrelsError(f"{label}: invalid JSON: {error}") from error
                if not isinstance(item, dict):
                    raise QrelsError(f"{label}: record must be an object")
                query_id = item.get("question_id", item.get("query_id"))
                page_id = item.get("page_id")
                grade = item.get("grade")
                rationale = item.get("rationale")
                if not isinstance(query_id, str) or not query_id.strip():
                    raise QrelsError(f"{label}: invalid query_id")
                if (
                    not isinstance(page_id, str)
                    or not PAGE_ID_PATTERN.fullmatch(page_id)
                ):
                    raise QrelsError(f"{label}: invalid page_id")
                if type(grade) is not int or not 0 <= grade <= 4:
                    raise QrelsError(f"{label}: grade must be an integer 0..4")
                if not isinstance(rationale, str) or not rationale.strip():
                    raise QrelsError(f"{label}: rationale must be non-empty")
                key = (query_id.strip(), page_id)
                if key not in candidate_keys:
                    raise QrelsError(f"{label}: judgment is not in candidate pool: {key}")
                if key in grades[judge]:
                    raise QrelsError(f"{label}: duplicate judgment for {key}")
                grades[judge][key] = grade
    return grades, available, missing


def round_grade(mean_grade: float) -> int:
    """Round to the nearest integer and clamp to the qrels range 0..4."""

    return max(0, min(4, int(math.floor(mean_grade + 0.5))))


def aggregate_grades(
    candidate: PoolCandidate,
    grades: tuple[int, int, int],
) -> ConsensusRecord:
    """Apply the exact Milestone E math and consensus thresholds."""

    r1, r2, r3 = grades
    if any(type(grade) is not int or not 0 <= grade <= 4 for grade in grades):
        raise QrelsError("All consensus grades must be integers from 0 through 4")
    mean_grade = sum(grades) / 3.0
    grade_range = max(grades) - min(grades)
    variance = sum((grade - mean_grade) ** 2 for grade in grades) / 3.0
    relevant = mean_grade >= 2.5 and sum(grade >= 2 for grade in grades) >= 2
    core = mean_grade >= 3.5 and sum(grade == 4 for grade in grades) >= 2
    return ConsensusRecord(
        query_id=candidate.query_id,
        page_id=candidate.page_id,
        r1=r1,
        r2=r2,
        r3=r3,
        mean_grade=mean_grade,
        final_grade=round_grade(mean_grade),
        core=core,
        relevant=relevant,
        grade_range=grade_range,
        variance=variance,
        selection_count=candidate.selection_count,
    )


def build_consensus_records(
    candidates: list[PoolCandidate],
    judge_grades: dict[str, dict[CandidateKey, int]],
    *,
    judges: Iterable[str] = JUDGES,
) -> tuple[list[ConsensusRecord], list[CandidateKey]]:
    """Aggregate complete triples and report incomplete candidate keys."""

    judge_order = tuple(judges)
    if len(judge_order) != 3:
        raise ValueError("Consensus aggregation requires exactly three judges")
    records: list[ConsensusRecord] = []
    incomplete: list[CandidateKey] = []
    for candidate in candidates:
        key = candidate.key
        if any(key not in judge_grades.get(judge, {}) for judge in judge_order):
            incomplete.append(key)
            continue
        grades = tuple(judge_grades[judge][key] for judge in judge_order)
        records.append(aggregate_grades(candidate, grades))
    return records, incomplete


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def render_full_csv(records: list[ConsensusRecord]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=FULL_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(record.csv_row() for record in records)
    return buffer.getvalue()


def render_trec_qrels(
    records: list[ConsensusRecord],
    *,
    relevant_only: bool,
) -> str:
    selected = (
        [record for record in records if record.relevant or record.core]
        if relevant_only
        else records
    )
    return "".join(
        f"{record.query_id}\t0\t{record.page_id}\t{record.final_grade}\n"
        for record in selected
    )


def build_consensus_qrels(
    *,
    pool_path: Path = DEFAULT_POOL,
    judgment_dir: Path = DEFAULT_JUDGMENT_DIR,
    full_output: Path | None = None,
    trec_output: Path | None = None,
    relevant_only: bool = False,
    judges: Iterable[str] = JUDGES,
) -> dict[str, Any]:
    """Build rich CSV and standard TREC qrels from complete three-judge votes."""

    output_dir = DEFAULT_OUTPUT_DIR
    full_output = full_output or output_dir / "consensus_qrels_full.csv"
    trec_output = trec_output or output_dir / "consensus_qrels.tsv"
    judge_order = tuple(judges)
    candidates = load_candidate_pool(pool_path)
    candidate_keys = {candidate.key for candidate in candidates}
    judge_grades, available, missing = load_judgments(
        judgment_dir,
        candidate_keys,
        judges=judge_order,
    )
    records, incomplete = build_consensus_records(
        candidates,
        judge_grades,
        judges=judge_order,
    )
    if incomplete:
        LOGGER.warning(
            "Skipping %d candidates without a complete three-judge grade tuple",
            len(incomplete),
        )

    atomic_write_text(full_output, render_full_csv(records))
    atomic_write_text(
        trec_output,
        render_trec_qrels(records, relevant_only=relevant_only),
    )
    return {
        "candidate_count": len(candidates),
        "consensus_count": len(records),
        "incomplete_count": len(incomplete),
        "relevant_count": sum(record.relevant for record in records),
        "core_count": sum(record.core for record in records),
        "judges_available": available,
        "judges_missing": missing,
        "trec_relevant_only": relevant_only,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate three blind judge runs into consensus graded qrels."
    )
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--judgment-dir", type=Path, default=DEFAULT_JUDGMENT_DIR)
    parser.add_argument(
        "--full-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "consensus_qrels_full.csv",
    )
    parser.add_argument(
        "--trec-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "consensus_qrels.tsv",
    )
    parser.add_argument(
        "--trec-relevant-only",
        action="store_true",
        help="Exclude non-relevant candidates from the TREC file.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )
    try:
        stats = build_consensus_qrels(
            pool_path=args.pool,
            judgment_dir=args.judgment_dir,
            full_output=args.full_output,
            trec_output=args.trec_output,
            relevant_only=args.trec_relevant_only,
        )
    except (OSError, QrelsError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    LOGGER.info(
        "Wrote %d consensus qrels (%d relevant, %d core); skipped %d incomplete",
        stats["consensus_count"],
        stats["relevant_count"],
        stats["core_count"],
        stats["incomplete_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
