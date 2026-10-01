"""Build the Milestone C candidate pool and retrieval overlap diagnostics.

Candidate identity is the pair (query_id, page_id). This preserves query scope
when the same canonical page is selected for more than one benchmark question.

Example:
    python scripts/build_candidate_pool.py
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable


LOGGER = logging.getLogger("build_candidate_pool")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPOSITORY_ROOT / "benchmark" / "candidate_runs"
DEFAULT_POOL_PATH = REPOSITORY_ROOT / "benchmark" / "pools" / "candidate_pool.jsonl"
DEFAULT_OVERLAP_PATH = REPOSITORY_ROOT / "benchmark" / "analysis" / "pool_overlap.csv"
DEFAULT_STATS_PATH = REPOSITORY_ROOT / "benchmark" / "analysis" / "pool_stats.json"
SYSTEMS = ("bm25", "dense", "hybrid", "llm_a", "llm_b", "llm_c", "copilot", "gpt_6_luna", "claude_haiku", "gemini_flash")
PAGE_ID_PATTERN = re.compile(r"^DOC\d{3}_P\d{4}$")
CandidateKey = tuple[str, str]


class CandidatePoolError(ValueError):
    """Raised when an available candidate run is malformed."""


def natural_key(value: str) -> tuple[tuple[int, object], ...]:
    """Return a deterministic key that sorts Q2 before Q10."""

    parts = re.split(r"(\d+)", value)
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in parts
        if part
    )


def _require_string(item: dict[str, Any], field: str, source: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CandidatePoolError(f"{source}: {field} must be a non-empty string")
    return value.strip()


def load_candidate_runs(
    run_dir: Path,
    *,
    systems: Iterable[str] = SYSTEMS,
) -> tuple[
    dict[CandidateKey, dict[str, int]],
    dict[str, set[CandidateKey]],
    list[str],
    list[str],
]:
    """Load available runs, deduplicate candidates, and retain best system rank."""

    system_order = tuple(systems)
    provenance: dict[CandidateKey, dict[str, int]] = {}
    selections = {system: set() for system in system_order}
    available: list[str] = []
    missing: list[str] = []

    for system in system_order:
        path = run_dir / f"{system}.jsonl"
        if not path.is_file():
            LOGGER.warning("Candidate run missing; continuing without %s: %s", system, path)
            missing.append(system)
            continue
        available.append(system)
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                label = f"{path}:{line_number}"
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CandidatePoolError(f"{label}: invalid JSON: {error}") from error
                if not isinstance(item, dict):
                    raise CandidatePoolError(f"{label}: record must be a JSON object")

                query_id = _require_string(item, "query_id", label)
                page_id = _require_string(item, "page_id", label).upper()
                record_system = _require_string(item, "system", label)
                rank = item.get("rank")
                if record_system != system:
                    raise CandidatePoolError(
                        f"{label}: system={record_system!r} does not match file {system!r}"
                    )
                if not PAGE_ID_PATTERN.fullmatch(page_id):
                    raise CandidatePoolError(
                        f"{label}: invalid canonical page_id {page_id!r}"
                    )
                if type(rank) is not int or rank < 1:
                    raise CandidatePoolError(f"{label}: rank must be a positive integer")

                key = (query_id, page_id)
                previous = provenance.setdefault(key, {}).get(system)
                if previous is not None:
                    best_rank = min(previous, rank)
                    provenance[key][system] = best_rank
                    LOGGER.warning(
                        "Duplicate %s/%s in %s; retaining best rank %d",
                        query_id,
                        page_id,
                        system,
                        best_rank,
                    )
                else:
                    provenance[key][system] = rank
                selections[system].add(key)

    return provenance, selections, available, missing


def build_pool_records(
    provenance: dict[CandidateKey, dict[str, int]],
    *,
    systems: Iterable[str] = SYSTEMS,
) -> list[dict[str, Any]]:
    """Create stable, strictly deduplicated candidate pool records."""

    system_order = tuple(systems)
    records: list[dict[str, Any]] = []
    for query_id, page_id in sorted(
        provenance,
        key=lambda key: (natural_key(key[0]), key[1]),
    ):
        ranks = provenance[(query_id, page_id)]
        ordered_provenance = {
            system: ranks[system] for system in system_order if system in ranks
        }
        records.append(
            {
                "query_id": query_id,
                "page_id": page_id,
                "selection_count": len(ordered_provenance),
                "provenance": ordered_provenance,
            }
        )
    return records


def jaccard(left: set[CandidateKey], right: set[CandidateKey]) -> float:
    """Compute Jaccard overlap; define two empty sets as 0 (no observations)."""

    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def calculate_overlap_matrix(
    selections: dict[str, set[CandidateKey]],
    *,
    systems: Iterable[str] = SYSTEMS,
) -> dict[str, dict[str, float]]:
    """Return the complete ordered system-by-system Jaccard matrix."""

    system_order = tuple(systems)
    return {
        left: {
            right: jaccard(selections.get(left, set()), selections.get(right, set()))
            for right in system_order
        }
        for left in system_order
    }


def calculate_pool_stats(
    records: list[dict[str, Any]],
    *,
    available_systems: list[str],
    missing_systems: list[str],
    systems: Iterable[str] = SYSTEMS,
) -> dict[str, Any]:
    """Summarize pool size, agreement, and exclusive system contributions."""

    system_order = tuple(systems)
    selection_distribution = {
        str(count): sum(
            1 for record in records if record["selection_count"] == count
        )
        for count in range(1, len(system_order) + 1)
    }
    exclusive = {system: 0 for system in system_order}
    exclusive_page_ids = {system: set() for system in system_order}
    for record in records:
        provenance = record["provenance"]
        if len(provenance) == 1:
            system = next(iter(provenance))
            exclusive[system] += 1
            exclusive_page_ids[system].add(record["page_id"])

    query_ids = {record["query_id"] for record in records}
    distinct_page_ids = {record["page_id"] for record in records}
    return {
        "systems_expected": list(system_order),
        "systems_available": available_systems,
        "systems_missing": missing_systems,
        "query_count": len(query_ids),
        "total_unique_candidate_pages_across_queries": len(records),
        "total_distinct_page_ids": len(distinct_page_ids),
        "selection_count_distribution": selection_distribution,
        "exclusive_pages_by_system": exclusive,
        "exclusive_distinct_page_ids_by_system": {
            system: len(exclusive_page_ids[system]) for system in system_order
        },
    }


def atomic_write_text(path: Path, content: str) -> None:
    """Write UTF-8 text durably, then atomically replace the destination."""

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


def render_pool_jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )


def render_overlap_csv(
    matrix: dict[str, dict[str, float]],
    *,
    systems: Iterable[str] = SYSTEMS,
) -> str:
    system_order = tuple(systems)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["system", *system_order])
    for left in system_order:
        writer.writerow(
            [left, *(f"{matrix[left][right]:.6f}" for right in system_order)]
        )
    return buffer.getvalue()


def build_candidate_pool(
    *,
    run_dir: Path = DEFAULT_RUN_DIR,
    pool_path: Path = DEFAULT_POOL_PATH,
    overlap_path: Path = DEFAULT_OVERLAP_PATH,
    stats_path: Path = DEFAULT_STATS_PATH,
    systems: Iterable[str] = SYSTEMS,
) -> dict[str, Any]:
    """Build all Milestone C artifacts and return the summary metrics."""

    system_order = tuple(systems)
    provenance, selections, available, missing = load_candidate_runs(
        run_dir,
        systems=system_order,
    )
    records = build_pool_records(provenance, systems=system_order)
    matrix = calculate_overlap_matrix(selections, systems=system_order)
    stats = calculate_pool_stats(
        records,
        available_systems=available,
        missing_systems=missing,
        systems=system_order,
    )

    # All in-memory computation and validation completes before any destination
    # is replaced. Each individual artifact replacement is atomic.
    atomic_write_text(pool_path, render_pool_jsonl(records))
    atomic_write_text(overlap_path, render_overlap_csv(matrix, systems=system_order))
    atomic_write_text(
        stats_path,
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deduplicated Milestone C candidate pool and diagnostics."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--pool-output", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--overlap-output", type=Path, default=DEFAULT_OVERLAP_PATH)
    parser.add_argument("--stats-output", type=Path, default=DEFAULT_STATS_PATH)
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
        stats = build_candidate_pool(
            run_dir=args.run_dir,
            pool_path=args.pool_output,
            overlap_path=args.overlap_output,
            stats_path=args.stats_output,
        )
    except (CandidatePoolError, OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1

    LOGGER.info(
        "Wrote %d unique query-page candidates from %d available systems",
        stats["total_unique_candidate_pages_across_queries"],
        len(stats["systems_available"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
