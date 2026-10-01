#!/usr/bin/env python3
"""Milestone G: external sanity check against Songxi's manual audit.

The manual reference set below was extracted from:

    CDIRP_V3_Independent_Rescoring_Report_2026-09-17.docx

The report states that all page coordinates are one-based PDF pages. Most
coordinates appear directly in its corrected reference answers or citation
corrections. Three evidence descriptions omit a PDF page; their canonical page
IDs were resolved by matching the quoted facts against the frozen
``benchmark/corpus_pages.jsonl``:

* Q1 DOC002, the explicit 2023-2028 planning period -> PDF page 1.
* Q5 DOC016, the 19-ton / 10,000-fish / 27% result -> its sole frozen page 1.
* Q8 DOC002, Clean Water Act coordination, Section 319, and wetland treatment /
  easement measures -> PDF pages 9, 12, and 14.

No retrieval ranks, system identities, or model outputs are used by this check.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QRELS = REPO_ROOT / "benchmark" / "qrels" / "consensus_qrels_full.csv"


@dataclass(frozen=True)
class ManualEvidence:
    """One human-audited query/page pair and its report-level basis."""

    query_id: str
    page_id: str
    evidence: str


# Keep this tuple ordered by query and by the order in which evidence is
# discussed in the independent report. Canonical page IDs use one-based PDF
# pages padded to four digits.
SONGXI_CRITICAL_PAIRS: tuple[ManualEvidence, ...] = (
    ManualEvidence("Q1", "DOC002_P0001", "2023-2028 Missouri Wetland Program Plan"),
    ManualEvidence("Q1", "DOC003_P0002", "1998 update to the plan launched in 1986"),
    ManualEvidence("Q2", "DOC001_P0028", "national, rather than Missouri, threat ranking"),
    ManualEvidence("Q3", "DOC036_P0106", "wetland loss, WRE acres, and 2018 bird sample"),
    ManualEvidence("Q3", "DOC001_P0140", "later 13-15% wetland estimate"),
    ManualEvidence("Q3", "DOC002_P0015", "ACEP-WRE and EWPP-FPE combined acreage"),
    ManualEvidence("Q4", "DOC001_P0028", "statewide invasive-species framing"),
    ManualEvidence("Q4", "DOC001_P0029", "continuation of statewide cross-habitat framing"),
    ManualEvidence("Q4", "DOC036_P0308", "2022 aquatic invasive-species list and treatment"),
    ManualEvidence("Q4", "DOC036_P0358", "separate grass-carp harvest discussion"),
    ManualEvidence("Q5", "DOC016_P0001", "19 tons, 10,000 silver carp, and 27% density decline"),
    ManualEvidence("Q6", "DOC036_P0126", "flood-to-establishment evidence without climate attribution"),
    ManualEvidence("Q7", "DOC036_P0069", "explicit countries and partner organizations"),
    ManualEvidence("Q7", "DOC036_P0070", "full-life-cycle conservation case details"),
    ManualEvidence("Q7", "DOC036_P0071", "full-life-cycle conservation case details"),
    ManualEvidence("Q8", "DOC036_P0382", "buffers, bioreactors, treatment, and external trading example"),
    ManualEvidence("Q8", "DOC002_P0009", "Clean Water Act Sections 401/404 coordination"),
    ManualEvidence("Q8", "DOC002_P0012", "Section 319 nonpoint-source projects"),
    ManualEvidence("Q8", "DOC002_P0014", "wetland treatment, buffers, and WRE measures"),
    ManualEvidence("Q9", "DOC034_P0131", "Montreal fund contribution, secretariat, and assistance"),
    ManualEvidence("Q9", "DOC033_P0009", "Cleveland projects, matching, and funding amounts"),
    ManualEvidence("Q9", "DOC033_P0077", "need for larger-scale Cleveland financing"),
)


def parse_bool(value: object) -> bool:
    """Parse the boolean encodings accepted in the qrels CSV."""

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Unrecognized boolean value: {value!r}")


def load_qrels(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Load qrels keyed by (query_id, page_id), rejecting duplicate pairs."""

    required = {"query_id", "page_id", "mean_grade", "relevant", "core"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = required.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing_columns)}"
            )

        rows: dict[tuple[str, str], dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            key = (row["query_id"].strip(), row["page_id"].strip())
            if key in rows:
                raise ValueError(f"Duplicate qrels pair {key} at CSV line {line_number}")
            rows[key] = row
    return rows


def check_pairs(
    qrels: Mapping[tuple[str, str], Mapping[str, str]],
    references: Iterable[ManualEvidence] = SONGXI_CRITICAL_PAIRS,
) -> list[dict[str, object]]:
    """Return normalized check results without changing qrels semantics."""

    results: list[dict[str, object]] = []
    for reference in references:
        row = qrels.get((reference.query_id, reference.page_id))
        if row is None:
            results.append(
                {
                    "query_id": reference.query_id,
                    "page_id": reference.page_id,
                    "evidence": reference.evidence,
                    "found": False,
                }
            )
            continue

        results.append(
            {
                "query_id": reference.query_id,
                "page_id": reference.page_id,
                "evidence": reference.evidence,
                "found": True,
                "mean_grade": float(row["mean_grade"]),
                "relevant": parse_bool(row["relevant"]),
                "core": parse_bool(row["core"]),
            }
        )
    return results


def print_report(results: Iterable[Mapping[str, object]], qrels_path: Path) -> int:
    """Print the external check and return 0 even when pages are missing.

    A missing pair is a benchmark diagnostic result, not a script execution
    error. File/schema failures still raise and produce a non-zero process exit.
    """

    rows = list(results)
    print("Milestone G - External Sanity Check")
    print(f"Qrels: {qrels_path}")
    print(f"Manual reference pairs: {len(rows)}")
    print()

    current_query: str | None = None
    for row in rows:
        query_id = str(row["query_id"])
        if query_id != current_query:
            if current_query is not None:
                print()
            print(f"{query_id}")
            current_query = query_id

        prefix = f"  {row['page_id']}"
        if not row["found"]:
            print(f"{prefix}: MISSING (Not retrieved by any system)")
        else:
            print(
                f"{prefix}: mean_grade={row['mean_grade']:.6f}, "
                f"relevant={str(row['relevant']).lower()}, "
                f"core={str(row['core']).lower()}"
            )
        print(f"    Evidence: {row['evidence']}")

    found = sum(bool(row["found"]) for row in rows)
    relevant = sum(bool(row.get("relevant", False)) for row in rows)
    core = sum(bool(row.get("core", False)) for row in rows)
    missing = len(rows) - found
    print()
    print("Summary")
    print(f"  Present in qrels: {found}/{len(rows)}")
    print(f"  Missing: {missing}")
    print(f"  Rated relevant: {relevant}/{found}")
    print(f"  Rated core: {core}/{found}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Songxi's manual critical pages against consensus qrels."
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=DEFAULT_QRELS,
        help=f"Consensus qrels CSV (default: {DEFAULT_QRELS})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qrels_path = args.qrels.expanduser().resolve()
    qrels = load_qrels(qrels_path)
    return print_report(check_pairs(qrels), qrels_path)


if __name__ == "__main__":
    raise SystemExit(main())
