"""Professor Shang Section 7 core-evidence retrieval diagnostics.

Uses reviewed core groups, V2 qrels, frozen Top-20 runs, and isolated Top-100
runs. Outputs deterministic CSV artifacts; failure-taxonomy labels are explicitly
marked as heuristic diagnoses with their supporting rule.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = (
    "bm25", "dense", "hybrid", "llm_a", "llm_b", "llm_c",
    "copilot", "gpt_6_luna", "claude_haiku", "gemini_flash",
)
ORIGINAL_SIX = ("bm25", "dense", "hybrid", "llm_a", "llm_b", "llm_c")
CONVENTIONAL = {"bm25", "dense", "hybrid"}
LLM_SYSTEMS = set(SYSTEMS) - CONVENTIONAL - {"copilot"}
PAGE_RE = re.compile(r"^(DOC\d{3})_P(\d{4})$")
TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
EXPECTED_STRUCTURE = {
    "Q1": "Current document identity; active planning horizon; publication vs. planning/revision period distinction.",
    "Q2": "Explicit #1 and #2 threat evidence; evidence boundary showing no supported official #3.",
    "Q3": "Wetland status/loss; impacts/ecological services; restoration progress; current management/recommendations; at least three documents.",
    "Q4": "2015 SWAP treatment of AIS; 2022 CCS treatment of AIS; framework-level difference.",
    "Q5": "Removal amount; fish count; 27% density reduction; Lower Grand River geographic scope.",
    "Q6": "Evidence supporting interacting-risk framing; evidence limiting or rejecting direct/sole causation.",
    "Q7": "Species; Missouri breeding context; migration/wintering geography; named countries; named partner organizations.",
    "Q8": "Technological/land-management actions; regulatory mechanisms; market/financial mechanisms; implementation sequence.",
    "Q9": "Montreal fund structure and administrative features; Cleveland fund structure and administrative features; supported comparison boundaries.",
}


class DiagnosticError(ValueError):
    pass


def atomic_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def load_runs(directory: Path) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for system in SYSTEMS:
        path = directory / f"{system}.jsonl"
        if not path.is_file():
            continue
        grouped: dict[str, list[tuple[int, str]]] = {}
        with path.open(encoding="utf-8-sig") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                qid, pid, rank = row.get("query_id"), row.get("page_id"), row.get("rank")
                if not isinstance(qid, str) or not isinstance(pid, str) or type(rank) is not int:
                    raise DiagnosticError(f"{path}:{number}: malformed run row")
                grouped.setdefault(qid, []).append((rank, pid))
        result[system] = {}
        for qid, items in grouped.items():
            items.sort()
            ranks = [rank for rank, _ in items]
            if ranks != list(range(1, len(items) + 1)):
                raise DiagnosticError(f"{path}: non-contiguous ranks for {qid}")
            pages = [pid for _, pid in items]
            if len(pages) != len(set(pages)):
                raise DiagnosticError(f"{path}: duplicate pages for {qid}")
            result[system][qid] = pages
    return result


def load_qrels(path: Path) -> tuple[dict[str, int], set[tuple[str, str]]]:
    relevant: Counter[str] = Counter()
    core: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["relevant"].lower() == "true":
                relevant[row["query_id"]] += 1
            if row["core"].lower() == "true":
                core.add((row["query_id"], row["page_id"]))
    return dict(relevant), core


def load_groups(path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("manual_review_status") != "completed":
        raise DiagnosticError("core_groups.json is not marked completed")
    groups: list[dict[str, Any]] = []
    questions: dict[str, str] = {}
    for question in payload.get("questions", []):
        qid = question["question_id"]
        questions[qid] = question["question"]
        if question.get("manual_review_required"):
            raise DiagnosticError(f"{qid} is still marked for manual review")
        for group in question["groups"]:
            members = list(group["member_page_ids"])
            if not members:
                raise DiagnosticError(f"{group['group_id']} has no members")
            groups.append({
                "query_id": qid,
                "group_id": group["group_id"],
                "facet_description": group["facet_description"],
                "members": members,
                "confidence": group["redundancy_confidence"],
            })
    return groups, questions


def load_corpus(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                result[row["page_id"]] = row.get("page_text", "")
    return result


def rank_of_group(ranking: list[str], members: list[str]) -> int | None:
    positions = {pid: i for i, pid in enumerate(ranking, 1)}
    ranks = [positions[pid] for pid in members if pid in positions]
    return min(ranks) if ranks else None


def complementarity_rows(
    systems: Iterable[str],
    groups: list[dict[str, Any]],
    runs: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    """Build cumulative core-group coverage in the supplied system order.

    The explicit order preserves Professor Shang's original six-system
    experiment while allowing an extended experiment that includes Copilot
    and the later lightweight-model runs.
    """
    covered: set[tuple[str, str]] = set()
    all_group_keys = {(g["query_id"], g["group_id"]) for g in groups}
    rows: list[dict[str, Any]] = []
    for index, system in enumerate(tuple(systems), 1):
        before = len(covered)
        for group in groups:
            ranking = runs.get(system, {}).get(group["query_id"], [])
            if rank_of_group(ranking[:20], group["members"]) is not None:
                covered.add((group["query_id"], group["group_id"]))
        rows.append({
            "addition_order": index,
            "system_added": system,
            "new_core_groups_contributed": len(covered) - before,
            "covered_core_groups": len(covered),
            "total_core_groups": len(all_group_keys),
            "cumulative_core_group_coverage": f"{len(covered)/len(all_group_keys):.6f}",
        })
    return rows


def available_depth_diagnostics(
    system: str,
    groups_by_query: dict[str, list[dict[str, Any]]],
    runs: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    """Measure a sparse external run at its actual available depth."""
    rows: list[dict[str, Any]] = []
    total_candidates = total_groups = total_covered = 0
    recalls: list[float] = []
    for qid in sorted(groups_by_query, key=lambda x: int(x[1:])):
        ranking = runs.get(system, {}).get(qid, [])
        qgroups = groups_by_query[qid]
        covered = sum(
            rank_of_group(ranking, group["members"]) is not None
            for group in qgroups
        )
        recall = covered / len(qgroups) if qgroups else 0.0
        recalls.append(recall)
        total_candidates += len(ranking)
        total_groups += len(qgroups)
        total_covered += covered
        rows.append({
            "scope": "query",
            "query_id": qid,
            "candidate_count": len(ranking),
            "requested_top20_complete": len(ranking) >= 20,
            "top50_available": len(ranking) >= 50,
            "top100_available": len(ranking) >= 100,
            "core_group_count": len(qgroups),
            "covered_core_groups_at_available_depth": covered,
            "core_group_recall_at_available_depth": f"{recall:.6f}",
            "limitation": "" if len(ranking) >= 20 else "SPARSE_EXTERNAL_RUN_BELOW_TOP20",
        })
    rows.append({
        "scope": "overall",
        "query_id": "ALL",
        "candidate_count": total_candidates,
        "requested_top20_complete": all(
            len(runs.get(system, {}).get(qid, [])) >= 20 for qid in groups_by_query
        ),
        "top50_available": False,
        "top100_available": False,
        "core_group_count": total_groups,
        "covered_core_groups_at_available_depth": total_covered,
        "core_group_recall_at_available_depth": f"{total_covered/total_groups:.6f}",
        "limitation": (
            f"MACRO_RECALL={sum(recalls)/len(recalls):.6f}; "
            "sponsor-supplied candidates only; no Copilot Top-100 run"
        ),
    })
    return rows


def adjacent_hits(page_id: str, ranking: list[str], cutoff: int = 20) -> list[tuple[str, int, int]]:
    match = PAGE_RE.fullmatch(page_id)
    if not match:
        return []
    doc, page = match.group(1), int(match.group(2))
    output = []
    for rank, candidate in enumerate(ranking[:cutoff], 1):
        other = PAGE_RE.fullmatch(candidate)
        if other and other.group(1) == doc:
            distance = int(other.group(2)) - page
            if 0 < abs(distance) <= 2:
                output.append((candidate, rank, distance))
    return output


def token_overlap(query: str, texts: Iterable[str]) -> float:
    q = {x.casefold() for x in TOKEN_RE.findall(query) if len(x) > 2}
    p = {x.casefold() for text in texts for x in TOKEN_RE.findall(text) if len(x) > 2}
    return len(q & p) / len(q) if q else 0.0


def classify_failure(
    *, group: dict[str, Any], query: str, ranking20: list[str],
    ranking100: list[str] | None, corpus: dict[str, str],
    covered_groups_for_query: int,
) -> tuple[str, str]:
    members = group["members"]
    if group["confidence"] == "low":
        return "Question/qrel ambiguity", "reviewed redundancy_confidence=low"
    deep_rank = rank_of_group(ranking100 or [], members)
    if deep_rank is not None and deep_rank > 20:
        return "Ranking failure", f"first valid group member occurs at rank {deep_rank}"
    adjacency = [hit for pid in members for hit in adjacent_hits(pid, ranking20)]
    if adjacency:
        return "Semantic near miss", "Top-20 contains a same-document page within +/-2 PDF pages"
    docs = {PAGE_RE.fullmatch(pid).group(1) for pid in members if PAGE_RE.fullmatch(pid)}
    reached_docs = {
        PAGE_RE.fullmatch(pid).group(1)
        for pid in (ranking100 or ranking20)
        if PAGE_RE.fullmatch(pid)
    }
    if docs and not (docs & reached_docs):
        return "Wrong-document routing", "no target core-group document appears in deepest available ranking"
    texts = [corpus.get(pid, "") for pid in members]
    if texts and max((len(t.strip()) for t in texts), default=0) < 120:
        return "Extraction/layout failure", "all designated core pages contain less than 120 extracted characters"
    if group["query_id"] == "Q1":
        return "Temporal/metadata failure", "Q1 requires publication/planning/revision lifecycle distinctions"
    overlap = token_overlap(query, texts)
    if overlap < 0.12:
        return "Lexical mismatch", f"query-to-core-page token overlap={overlap:.3f}"
    if covered_groups_for_query > 0:
        return "Multi-facet query failure", "other core facets for this question were covered at Top-20"
    return "Semantic near miss", "target document was reached but no valid group member ranked in available depth"


def analyze(
    *, qrels_path: Path, groups_path: Path, corpus_path: Path,
    top20_dir: Path, depth_dir: Path, output_dir: Path,
) -> dict[str, int]:
    relevant_counts, core_qrels = load_qrels(qrels_path)
    groups, questions = load_groups(groups_path)
    corpus = load_corpus(corpus_path)
    top20 = load_runs(top20_dir)
    deep = load_runs(depth_dir)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        grouped.setdefault(group["query_id"], []).append(group)

    # 7.1 sanity audit
    sanity = []
    for qid in sorted(questions, key=lambda x: int(x[1:])):
        qgroups = grouped.get(qid, [])
        members = {(qid, pid) for g in qgroups for pid in g["members"]}
        expected = {pair for pair in core_qrels if pair[0] == qid}
        sanity.append({
            "query_id": qid,
            "relevant_page_count": relevant_counts.get(qid, 0),
            "provisional_core_page_count": len(expected),
            "final_core_group_count": len(qgroups),
            "page_instances_per_group": "; ".join(
                f"{g['group_id']}={len(g['members'])}" for g in qgroups
            ),
            "all_core_pages_grouped": members == expected,
            "expected_evidence_structure": EXPECTED_STRUCTURE[qid],
            "reviewed_facet_descriptions": " | ".join(g["facet_description"] for g in qgroups),
            "major_requirements_covered": "YES_HUMAN_REVIEW_COMPLETED",
            "sanity_question_answer": "YES_HUMAN_REVIEW_COMPLETED",
        })
    atomic_csv(output_dir / "core_set_sanity_audit.csv", list(sanity[0]), sanity)

    # 7.2 group depth coverage and 7.5 contribution matrix
    coverage_rows = []
    depth_rows = []
    for system in SYSTEMS:
        for qid in sorted(questions, key=lambda x: int(x[1:])):
            qgroups = grouped.get(qid, [])
            deep_ranking = deep.get(system, {}).get(qid)
            ranking = deep_ranking or top20.get(system, {}).get(qid, [])
            max_depth = len(ranking)
            for cutoff in (20, 50, 100):
                available = cutoff <= max_depth
                hits = sum(
                    rank_of_group(ranking[:cutoff], group["members"]) is not None
                    for group in qgroups
                ) if available else None
                depth_rows.append({
                    "system": system, "query_id": qid, "cutoff": cutoff,
                    "available": available, "retrieved_depth": max_depth,
                    "core_group_count": len(qgroups),
                    "covered_core_groups": "" if hits is None else hits,
                    "core_group_recall": "" if hits is None or not qgroups else f"{hits/len(qgroups):.6f}",
                })

    for group in groups:
        qid = group["query_id"]
        row = {
            "query_id": qid, "group_id": group["group_id"],
            "facet_description": group["facet_description"],
            "member_page_ids": ";".join(group["members"]),
        }
        found_original = []
        found_all = []
        for system in SYSTEMS:
            ranking = top20.get(system, {}).get(qid, [])
            rank = rank_of_group(ranking[:20], group["members"])
            row[f"{system}_rank_at_20"] = "" if rank is None else rank
            if rank is not None:
                found_all.append(system)
                if system in ORIGINAL_SIX:
                    found_original.append(system)
        original_set = set(found_original)
        if len(original_set) == len(ORIGINAL_SIX):
            category = "FOUND_BY_ALL_ORIGINAL_SIX"
        elif original_set and original_set <= CONVENTIONAL:
            category = "CONVENTIONAL_ONLY"
        elif original_set and original_set <= set(ORIGINAL_SIX) - CONVENTIONAL:
            category = "LLM_ONLY"
        elif original_set:
            category = "MIXED"
        else:
            category = "MISSING_FROM_ORIGINAL_SIX"
        row.update({
            "original_six_finders": ";".join(found_original),
            "all_ten_finders": ";".join(found_all),
            "contribution_category": category,
            "unique_contributor": found_original[0] if len(found_original) == 1 else "",
        })
        coverage_rows.append(row)
    coverage_fields = list(coverage_rows[0])
    atomic_csv(output_dir / "core_group_coverage_matrix.csv", coverage_fields, coverage_rows)
    atomic_csv(output_dir / "retrieval_depth_analysis.csv", list(depth_rows[0]), depth_rows)

    # Preserve the professor's original six-system experiment, then publish an
    # extended ten-system version in which Copilot is explicitly included.
    cumulative = complementarity_rows(ORIGINAL_SIX, groups, top20)
    cumulative_all = complementarity_rows(SYSTEMS, groups, top20)
    atomic_csv(output_dir / "system_complementarity.csv", list(cumulative[0]), cumulative)
    atomic_csv(
        output_dir / "system_complementarity_all_systems.csv",
        list(cumulative_all[0]), cumulative_all,
    )

    copilot_rows = available_depth_diagnostics("copilot", grouped, top20)
    atomic_csv(
        output_dir / "copilot_available_depth_diagnostics.csv",
        list(copilot_rows[0]), copilot_rows,
    )

    # 7.3 and 7.4 at primary cutoff
    adjacent_rows = []
    missed_rows = []
    covered_counts = {}
    for system in SYSTEMS:
        for qid, qgroups in grouped.items():
            ranking20 = top20.get(system, {}).get(qid, [])[:20]
            covered_counts[(system, qid)] = sum(
                rank_of_group(ranking20, g["members"]) is not None for g in qgroups
            )
    for system in SYSTEMS:
        for group in groups:
            qid = group["query_id"]
            ranking20 = top20.get(system, {}).get(qid, [])[:20]
            if rank_of_group(ranking20, group["members"]) is not None:
                continue
            ranking100 = deep.get(system, {}).get(qid)
            reason, basis = classify_failure(
                group=group, query=questions[qid], ranking20=ranking20,
                ranking100=ranking100, corpus=corpus,
                covered_groups_for_query=covered_counts[(system, qid)],
            )
            deep_rank = rank_of_group(ranking100 or [], group["members"])
            adj_for_group = []
            for member in group["members"]:
                for candidate, rank, distance in adjacent_hits(member, ranking20):
                    adjacent_rows.append({
                        "system": system, "query_id": qid, "group_id": group["group_id"],
                        "missed_core_page_id": member, "adjacent_page_id": candidate,
                        "adjacent_page_rank": rank, "pdf_page_offset": distance,
                    })
                    adj_for_group.append(candidate)
            missed_rows.append({
                "system": system, "query_id": qid, "group_id": group["group_id"],
                "facet_description": group["facet_description"],
                "member_page_ids": ";".join(group["members"]),
                "first_member_rank_at_100": "" if deep_rank is None else deep_rank,
                "adjacent_near_miss": bool(adj_for_group),
                "heuristic_primary_reason": reason,
                "taxonomy_basis": basis,
                "requires_manual_confirmation": True,
            })
    adjacent_fields = [
        "system","query_id","group_id","missed_core_page_id",
        "adjacent_page_id","adjacent_page_rank","pdf_page_offset",
    ]
    atomic_csv(output_dir / "adjacent_page_near_misses.csv", adjacent_fields, adjacent_rows)
    atomic_csv(output_dir / "missed_core_groups.csv", list(missed_rows[0]), missed_rows)
    summary_rows = []
    counts = Counter((row["system"], row["heuristic_primary_reason"]) for row in missed_rows)
    for (system, reason), count in sorted(counts.items()):
        summary_rows.append({"system": system, "heuristic_primary_reason": reason, "count": count})
    atomic_csv(output_dir / "failure_taxonomy_summary.csv", list(summary_rows[0]), summary_rows)
    return {
        "questions": len(questions), "groups": len(groups),
        "missed_rows": len(missed_rows), "adjacent_rows": len(adjacent_rows),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qrels", type=Path, default=ROOT/"benchmark/qrels/consensus_qrels_v2.csv")
    p.add_argument("--groups", type=Path, default=ROOT/"benchmark/qrels/core_groups.json")
    p.add_argument("--corpus", type=Path, default=ROOT/"benchmark/corpus_pages.jsonl")
    p.add_argument("--top20-dir", type=Path, default=ROOT/"benchmark/candidate_runs")
    p.add_argument("--depth-dir", type=Path, default=ROOT/"benchmark/candidate_runs_depth100_final")
    p.add_argument("--output-dir", type=Path, default=ROOT/"benchmark/analysis/core_diagnostics")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        counts = analyze(
            qrels_path=args.qrels, groups_path=args.groups, corpus_path=args.corpus,
            top20_dir=args.top20_dir, depth_dir=args.depth_dir, output_dir=args.output_dir,
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(counts, indent=2))
    print(f"Diagnostics: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
