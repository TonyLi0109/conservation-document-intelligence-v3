"""Inspectable reports and fingerprint-checked deterministic baselines."""

import json
from pathlib import Path

REPORT_VERSION = 1
RANK_METRICS = ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "hit_at_k")
QUALITY_METRICS = ("exact_span_hit_rate", "quantitative_evidence_hit_rate", "exact_document_hit_rate",
                   "required_document_recall", "compatibility_pass_rate")
LOWER_BETTER_METRICS = ("duplicate_evidence_rate", "wrong_topic_retrieval_rate")


def summarize(rows):
    summary = {}
    for category in sorted({row["category"] for row in rows}):
        members = [row for row in rows if row["category"] == category]
        # Distinct modes avoid mixing synthetic fixture quality with corpus/model quality.
        for mode in sorted({row["mode"] for row in members}):
            group = [row for row in members if row["mode"] == mode]
            metrics = {}
            keys = set().union(*(row["metrics"].keys() for row in group))
            for key in sorted(keys):
                values = [row["metrics"].get(key) for row in group]
                numeric = [value for value in values if type(value) in (int, float)]
                if numeric:
                    metrics[key] = sum(numeric) / len(numeric)
            summary[f"{category}/{mode}"] = {
                "case_count": len(group), "passed": sum(row["passed"] is True for row in group),
                "metrics_mean": metrics,
            }
    return summary


def baseline_from(report):
    return {"report_version": REPORT_VERSION, "configuration": report["configuration"],
            "fingerprints": report["fingerprints"],
            "cases": [{"case_id": row["case_id"], "category": row["category"], "mode": row["mode"],
                       "passed": row["passed"], "metrics": row["metrics"]} for row in report["cases"]]}


def compare_baseline(report, baseline):
    if baseline.get("report_version") != REPORT_VERSION:
        raise ValueError("Unsupported baseline report version")
    compatible = (report["fingerprints"] == baseline.get("fingerprints")
                  and report["configuration"] == baseline.get("configuration"))
    result = {"compatible": compatible, "regressions": [], "warnings": [], "metric_deltas": []}
    if not compatible:
        result["warnings"].append("Dataset, corpus or execution settings changed; baseline comparison skipped.")
        return result
    old_rows = {(row["category"], row["mode"], row["case_id"]): row for row in baseline["cases"]}
    current_keys = {(row["category"], row["mode"], row["case_id"]) for row in report["cases"]}
    for category, mode, case_id in sorted(old_rows.keys() - current_keys):
        key = "warnings" if mode.startswith("live") else "regressions"
        result[key].append(f"{category}/{mode}/{case_id}: baseline case missing from current run")
    for row in report["cases"]:
        identifier = f"{row['category']}/{row['mode']}/{row['case_id']}"
        destination = "warnings" if row["mode"].startswith("live") else "regressions"
        old = old_rows.get((row["category"], row["mode"], row["case_id"]))
        if old is None:
            result["warnings"].append(f"No baseline case: {identifier}")
            continue
        if old["passed"] and not row["passed"]:
            result[destination].append(f"{identifier}: invariant changed from pass to fail")
        for metric in RANK_METRICS + QUALITY_METRICS + LOWER_BETTER_METRICS:
            before, after = old["metrics"].get(metric), row["metrics"].get(metric)
            if type(before) in (int, float) and type(after) not in (int, float):
                result[destination].append(f"{identifier}: previously measured {metric} is missing or nonnumeric")
            elif type(before) in (int, float) and type(after) in (int, float):
                delta = after - before
                result["metric_deltas"].append({"case_id": row["case_id"], "category": row["category"],
                                                "mode": row["mode"], "metric": metric, "delta": delta})
                adverse = delta if metric in LOWER_BETTER_METRICS else -delta
                if adverse > 1e-9:
                    direction = "increased" if metric in LOWER_BETTER_METRICS else "decreased"
                    result[destination].append(f"{identifier}: {metric} {direction} by {adverse:.4f}")
    return result


def render_summary(report):
    lines = ["# V3 evaluation", "", f"Generated: {report['generated_at']}",
             f"Cases: {report['case_count']} | Passed: {report['passed_count']} | Failed: {report['failed_count']}",
             "", "Modes are reported separately. Fixture passes measure pipeline invariants, not live-model quality.",
             "Corpus relevance labels are partial; precision/nDCG are judged-label proxies, not exhaustive relevance scores.", ""]
    for name, group in report["summary"].items():
        lines += [f"## {name}", "", f"{group['passed']}/{group['case_count']} checks passed.", ""]
        if "adversarial" in name:
            lines.append("Negative fixtures: pass means the expected violation was detected, not that citations were valid.")
        else:
            visible = RANK_METRICS if name.startswith("retrieval/") else (
                "citation_validity_rate", "citation_coverage", "claims_numeric_support_rate",
                "raw_numeric_support_rate", "raw_provenance_validity_rate", "raw_grounded_proxy_rate",
                "overview_characters", "important_fact_count", "evidence_count", "invalid_citation_count",
                "missing_source_url_count", "turn_count", "refresh_checks",
                "correct_current_document_rate", "historical_query_accuracy", "version_relationship_accuracy",
                "unsupported_supersession_claim_count", "temporal_intent_accuracy",
                "latest_report_accuracy", "lifecycle_metadata_accuracy", "inapplicable_document_preference_errors",
            )
            if name.startswith("retrieval_quality/"):
                visible = RANK_METRICS + QUALITY_METRICS + LOWER_BETTER_METRICS + visible
            for metric in visible:
                if metric in group["metrics_mean"]:
                    lines.append(f"- {metric}: {group['metrics_mean'][metric]:.4f}")
        lines.append("")
    for row in report["cases"]:
        if not row["passed"]:
            lines.append(f"- FAIL {row['case_id']}: " + "; ".join(row["failure_reasons"]))
        for warning in row.get("warnings", []):
            lines.append(f"- NOTE {row['case_id']}: {warning}")
    comparison = report.get("baseline_comparison")
    if comparison:
        lines += ["", "## Baseline comparison", "", f"Compatible: {comparison['compatible']}"]
        lines += [f"- {item}" for item in comparison["regressions"] + comparison["warnings"]]
    lines += ["", "Exact spans, source ownership and numeric-token checks do not establish semantic entailment.",
              "Semantic support is not assessed unless an explicitly selected judge is supplied.", ""]
    return "\n".join(lines)


def write_reports(report, output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (path / "latest.md").write_text(render_summary(report), encoding="utf-8")
