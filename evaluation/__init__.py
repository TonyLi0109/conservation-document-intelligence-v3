"""V3 evaluation entry point; UI runs fixture work in an isolated process."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from config import SETTINGS, V3_ROOT
from evaluation.reporting import REPORT_VERSION, write_reports


def _read_completed_report(path):
    """Accept only a complete report emitted by this request's subprocess."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("report_version") != REPORT_VERSION:
            raise ValueError("Invalid report version")
        cases = report.get("cases")
        if (not isinstance(cases, list) or not isinstance(report.get("summary"), dict)
                or not isinstance(report.get("generated_at"), str)
                or any(type(report.get(key)) is not int for key in ("case_count", "passed_count", "failed_count"))
                or any(not isinstance(row, dict) or type(row.get("passed")) is not bool for row in cases)):
            raise ValueError("Incomplete evaluation report")
        if (report["case_count"] != len(cases)
                or report["passed_count"] != sum(row["passed"] for row in cases)
                or report["failed_count"] != sum(not row["passed"] for row in cases)):
            raise ValueError("Inconsistent evaluation report counts")
        return report
    except (OSError, ValueError, TypeError) as error:
        # Report/provider contents are not safe diagnostics for a shared UI.
        raise RuntimeError("Evaluation process produced an invalid report") from None


def run_evaluation(store, *, cases_path=None, top_k=SETTINGS.retrieval.top_k,
                   include_generation=False, model=None, output_dir=None):
    """Run offline by default, without fixture patches touching live chat users."""
    # A caller-selected output directory may contain yesterday's valid report.
    # Always execute into a fresh directory, then publish only a completed report.
    with tempfile.TemporaryDirectory(prefix="v3-evaluation-") as directory:
        command = [sys.executable, "-X", "utf8", "-m", "evaluation.run", "--corpus", str(store.database_path),
                   "--output-dir", str(directory), "--top-k", str(top_k)]
        if cases_path is not None:
            command += ["--retrieval-cases", str(cases_path)]
        if include_generation:
            command += ["--live", "--model", model or SETTINGS.models.llm_model]
        result = subprocess.run(command, cwd=V3_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=600)
        report_path = Path(directory) / "latest.json"
        if result.returncode not in (0, 1) or not report_path.exists():
            # Do not propagate provider stderr/credentials into the shared web UI.
            raise RuntimeError(f"Evaluation process failed (exit {result.returncode})")
        report = _read_completed_report(report_path)
        if output_dir is not None:
            write_reports(report, output_dir)
        return report
