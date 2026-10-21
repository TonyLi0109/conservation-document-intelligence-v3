"""Validate and export the experiment using the converted V3/PDF corpus."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_pdf_benchmark as run
from scripts.report_revised_benchmark import make_report
from scripts.validate_pdf_benchmark import validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=run.DEFAULT)
    root = parser.parse_args().root.resolve()
    complete = make_report(root, runner=run)
    validate(root)
    raise SystemExit(0 if complete else 1)
