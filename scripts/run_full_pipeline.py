"""Run the complete CDIRP benchmark-generation pipeline.

Live run:
    python scripts/run_full_pipeline.py --strict-env

Safe end-to-end smoke run with isolated outputs:
    python scripts/run_full_pipeline.py --mock
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOCK_ROOT = REPOSITORY_ROOT / "benchmark" / "mock_pipeline"
REQUIRED_API_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
)


@dataclass(frozen=True)
class PipelineStep:
    name: str
    script: Path
    arguments: tuple[str, ...] = ()

    def command(self, python_executable: str = sys.executable) -> list[str]:
        return [
            python_executable,
            str(self.script),
            *self.arguments,
        ]


def build_steps(
    *,
    mock: bool,
    resume: bool = False,
    mock_output_root: Path = DEFAULT_MOCK_ROOT,
) -> list[PipelineStep]:
    """Construct the exact B-through-E dependency order."""

    scripts = REPOSITORY_ROOT / "scripts"
    resume_args = ("--resume",) if resume else ()
    if not mock:
        return [
            PipelineStep(
                "Milestone B: Conventional Retrievers",
                scripts / "run_conventional_retrievers.py",
                resume_args,
            ),
            PipelineStep(
                "Milestone B: Frontier LLM Retrievers",
                scripts / "run_llm_retrievers.py",
                resume_args,
            ),
            PipelineStep(
                "Milestone C: Candidate Pool Construction",
                scripts / "build_candidate_pool.py",
            ),
            PipelineStep(
                "Milestone D: Independent LLM Judges",
                scripts / "run_llm_judges.py",
                resume_args,
            ),
            PipelineStep(
                "Milestone E: Consensus Qrels",
                scripts / "build_consensus_qrels.py",
            ),
        ]

    root = mock_output_root.resolve()
    run_dir = root / "candidate_runs"
    pool = root / "pools" / "candidate_pool.jsonl"
    judgments = root / "judgments"
    return [
        PipelineStep(
            "Milestone B: Conventional Retrievers",
            scripts / "run_conventional_retrievers.py",
            (*resume_args, "--output-dir", str(run_dir)),
        ),
        PipelineStep(
            "Milestone B: Frontier LLM Retrievers [MOCK]",
            scripts / "run_llm_retrievers.py",
            (
                "--mock",
                *resume_args,
                "--output-dir",
                str(run_dir),
                "--trace-dir",
                str(root / "llm_traces"),
            ),
        ),
        PipelineStep(
            "Milestone C: Candidate Pool Construction",
            scripts / "build_candidate_pool.py",
            (
                "--run-dir",
                str(run_dir),
                "--pool-output",
                str(pool),
                "--overlap-output",
                str(root / "analysis" / "pool_overlap.csv"),
                "--stats-output",
                str(root / "analysis" / "pool_stats.json"),
            ),
        ),
        PipelineStep(
            "Milestone D: Independent LLM Judges [MOCK]",
            scripts / "run_llm_judges.py",
            (
                "--mock",
                *resume_args,
                "--pool",
                str(pool),
                "--output-dir",
                str(judgments),
            ),
        ),
        PipelineStep(
            "Milestone E: Consensus Qrels",
            scripts / "build_consensus_qrels.py",
            (
                "--pool",
                str(pool),
                "--judgment-dir",
                str(judgments),
                "--full-output",
                str(root / "qrels" / "consensus_qrels_full.csv"),
                "--trec-output",
                str(root / "qrels" / "consensus_qrels.tsv"),
            ),
        ),
    ]


def missing_api_keys(environment: Mapping[str, str]) -> list[str]:
    """Return missing or blank required credentials without exposing values."""

    return [
        name
        for name in REQUIRED_API_KEYS
        if not environment.get(name, "").strip()
    ]


def validate_environment(
    *,
    environment: Mapping[str, str],
    strict: bool,
    mock: bool,
) -> bool:
    """Report key presence and enforce strict mode before subprocess execution."""

    missing = missing_api_keys(environment)
    if not missing:
        print("API environment: all required keys are configured.", flush=True)
        return True

    print(
        "WARNING: Missing API environment variables: " + ", ".join(missing),
        file=sys.stderr,
        flush=True,
    )
    if strict:
        print(
            "ERROR: Strict environment validation is enabled; pipeline halted.",
            file=sys.stderr,
            flush=True,
        )
        return False
    if mock:
        print(
            "Mock mode selected; continuing without live API credentials.",
            flush=True,
        )
        return True

    print(
        "WARNING: Continuing in non-strict live mode. An LLM step will likely "
        "fail unless credentials are supplied by another provider mechanism.",
        file=sys.stderr,
        flush=True,
    )
    return True


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def run_pipeline(
    steps: Sequence[PipelineStep],
    *,
    environment: Mapping[str, str],
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    clock: Callable[[], float] = time.perf_counter,
    python_executable: str = sys.executable,
) -> int:
    """Execute steps sequentially and stop immediately on the first failure."""

    total_start = clock()
    for index, step in enumerate(steps, start=1):
        separator = "=" * 72
        print(f"\n{separator}", flush=True)
        print(
            f"=== Starting Step {index}/{len(steps)}: {step.name} ===",
            flush=True,
        )
        print(separator, flush=True)
        step_start = clock()
        command = step.command(python_executable)
        try:
            result = runner(
                command,
                cwd=REPOSITORY_ROOT,
                env=dict(environment),
                check=False,
            )
        except KeyboardInterrupt:
            elapsed = clock() - step_start
            print(
                f"\nINTERRUPTED: {step.name} after {format_seconds(elapsed)}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"Total elapsed: {format_seconds(clock() - total_start)}",
                flush=True,
            )
            return 130
        except OSError as error:
            elapsed = clock() - step_start
            print(
                f"ERROR: Could not start {step.name} after "
                f"{format_seconds(elapsed)}: {error}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"Total elapsed: {format_seconds(clock() - total_start)}",
                flush=True,
            )
            return 1

        elapsed = clock() - step_start
        if result.returncode != 0:
            print(
                f"ERROR: {step.name} failed with exit code "
                f"{result.returncode} after {format_seconds(elapsed)}.",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"Pipeline halted. Total elapsed: "
                f"{format_seconds(clock() - total_start)}",
                flush=True,
            )
            return result.returncode or 1
        print(
            f"=== Completed: {step.name} in {format_seconds(elapsed)} ===",
            flush=True,
        )

    total_elapsed = clock() - total_start
    print("\n" + "=" * 72, flush=True)
    print(
        f"=== Full pipeline completed successfully in "
        f"{format_seconds(total_elapsed)} ===",
        flush=True,
    )
    print("=" * 72, flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete Milestone B-through-E benchmark pipeline."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Use mock LLM calls and isolate every output under "
            "benchmark/mock_pipeline."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume validated retrieval and judging work without repeating paid calls.",
    )
    parser.add_argument(
        "--strict-env",
        action="store_true",
        help="Halt before execution when any required API key is missing.",
    )
    parser.add_argument(
        "--mock-output-root",
        type=Path,
        default=DEFAULT_MOCK_ROOT,
        help="Isolated output root used with --mock.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(REPOSITORY_ROOT / ".env.local", override=False)
    environment = os.environ.copy()
    if not validate_environment(
        environment=environment,
        strict=args.strict_env,
        mock=args.mock,
    ):
        return 2

    steps = build_steps(
        mock=args.mock,
        resume=args.resume,
        mock_output_root=args.mock_output_root,
    )
    return run_pipeline(steps, environment=environment)


if __name__ == "__main__":
    raise SystemExit(main())
