from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.run_full_pipeline import (
    REQUIRED_API_KEYS,
    build_steps,
    run_pipeline,
    validate_environment,
)


EXPECTED_SCRIPTS = [
    "run_conventional_retrievers.py",
    "run_llm_retrievers.py",
    "build_candidate_pool.py",
    "run_llm_judges.py",
    "build_consensus_qrels.py",
]


def _clock():
    current = -1.0

    def tick() -> float:
        nonlocal current
        current += 1.0
        return current

    return tick


def test_live_steps_have_required_order_and_no_mock_arguments() -> None:
    steps = build_steps(mock=False)

    assert [step.script.name for step in steps] == EXPECTED_SCRIPTS
    assert all(step.arguments == () for step in steps)
    assert steps[0].command("test-python")[0] == "test-python"


def test_mock_steps_isolate_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "isolated"
    steps = build_steps(mock=True, mock_output_root=output_root)

    assert [step.script.name for step in steps] == EXPECTED_SCRIPTS
    assert "--mock" in steps[1].arguments
    assert "--mock" in steps[3].arguments
    assert all(
        str(output_root.resolve()) in " ".join(step.arguments)
        for step in steps
    )


def test_environment_validation_allows_complete_environment() -> None:
    environment = {name: "configured" for name in REQUIRED_API_KEYS}

    assert validate_environment(
        environment=environment,
        strict=True,
        mock=False,
    )


def test_environment_validation_warns_for_non_strict_mock(capsys) -> None:
    assert validate_environment(environment={}, strict=False, mock=True)

    captured = capsys.readouterr()
    assert "Missing API environment variables" in captured.err
    assert "Mock mode selected" in captured.out


def test_strict_environment_validation_preempts_mock(capsys) -> None:
    assert not validate_environment(environment={}, strict=True, mock=True)

    captured = capsys.readouterr()
    assert "pipeline halted" in captured.err


def test_pipeline_halts_immediately_after_failed_step() -> None:
    calls: list[tuple[list[str], dict]] = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=9 if len(calls) == 2 else 0)

    result = run_pipeline(
        build_steps(mock=False),
        environment={"PIPELINE_TEST": "1"},
        runner=runner,
        clock=_clock(),
        python_executable="test-python",
    )

    assert result == 9
    assert [Path(call[0][1]).name for call in calls] == EXPECTED_SCRIPTS[:2]
    assert all(call[0][0] == "test-python" for call in calls)
    assert all(call[1]["check"] is False for call in calls)
    assert all(call[1]["env"]["PIPELINE_TEST"] == "1" for call in calls)


def test_pipeline_runs_every_step_on_success() -> None:
    commands: list[list[str]] = []

    def runner(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    result = run_pipeline(
        build_steps(mock=False),
        environment={},
        runner=runner,
        clock=_clock(),
        python_executable="test-python",
    )

    assert result == 0
    assert [Path(command[1]).name for command in commands] == EXPECTED_SCRIPTS

def test_resume_flag_is_forwarded_to_retrieval_steps(tmp_path: Path) -> None:
    live_steps = build_steps(mock=False, resume=True)
    assert live_steps[0].arguments == ("--resume",)
    assert live_steps[1].arguments == ("--resume",)
    assert live_steps[3].arguments == ("--resume",)
    assert all(
        "--resume" not in step.arguments
        for index, step in enumerate(live_steps)
        if index not in (0, 1, 3)
    )

    mock_steps = build_steps(
        mock=True,
        resume=True,
        mock_output_root=tmp_path / "mock",
    )
    assert "--resume" in mock_steps[0].arguments
    assert "--resume" in mock_steps[1].arguments
    assert "--resume" in mock_steps[3].arguments
