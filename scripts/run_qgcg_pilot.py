r"""QGCG pilot and stress checks with durable audit logging.

Default execution is AUTOMATIC through logged-in official account CLIs:
Codex/ChatGPT, Claude/claude.ai, and Gemini/Antigravity. No developer API keys
are used by default. API-key routes and Gemini extra-credit fallback are disabled.
Explicit --gemini-transport api authorizes ONLY Pass 3 through the Google API;
GPT/Claude remain on subscription CLIs. GEMINI_API_KEY/GOOGLE_API_KEY or the
existing new-V3/.env.local credential is read privately, never archived.

    python scripts/run_qgcg_pilot.py --gemini-transport api --query-ids Q1 Q6 Q7 \
        --output-root benchmark/qgcg_pilot/stress_checks

    python scripts/run_qgcg_pilot.py --check-accounts
    python scripts/run_qgcg_pilot.py --gemini-model gemini-3.1-pro-low

The Gemini account slug must be explicit and available to the signed-in account;
the API's gemini-3.1-pro-preview name is not silently substituted. Low is the
existing benchmark's reasoning setting; High can be selected explicitly.
Raw CLI output is captured before decoding; backend revisions remain unknown.

The earlier clipboard workflow is available ONLY with --transport manual:

    python -m pip install -r scripts/requirements-qgcg-pilot.txt
    python scripts/run_qgcg_pilot.py --transport manual

Defaults resolve under new-V3: benchmark/questions.csv, the frozen canonical
benchmark/fair_v2/corpus.jsonl, and benchmark/qgcg_pilot/{E1,E2}/. Questions are
read unchanged. Use --prepare-only to snapshot inputs without clipboard/input.
Existing output directories are never overwritten; use a new --output-root.
Use --resume-failed after a finished run to retry failed queries from frozen
inputs, preserving successful queries and all previous audit records.

In manual mode, the exact system instructions + input JSON + output schema are
saved and copied with pyperclip. Paste into a fresh web chat with the displayed
model, then paste its raw response here. Finish with EOF alone on a new line.
Alternatively enter :clipboard, :file PATH, or :file for the displayed inbox.
Use :copy to copy the prompt again; :abort or Ctrl+C stops with audit preserved.
Invalid responses are logged verbatim and retried until corrected or aborted.

In manual mode no API keys, SDK calls, browser automation, or paid-API fallback are used.
Select GPT-5.6 in ChatGPT Plus, Claude-Sonnet-5 in Claude Pro, and Gemini-3.1
in Gemini Advanced. Model selection is human-attested, not provider-verified;
unknown backend revisions remain null. A web paste is a single user message,
not a privileged API system message. Both limitations are recorded in manifests.

62/127 are global query-page membership counts, not per-question page counts.
For the frozen v2 labels, E1 has Q2=4/Q3=22 pages; E2 has Q2=13/Q3=41 pages.
Only memberships for the current query are sent to its models.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import io
import json
import os
from pathlib import Path
import platform
import re
import sqlite3
import sys
import time
from typing import Any, Callable
from uuid import uuid4

import pydantic
from pydantic import ValidationError

# Works both as python scripts/run_qgcg_pilot.py and as an imported module.
if __package__:
    from . import build_qgcg_groups as qgcg
else:
    import build_qgcg_groups as qgcg


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = WORKSPACE / "new-V3" if (WORKSPACE / "new-V3").is_dir() else WORKSPACE
QUERY_IDS = ("Q2", "Q3")
ARMS = {"E1": "core", "E2": "relevant"}
EXPECTED_MEMBERSHIPS = {"core": 62, "relevant": 127}
PASS_NUMBERS = {"model_a": 1, "model_b": 2, "model_c": 3}
DEFAULT_MODELS = {
    "model_a": "gpt-5.6-sol",
    "model_b": "claude-sonnet-5",
    "model_c": "gemini-3.1-pro-preview",
}
PROMPTS = {
    "model_a": qgcg.MODEL_A_SYSTEM_PROMPT,
    "model_b": qgcg.MODEL_B_SYSTEM_PROMPT,
    "model_c": qgcg.MODEL_C_SYSTEM_PROMPT,
    "model_b_repair": qgcg.MODEL_B_REPAIR_SYSTEM_PROMPT,
}
RUNNER_VERSION = "qgcg-pilot-1.4.1-gemini-api-option"
WEB_MODELS = {
    "model_a": ("ChatGPT Plus", "GPT-5.6"),
    "model_b": ("Claude Pro", "Claude-Sonnet-5"),
    "model_c": ("Gemini Advanced", "Gemini-3.1"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def write_durable(path: Path, data: bytes, *, replace: bool = False) -> None:
    """Flush before continuing; manifest/output replacements are atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    try:
        write_durable(temporary, data)
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as error:
                # OneDrive/Windows scanners can briefly open the destination
                # without FILE_SHARE_DELETE. Keep the same flushed bytes and
                # retry the atomic rename, never truncate the existing file.
                if getattr(error, "winerror", None) not in {5, 32, 33} or attempt == 9:
                    raise
                time.sleep(min(0.1 * 2 ** attempt, 1.0))
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class ProviderRequest:
    call_id: str
    attempt: int
    query_id: str
    arm: str
    role: str
    model: str
    system_prompt: str
    user_prompt: str  # Exact serialized QGCG payload; do not reformat it.
    json_schema: dict[str, Any]
    seed: int | None
    timeout_seconds: float
    prompt_path: Path | None = None
    response_path: Path | None = None
    event_logger: Callable[..., None] | None = field(default=None, repr=False)
    raw_logger: Callable[[str, str], None] | None = field(default=None, repr=False)
    artifact_writer: Callable[[str, bytes], None] | None = field(default=None, repr=False)
    validation_feedback: str | None = None


@dataclass(frozen=True)
class RawProviderResponse:
    raw_text: str
    stderr: str | None = None
    exit_code: int = 0
    human_model_label: str | None = None
    capture_method: str | None = None
    capture_source: str | None = None
    final_text: str | None = None


@dataclass(frozen=True)
class DecodedResponse:
    text: str  # Exact model text, not json.dumps(parsed_model_output).
    reported_model: str | None
    backend_revision: str | None = None
    provider_request_id: str | None = None
    finish_reason: str | None = None
    identity_source: str = "provider_report"
    human_model_label: str | None = None


class ProviderFailure(RuntimeError):
    """One unsuccessful attempt. Attach any partial stdout before raising."""

    def __init__(self, message: str, *, retryable: bool = False,
                 raw_response: RawProviderResponse | None = None,
                 retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.raw_response = raw_response
        self.retry_after_seconds = retry_after_seconds


class ManualRunAborted(KeyboardInterrupt):
    """Operator cancellation/closed stdin stops the whole run, not just one query."""

    def __init__(self, message: str, partial: RawProviderResponse | None = None):
        super().__init__(message)
        self.partial = partial


def assemble_hitl_prompt(system_prompt: str, user_prompt: str, json_schema: dict) -> str:
    # Preserve each supplied component exactly; no truncation or JSON extraction.
    return ("=== SYSTEM INSTRUCTIONS ===\n" + system_prompt
            + "\n=== USER INPUT JSON (DATA) ===\n" + user_prompt
            + "\n=== REQUIRED OUTPUT JSON SCHEMA ===\n"
            + json_bytes(json_schema).decode("utf-8"))


def _pyperclip():
    try:
        import pyperclip
    except ImportError as error:
        raise RuntimeError("Install clipboard support: python -m pip install -r scripts/requirements-qgcg-pilot.txt") from error
    return pyperclip


def _matches_web_model(role: str, label: str) -> bool:
    normalize = lambda value: re.sub(r"[^a-z0-9]", "", value.casefold())
    labels = {WEB_MODELS[role][1], DEFAULT_MODELS[role]}
    if role == "model_c":
        labels.add("Gemini 3.1 Pro")
    return normalize(label) in {normalize(value) for value in labels}


class HITLClipboardTransport:
    """Terminal/clipboard I/O only; validation belongs to AuditedLLMAdapter."""

    def __init__(self, *, input_fn=None, print_fn=None, copy_fn=None, paste_fn=None):
        self.input = input if input_fn is None else input_fn
        self.print = print if print_fn is None else print_fn
        self.copy = (lambda text: _pyperclip().copy(text)) if copy_fn is None else copy_fn
        self.paste = (lambda: _pyperclip().paste()) if paste_fn is None else paste_fn
        self.model_labels: dict[str, str] = {}

    @staticmethod
    def _event(request: ProviderRequest, event: str, **details):
        if request.event_logger:
            request.event_logger(event, **details)

    def _copy(self, request: ProviderRequest, prompt: str) -> None:
        try:
            self.copy(prompt)
        except Exception as error:
            self._event(request, "clipboard_error", error=error)
            self.print(f"Clipboard unavailable: {error}\nCopy the complete saved prompt from: {request.prompt_path}")
        else:
            self._event(request, "clipboard_copied", characters=len(prompt), sha256=sha256(prompt.encode("utf-8")))
            self.print(f"Prompt copied ({len(prompt):,} characters). Saved at: {request.prompt_path}")

    def request(self, request: ProviderRequest) -> RawProviderResponse:
        service, model = WEB_MODELS[request.role]
        prompt = assemble_hitl_prompt(request.system_prompt, request.user_prompt, request.json_schema)
        self.print(f"\n{'=' * 72}\n{request.arm} / {request.query_id} / PASS {PASS_NUMBERS[request.role]}"
                   f" / submission {request.attempt}\nUSE {model} IN {service}\n{'=' * 72}")
        if request.attempt == 1:
            self._copy(request, prompt)
            self.print("Open a fresh web chat, select the indicated model, and paste the entire prompt."
                       " Use only the supplied sources; disable browsing/tools.")
        else:
            # Never clobber a corrected response the operator just copied.
            self.print("Paste the corrected response for this SAME prompt. Clipboard has not been overwritten.")
        lines: list[str] = []
        try:
            while request.role not in self.model_labels:
                label = self.input("Exact model label displayed in the web UI (once per model; :abort to stop): ")
                if label == ":abort":
                    raise ManualRunAborted("Operator cancelled model selection")
                if not _matches_web_model(request.role, label):
                    self._event(request, "manual_model_selection_error", entered_label=label, expected=model)
                    self.print(f"Expected {model}. Select it first; if unavailable, use :abort. No model fallback.")
                    continue
                self.model_labels[request.role] = label
                self._event(request, "manual_model_attestation", web_model_label=label,
                            provider_verified=False, backend_revision=None)
            self.print(f"Human-selected model: {self.model_labels[request.role]} (backend version unverified).")
            self.print("Paste raw JSON, then enter EOF alone on a new line."
                       "\nOr :clipboard | :file PATH | :file (inbox) | :copy | :abort")
            self.print(f"Response inbox: {request.response_path}")
            while True:
                line = self.input("")
                if line == ":abort":
                    raise ManualRunAborted("Operator cancelled response entry")
                if line == "EOF":
                    return RawProviderResponse("\n".join(lines), human_model_label=self.model_labels[request.role],
                                               capture_method="terminal_lines_joined_with_LF")
                if not lines and (line in {":copy", ":clipboard", ":file"} or line.startswith(":file ")):
                    if line == ":copy":
                        self._copy(request, prompt)
                        continue
                    try:
                        source = None
                        if line == ":clipboard":
                            text = self.paste()
                            if text == prompt:
                                raise ValueError("Clipboard still contains the prompt; copy the model's response first")
                            method = "clipboard"
                        else:
                            filename = line[5:].strip().strip('"')
                            path = Path(filename).expanduser() if filename else request.response_path
                            if path is None:
                                raise ValueError("No response file supplied")
                            text = path.read_bytes().decode("utf-8")
                            source, method = str(path.resolve()), "utf8_file"
                    except Exception as error:
                        self._event(request, "manual_input_error", error=error)
                        self.print(f"Cannot read response: {error}. Try again.")
                        continue
                    return RawProviderResponse(text, human_model_label=self.model_labels[request.role],
                                               capture_method=method, capture_source=source)
                lines.append(line)
        except (EOFError, KeyboardInterrupt) as error:
            partial = RawProviderResponse("\n".join(lines), capture_method="interrupted_terminal") if lines else None
            raise ManualRunAborted(str(error) or "Terminal input closed/interrupted", partial) from error

    def validation_error(self, error: Exception) -> None:
        self.print(f"\nVALIDATION FAILED — original response saved in raw_responses.jsonl.\n{error}"
                   "\nObtain/copy a corrected JSON response, then paste it below and end with EOF."
                   "\nNo automatic model call is made. Use :abort to stop.")


_DEFAULT_HITL = HITLClipboardTransport()


def request_gpt_subscription(request: ProviderRequest) -> RawProviderResponse:
    return _DEFAULT_HITL.request(request)


def request_claude_subscription(request: ProviderRequest) -> RawProviderResponse:
    return _DEFAULT_HITL.request(request)


def request_gemini_subscription(request: ProviderRequest) -> RawProviderResponse:
    return _DEFAULT_HITL.request(request)


def decode_subscription_response(response: RawProviderResponse) -> DecodedResponse:
    # A web paste has no provider envelope. Never invent verified model metadata.
    return DecodedResponse(response.raw_text, reported_model=None,
                           identity_source="human_attested_web_ui", human_model_label=response.human_model_label)


@dataclass(frozen=True)
class ProviderSpec:
    requested_model: str
    request: Callable[[ProviderRequest], RawProviderResponse]
    decode: Callable[[RawProviderResponse], DecodedResponse]
    transport_name: str
    transport_version: str
    billing_mode: str = "subscription"
    seed_supported: bool = False
    settings: dict[str, Any] = field(default_factory=dict)
    implementation_files: tuple[Path, ...] = ()
    manual: bool = False
    validation_feedback: Callable[[Exception], None] | None = None
    retry_invalid_output: bool = False


def hitl_providers(transport: HITLClipboardTransport | None = None) -> dict[str, ProviderSpec]:
    transport = HITLClipboardTransport() if transport is None else transport
    return {
        role: ProviderSpec(DEFAULT_MODELS[role], transport.request, decode_subscription_response,
                           "manual_web_subscription_clipboard", "hitl-1.0.0", manual=True,
                           validation_feedback=transport.validation_error,
                           settings={"service": WEB_MODELS[role][0], "target_web_model": WEB_MODELS[role][1],
                                     "response_delimiter": "EOF", "model_identity": "human_attested_unverified",
                                     "prompt_delivery": "single_web_user_message", "api_fallback": False})
        for role in PASS_NUMBERS
    }


@dataclass
class PilotInputs:
    questions: dict[str, str]
    pools: dict[str, dict[str, list[str]]]
    pages: dict[str, str]
    source_files: dict[str, tuple[Path, bytes]]
    source_provenance: list[dict[str, Any]]
    membership_counts: dict[str, int]
    unique_page_counts: dict[str, int]


def _one_field(record: dict, names: tuple[str, ...], description: str):
    values = [record[name] for name in names if name in record]
    if not values or any(value != values[0] for value in values):
        raise ValueError(f"Missing or conflicting {description} fields: {names}")
    return values[0]


def validate_query_ids(query_ids):
    ids = tuple(query_ids)
    if not ids or len(set(ids)) != len(ids) or any(
            not isinstance(qid, str) or not re.fullmatch(r"Q[1-9][0-9]*", qid) for qid in ids):
        raise ValueError("query_ids must be distinct question IDs such as Q1 Q6 Q7")
    return ids


def load_questions(path: Path, query_ids=QUERY_IDS) -> tuple[dict[str, str], bytes]:
    query_ids = validate_query_ids(query_ids)
    data = path.read_bytes()  # Preserve original bytes as well as decoded questions.
    text = data.decode("utf-8-sig")
    if path.suffix.lower() == ".csv":
        records = list(csv.DictReader(io.StringIO(text, newline="")))
    else:
        value = json.loads(text)
        if isinstance(value, dict) and "questions" in value:
            records = value["questions"]
        elif isinstance(value, dict):
            records = [{"query_id": key, "question": value} for key, value in value.items()]
        else:
            records = value
    if not isinstance(records, list):
        raise ValueError("Questions must be a JSON array, ID/text dictionary, or CSV")
    questions: dict[str, str] = {}
    for record in records:
        query_id = _one_field(record, ("query_id", "question_id"), "question ID")
        if query_id not in query_ids:
            continue
        question = _one_field(record, ("question", "query"), "question text")
        if not isinstance(query_id, str) or query_id in questions:
            raise ValueError(f"Invalid or duplicate question ID: {query_id}")
        qgcg.QUESTION_ADAPTER.validate_python(question)
        questions[query_id] = question
    missing = set(query_ids) - questions.keys()
    if missing:
        raise ValueError(f"Question file missing {sorted(missing)}")
    return {qid: questions[qid] for qid in query_ids}, data


def load_pools(path: Path, query_ids=QUERY_IDS):
    """Use explicit v2 consensus flags, never infer labels from a different file."""
    query_ids = validate_query_ids(query_ids)
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    if not {"query_id", "page_id", "core", "relevant"} <= set(reader.fieldnames or []):
        raise ValueError("Pool CSV requires query_id,page_id,core,relevant")
    selected = {flag: set() for flag in ARMS.values()}
    seen = set()
    for row in reader:
        key = (row["query_id"], row["page_id"])
        if not all(key) or key in seen:
            raise ValueError(f"Empty or duplicate query/page membership: {key}")
        seen.add(key)
        for flag in selected:
            value = row[flag].strip().lower()
            if value not in {"true", "false", "1", "0"}:
                raise ValueError(f"Invalid consensus flag {flag}={value!r}")
            if value in {"true", "1"}:
                selected[flag].add(key)
    counts = {flag: len(pairs) for flag, pairs in selected.items()}
    if counts != EXPECTED_MEMBERSHIPS:
        raise ValueError(f"Wrong consensus pool: expected {EXPECTED_MEMBERSHIPS}, got {counts}")
    if not selected["core"] <= selected["relevant"]:
        raise ValueError("E1 core memberships must be a subset of E2 relevant memberships")
    pools = {arm: {qid: sorted(page for query, page in selected[flag] if query == qid)
                   for qid in query_ids} for arm, flag in ARMS.items()}
    if any(not ids for pool in pools.values() for ids in pool.values()):
        raise ValueError("Each arm/query must have at least one selected page")
    unique = {flag: len({page for _, page in pairs}) for flag, pairs in selected.items()}
    return pools, data, counts, unique


def load_canonical_pages(path: Path, page_ids: set[str]):
    """Read frozen page texts; never reconstruct pages from overlapping chunks.

    Supported: JSONL/JSON page records (page_id + text/page_text), a directory of
    <page_id>.txt, or SQLite pages/corpus_pages with page_id + text/page_text.
    SQLite chunks in knowledge_artifacts are deliberately not treated as pages.
    """
    pages: dict[str, str] = {}
    sources: dict[str, tuple[Path, bytes]] = {}
    provenance: list[dict[str, Any]] = []

    def add(page_id, text):
        if not isinstance(page_id, str) or not isinstance(text, str):
            raise ValueError("Canonical page IDs and texts must be strings")
        if page_id not in page_ids:
            return
        if page_id in pages:
            raise ValueError(f"Duplicate canonical page: {page_id}")
        pages[page_id] = text

    if path.is_dir():
        for index, page_id in enumerate(sorted(page_ids)):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", page_id):
                raise ValueError(f"Unsafe page filename: {page_id}")
            source = (path / f"{page_id}.txt").resolve()
            source.relative_to(path.resolve())
            data = source.read_bytes()
            add(page_id, data.decode("utf-8"))
            sources[f"corpus/page_{index:04d}.txt"] = (source, data)
    elif path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        # These hashes describe the containers; extracted text snapshots below
        # are the authoritative inputs, including data visible through SQLite WAL.
        containers = [path] + ([Path(str(path) + "-wal")] if Path(str(path) + "-wal").is_file() else [])
        before = {str(p.resolve()): file_sha256(p) for p in containers}
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("BEGIN")
            for table in ("pages", "corpus_pages"):
                columns = {r[1] for r in connection.execute(f"PRAGMA table_info({table})")}
                text_column = next((name for name in ("text", "page_text") if name in columns), None)
                if "page_id" in columns and text_column:
                    query = f"SELECT page_id, {text_column} FROM {table} ORDER BY page_id"
                    for page_id, text in connection.execute(query):
                        add(page_id, text)
                    provenance.append({"source_path": str(path.resolve()), "sqlite_query": query,
                                       "extraction": "read-only transaction; exact page-text snapshots"})
                    break
            else:
                raise ValueError(
                    "Database has no canonical pages table. Do not concatenate chunks. "
                    "Use --corpus benchmark/fair_v2/corpus.jsonl or an explicit page-text source."
                )
        finally:
            connection.close()
        for container in containers:
            if file_sha256(container) != before[str(container.resolve())]:
                raise ValueError("Source database changed during extraction; freeze it before running")
            provenance.append({"source_path": str(container.resolve()),
                               "sha256": before[str(container.resolve())],
                               "kind": "sqlite_container", "archived": False})
    else:
        data = path.read_bytes()
        sources["corpus/source" + path.suffix] = (path, data)
        if path.suffix.lower() == ".jsonl":
            records = (json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip())
        else:
            value = json.loads(data.decode("utf-8-sig"))
            records = ([{"page_id": key, "text": text} for key, text in value.items()]
                       if isinstance(value, dict) else value)
        for record in records:
            add(record["page_id"], _one_field(record, ("text", "page_text"), "canonical text"))
    missing = page_ids - pages.keys()
    if missing:
        raise ValueError(f"Missing canonical page texts: {sorted(missing)}")
    return pages, sources, provenance


def load_pilot_inputs(questions_path: Path, pool_path: Path, corpus_path: Path,
                      query_ids=QUERY_IDS) -> PilotInputs:
    questions, question_bytes = load_questions(questions_path, query_ids)
    pools, pool_bytes, counts, unique = load_pools(pool_path, query_ids)
    wanted = {page for pool in pools.values() for ids in pool.values() for page in ids}
    pages, sources, provenance = load_canonical_pages(corpus_path, wanted)
    sources["questions" + questions_path.suffix] = (questions_path, question_bytes)
    sources["consensus_pool.csv"] = (pool_path, pool_bytes)
    return PilotInputs(questions, pools, pages, sources, provenance, counts, unique)


class ArmAudit:
    """All file I/O belongs here/the runner, not in the pure QGCG module."""

    def __init__(self, directory: Path, arm: str, providers: dict[str, ProviderSpec],
                 *, seed: int | None, max_attempts: int, max_repair_rounds: int,
                 run_id: str, query_ids=QUERY_IDS):
        query_ids = validate_query_ids(query_ids)
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory.resolve()
        self.arm = arm
        self.run_id = run_id
        for filename in ("raw_responses.jsonl", "error_logs.jsonl", "requests.jsonl"):
            write_durable(self.directory / filename, b"")
        self.manifest: dict[str, Any] = {
            "run_id": run_id, "arm": arm, "runner_version": RUNNER_VERSION,
            "qgcg_schema_version": qgcg.SCHEMA_VERSION, "prompt_version": qgcg.PROMPT_VERSION,
            "started_at_utc": utc_now(), "finished_at_utc": None, "status": "preparing",
            "pool_scope": "per_query", "query_ids": list(query_ids),
            "billing_policy": billing_policy(providers),
            "randomness": {"requested_provider_seed": seed, "local_random_seed": None,
                           "local_randomness": "none; UUID identifies the run only",
                           "deterministic_provider_output_guaranteed": False},
            "retry_policy": {"max_attempts": max_attempts, "max_delay_seconds": 30,
                             "schema_failures_retried": any(spec.manual or spec.retry_invalid_output for spec in providers.values()),
                             "manual_submissions": "until valid or operator abort; every submission audited"},
            "max_repair_rounds": max_repair_rounds,
            "environment": {"python": sys.version, "platform": platform.platform(),
                            "pydantic": pydantic.__version__, "pyperclip": package_version("pyperclip"),
                            "sqlite": sqlite3.sqlite_version},
            "models": {role: {"requested_model": spec.requested_model,
                               "transport": spec.transport_name,
                               "transport_version": spec.transport_version,
                               "billing_mode": spec.billing_mode,
                               "settings": spec.settings,
                               "effective_seed": seed if spec.seed_supported else None,
                               "seed_supported": spec.seed_supported,
                               "observed_calls": []}
                       for role, spec in providers.items()},
            "artifacts": [], "source_provenance": [], "queries": {},
            "model_version_note": "Account CLI selected-model/usage metadata or Gemini API modelVersion is checked. API modelVersion may still be an alias, not an immutable backend build. Manual mode uses human attestation. Unknown backend revisions remain null.",
            "prompt_delivery_note": "CLI wrapper/system settings are archived per attempt. Manual mode delivers a single web user message.",
        }
        self.groups: dict[str, Any] = {"run_id": run_id, "arm": arm, "queries": {}}
        self.checkpoint()

    @classmethod
    def reopen(cls, directory: Path, providers: dict[str, ProviderSpec]):
        """Read and verify a finalized audit before any recovery mutation."""
        audit = cls.__new__(cls)
        audit.directory = directory.resolve()
        audit.manifest = json.loads((directory / "run_manifest.json").read_bytes())
        audit.groups = json.loads((directory / "groups.json").read_bytes())
        audit.arm, audit.run_id = audit.manifest["arm"], audit.manifest["run_id"]
        if not audit.manifest.get("finished_at_utc"):
            raise ValueError(f"{audit.arm}: recovery requires a finalized run; do not run concurrently")
        if (audit.groups["run_id"], audit.groups["arm"]) != (audit.run_id, audit.arm):
            raise ValueError("Manifest/results identity mismatch")
        for record in audit.manifest["artifacts"]:
            path = (audit.directory / record["relative_path"]).resolve()
            path.relative_to(audit.directory)
            if file_sha256(path) != record["sha256"]:
                raise ValueError(f"Archived input changed: {path}")
        for name, record in audit.manifest["output_files"].items():
            if file_sha256(audit.directory / name) != record["sha256"]:
                raise ValueError(f"Audit output changed: {name}")
        for name, prompt in PROMPTS.items():
            if (audit.directory / f"prompts/{name}.txt").read_bytes() != prompt.encode("utf-8"):
                raise ValueError("Recovery cannot change the experimental prompts")
        core_snapshot = next(r for r in audit.manifest["artifacts"]
                             if r["kind"] == "implementation" and r["source_path"] == str(Path(qgcg.__file__).resolve()))
        if core_snapshot["sha256"] != file_sha256(Path(qgcg.__file__)):
            raise ValueError("Recovery cannot change the QGCG algorithm/schemas")
        for role, spec in providers.items():
            old = audit.manifest["models"][role]
            if (old["requested_model"], old["transport"], old["billing_mode"]) != (spec.requested_model, spec.transport_name, spec.billing_mode):
                raise ValueError("Recovery cannot change model selection or transport type")
        audit._groups_digest = sha256(json_bytes(audit.groups))
        return audit

    def begin_recovery(self, providers: dict[str, ProviderSpec], query_ids: list[str]) -> str:
        segment = "recovery-" + uuid4().hex[:12]
        for name in ("run_manifest.json", "groups.json"):
            self.artifact(f"{segment}/{name}", (self.directory / name).read_bytes(), "pre_recovery_snapshot")
        files = implementation_paths(providers)
        for index, path in enumerate(sorted(files)):
            self.artifact(f"{segment}/code/{index:02d}_{path.name}", path.read_bytes(),
                          "recovery_implementation", source=path)
        snapshot_provider_preflights(self, providers, prefix=segment + "/")
        self.manifest.setdefault("recovery_segments", []).append({
            "segment_id": segment, "runner_version": RUNNER_VERSION,
            "started_at_utc": utc_now(), "query_ids": query_ids,
            "prior_query_outcomes": {q: self.groups["queries"].get(q) for q in query_ids},
            "models": {role: {"requested_model": spec.requested_model,
                              "transport": spec.transport_name,
                              "transport_version": spec.transport_version, "settings": spec.settings}
                       for role, spec in providers.items()},
            "policy": "retry failed queries from immutable inputs; reuse matching validated responses, retain successful queries, and append all logs",
        })
        self.manifest.pop("output_files", None)
        self.manifest.update(status="running", finished_at_utc=None)
        self.log("requests.jsonl", "recovery_started", segment_id=segment, query_ids=query_ids)
        self.checkpoint()
        self.redecode_saved_responses(providers, query_ids, segment)
        return segment + "-"

    def redecode_saved_responses(self, providers, query_ids, segment):
        """Recover successful CLI envelopes rejected by an older local decoder.

        New code snapshots precede this operation. Original raw records remain
        unchanged, and the replay adapter still applies all Pydantic/context
        checks before accepting any recovered model text.
        """
        rows = [json.loads(line) for line in (self.directory / "raw_responses.jsonl").read_bytes().splitlines()]
        decoded_keys = {(r.get("call_id"), r.get("attempt")) for r in rows if r["event"] == "model_text"}
        channels = {}
        for row in rows:
            channels.setdefault((row.get("call_id"), row.get("attempt")), {})[row["event"]] = row
        for key, records in channels.items():
            raw = records.get("provider_raw")
            if not raw or key in decoded_keys or raw.get("query_id") not in query_ids or raw.get("exit_code") != 0:
                continue
            spec = providers[raw["role"]]
            if spec.manual:
                continue
            response = RawProviderResponse(raw["raw_text"],
                        records.get("provider_stderr", {}).get("raw_text"), exit_code=0,
                        final_text=records.get("provider_final", {}).get("raw_text"))
            try:
                decoded = spec.decode(response)
            except (ValueError, ProviderFailure):
                continue
            if decoded.reported_model != spec.requested_model:
                continue
            tags = {name: raw[name] for name in ("query_id", "pass_number", "role", "call_id",
                                                "phase", "requested_model", "attempt")}
            self.log("raw_responses.jsonl", "model_text", **tags,
                     raw_text=decoded.text, text_sha256=sha256(decoded.text.encode("utf-8")),
                     decoding_recovery_segment=segment, original_response_timestamp_utc=raw["timestamp_utc"])
            self.manifest["models"][raw["role"]]["observed_calls"].append({
                **tags, "timestamp_utc": raw["timestamp_utc"], "reported_model": decoded.reported_model,
                "backend_revision": decoded.backend_revision, "backend_revision_exposed": decoded.backend_revision is not None,
                "provider_request_id": decoded.provider_request_id, "finish_reason": decoded.finish_reason,
                "identity_source": decoded.identity_source, "human_model_label": None, "provider_verified": True,
                "decoding_recovery_segment": segment,
            })
            self.log("requests.jsonl", "saved_provider_response_redecoded", **tags,
                     segment_id=segment, original_response_sha256=raw["text_sha256"], provider_request_made=False)
        self.checkpoint()

    def checkpoint(self) -> None:
        write_durable(self.directory / "run_manifest.json", json_bytes(self.manifest), replace=True)
        groups_bytes = json_bytes(self.groups)
        digest = sha256(groups_bytes)
        if digest != getattr(self, "_groups_digest", None):
            write_durable(self.directory / "groups.json", groups_bytes, replace=True)
            self._groups_digest = digest

    def artifact(self, relative: str, data: bytes, kind: str, *, source: Path | None = None):
        path = self.directory / relative
        write_durable(path, data)
        record = {"path": str(path), "relative_path": relative, "sha256": sha256(data),
                  "bytes": len(data), "kind": kind}
        if source is not None:
            record["source_path"] = str(source.resolve())
        self.manifest["artifacts"].append(record)
        return record

    def log(self, filename: str, event: str, **fields) -> None:
        record = {"timestamp_utc": utc_now(), "run_id": self.run_id,
                  "arm": self.arm, "event": event, **fields}
        encoded = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        with (self.directory / filename).open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def error(self, event: str, error: BaseException | None = None, **fields) -> None:
        if error is not None:
            fields.update(error_type=type(error).__name__, message=str(error))
            if isinstance(error, ValidationError):
                fields["validation_errors"] = json.loads(error.json(include_url=False))
        self.log("error_logs.jsonl", event, **fields)

    def finish(self, status: str) -> None:
        self.manifest.update(status=status, finished_at_utc=utc_now())
        self.checkpoint()
        self.manifest["output_files"] = {
            name: {"path": str(self.directory / name), "sha256": file_sha256(self.directory / name)}
            for name in ("groups.json", "raw_responses.jsonl", "error_logs.jsonl", "requests.jsonl")
        }
        # The manifest cannot hash itself; output-file hashes describe final bytes.
        write_durable(self.directory / "run_manifest.json", json_bytes(self.manifest), replace=True)


def implementation_paths(providers: dict[str, ProviderSpec]) -> set[Path]:
    paths = {Path(__file__).resolve(), Path(qgcg.__file__).resolve()}
    for spec in providers.values():
        paths.update(Path(path).resolve() for path in spec.implementation_files)
        for callback in (spec.request, spec.decode):
            source = inspect.getsourcefile(callback)
            if source:
                paths.add(Path(source).resolve())
    return paths


def snapshot_provider_preflights(audit, providers, prefix=""):
    for role, spec in providers.items():
        folder = spec.settings.get("preflight_directory")
        if spec.billing_mode == "developer_api" and folder:
            for name in ("request.json", "response.json", "response_body.bin"):
                path = Path(folder) / name
                audit.artifact(f"{prefix}preflight/{role}/{name}", path.read_bytes(),
                               "provider_preflight", source=path)


def snapshot_inputs(audit: ArmAudit, inputs: PilotInputs, providers: dict[str, ProviderSpec]) -> None:
    snapshot_provider_preflights(audit, providers)
    for relative, (path, data) in inputs.source_files.items():
        audit.artifact("inputs/original/" + relative, data, "original_input", source=path)
    audit.manifest["source_provenance"] = inputs.source_provenance
    audit.manifest["pool_counts"] = {"global_memberships": inputs.membership_counts,
                                    "global_unique_pages": inputs.unique_page_counts,
                                    "selected_per_query": {qid: len(ids) for qid, ids in inputs.pools[audit.arm].items()}}
    for qid in audit.manifest["query_ids"]:
        audit.artifact(f"inputs/{qid}/question.txt", inputs.questions[qid].encode("utf-8"), "question_text")
        selected = inputs.pools[audit.arm][qid]
        audit.artifact(f"inputs/{qid}/page_ids.json", json_bytes(selected), "pool_membership")
        # Index-based filenames avoid interpreting externally supplied IDs as paths.
        page_index = []
        for index, page_id in enumerate(selected):
            record = audit.artifact(f"inputs/{qid}/pages/{index:04d}.txt",
                                    inputs.pages[page_id].encode("utf-8"), "canonical_page_text")
            page_index.append({"page_id": page_id, **record})
        audit.artifact(f"inputs/{qid}/pages.json", json_bytes(page_index), "page_text_index")
    for name, prompt in PROMPTS.items():
        audit.artifact(f"prompts/{name}.txt", prompt.encode("utf-8"), "system_prompt")
    for name, schema in (("pass1", qgcg.PASS1_ADAPTER.json_schema()),
                         ("pass2", qgcg.Pass2Output.model_json_schema()),
                         ("pass3", qgcg.PASS3_ADAPTER.json_schema())):
        audit.artifact(f"schemas/{name}.json", json_bytes(schema), "json_schema")
    for index, path in enumerate(sorted(implementation_paths(providers))):
        audit.artifact(f"code/{index:02d}_{path.name}", path.read_bytes(), "implementation", source=path)
    audit.manifest["status"] = "prepared"
    audit.checkpoint()


class AuditedLLMAdapter:
    """Implements configure_qgcg(adapter); capture precedes all output parsing."""

    def __init__(self, audit: ArmAudit, providers: dict[str, ProviderSpec], query_id: str,
                 *, max_attempts: int = 3, seed: int | None = None,
                 timeout_seconds: float = 600, sleeper: Callable[[float], None] = time.sleep,
                 call_prefix: str = "", reuse_validated: bool = False):
        self.audit, self.providers, self.query_id = audit, providers, query_id
        self.max_attempts, self.seed = max_attempts, seed
        self.timeout_seconds, self.sleeper = timeout_seconds, sleeper
        self.sequence = 0
        self.call_prefix = call_prefix
        self.last_call: dict[str, Any] | None = None
        self.cached_responses = []
        self.recovery_feedback = None
        if reuse_validated:
            rows = [json.loads(line) for line in (audit.directory / "raw_responses.jsonl").read_bytes().splitlines()]
            successful_exits = {(r.get("call_id"), r.get("attempt")) for r in rows
                                if r["event"] == "provider_raw" and r.get("exit_code") == 0}
            identities = {(c["call_id"], c["attempt"]): c
                          for model in audit.manifest["models"].values() for c in model["observed_calls"]
                          if c.get("provider_verified")}
            for row in reversed(rows):
                key = (row.get("call_id"), row.get("attempt"))
                if (row["event"] == "model_text" and row.get("query_id") == query_id
                        and key in successful_exits and key in identities):
                    self.cached_responses.append((row, identities[key]))

    def _reuse(self, role, system_prompt, user_prompt, json_schema, payload, repair, tags):
        self.recovery_feedback = None
        for row, identity in self.cached_responses:
            if row["role"] != role or identity.get("reported_model") != self.providers[role].requested_model:
                continue
            old = self.audit.directory / "requests" / row["call_id"]
            if ((old / "system_prompt.txt").read_bytes() != system_prompt.encode("utf-8")
                    or (old / "user_prompt.json").read_bytes() != user_prompt.encode("utf-8")
                    or (old / "schema.json").read_bytes() != json_bytes(json_schema)):
                continue
            text = row["raw_text"]
            if sha256(text.encode("utf-8")) != row["text_sha256"]:
                raise ValueError("Cached model text checksum mismatch")
            try:
                self._validate_model_text(text, role, payload, repair)
            except ValueError as error:
                if self.recovery_feedback is None:
                    self.recovery_feedback = self._correction_feedback(text, error, payload)
                    self.audit.log("requests.jsonl", "invalid_response_correction_seeded", **tags,
                                   original_call_id=row["call_id"], original_attempt=row["attempt"],
                                   text_sha256=row["text_sha256"], provider_request_made=False)
                continue  # A previously rejected candidate is never a cache hit.
            self.audit.log("requests.jsonl", "validated_response_reused", **tags,
                           original_call_id=row["call_id"], original_attempt=row["attempt"],
                           original_reported_model=identity["reported_model"],
                           text_sha256=row["text_sha256"], provider_request_made=False)
            return text
        return None

    @staticmethod
    def _correction_feedback(text: str, error: ValueError, payload: dict) -> str:
        details = {"validation_error": str(error), "previous_model_response": text,
                   "instruction": "Correct the rejected response using the original canonical pages and schema. Copy requirement strings and quotes exactly, preserving punctuation and line endings."}
        # Report ALL quote mismatches together, not just the first Pydantic
        # failure. This is diagnostic context; it never alters a model answer.
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        pages = payload.get("pages")
        if isinstance(value, dict) and isinstance(value.get("mappings"), list) and isinstance(pages, dict):
            invalid_quotes = []
            for index, mapping in enumerate(value["mappings"]):
                if not isinstance(mapping, dict) or not isinstance(mapping.get("source_excerpts"), list):
                    continue
                for excerpt in mapping["source_excerpts"]:
                    if not isinstance(excerpt, dict):
                        continue
                    page_id, quote = excerpt.get("page_id"), excerpt.get("quote")
                    if isinstance(page_id, str) and isinstance(quote, str) and quote not in pages.get(page_id, ""):
                        invalid_quotes.append({"mapping_index": index, "page_id": page_id, "submitted_quote": quote})
            details["invalid_quotes"] = invalid_quotes
        return json_bytes(details).decode("utf-8")

    def _raw(self, response: RawProviderResponse, tags: dict[str, Any]) -> None:
        for channel, text in (("provider_raw", response.raw_text), ("provider_stderr", response.stderr),
                              ("provider_final", response.final_text)):
            if text is not None:
                if not isinstance(text, str):
                    raise TypeError("Provider responses must retain the original text strings")
                self.audit.log("raw_responses.jsonl", channel, **tags, raw_text=text,
                               text_sha256=sha256(text.encode("utf-8")), exit_code=response.exit_code,
                               human_model_label=response.human_model_label,
                               capture_method=response.capture_method, capture_source=response.capture_source)

    def _validate_model_text(self, text: str, role: str, payload: dict, repair: bool) -> None:
        json.loads(text)  # Classify malformed JSON separately from strict schema failures.
        if role == "model_a":
            qgcg.PASS1_ADAPTER.validate_json(text)
        elif role == "model_b":
            reqs = None if repair else qgcg.PASS1_ADAPTER.validate_python(payload["requirements"])
            candidate = qgcg.Pass2Output.model_validate_json(
                text, context={"requirements": reqs, "pages": payload["pages"]}
            )
            if repair:
                # Check the same cross-proposal rule before a manual submission
                # reaches the core's repair rollback handler.
                original = qgcg.Pass2Output.model_validate(
                    payload["proposed_groups"], context={"requirements": None, "pages": payload["pages"]}
                )
                issues = qgcg.PASS3_ADAPTER.validate_python(
                    payload["verifier_issues"], context={"requirement_references": payload["requirement_references"]}
                )
                qgcg._check_unaffected_groups(original, candidate, issues)
        else:
            qgcg.PASS3_ADAPTER.validate_json(
                text, context={"requirement_references": payload["requirement_references"]}
            )

    def __call__(self, *, role: str, system_prompt: str, payload: dict, json_schema: dict) -> str:
        spec = self.providers[role]
        validate_providers(self.providers)  # One policy shared by orchestration and each invocation.
        self.sequence += 1
        call_id = f"{self.call_prefix}{self.query_id}-{self.sequence:04d}-{role}"
        repair = role == "model_b" and "verifier_issues" in payload
        phase = "repair" if repair else ("verification" if role == "model_c" else "initial")
        tags = {"query_id": self.query_id, "pass_number": PASS_NUMBERS[role], "role": role,
                "call_id": call_id, "phase": phase, "requested_model": spec.requested_model,
                "transport_version": spec.transport_version}
        self.last_call = tags
        user_prompt = json_bytes(payload).decode("utf-8")
        inputs = [
            self.audit.artifact(f"requests/{call_id}/system_prompt.txt", system_prompt.encode("utf-8"), "request_system_prompt"),
            self.audit.artifact(f"requests/{call_id}/user_prompt.json", user_prompt.encode("utf-8"), "request_user_prompt"),
            self.audit.artifact(f"requests/{call_id}/schema.json", json_bytes(json_schema), "request_schema"),
        ]
        prompt_path = response_path = None
        if spec.manual:
            prompt = assemble_hitl_prompt(system_prompt, user_prompt, json_schema)
            artifact = self.audit.artifact(f"requests/{call_id}/clipboard_prompt.txt",
                                           prompt.encode("utf-8"), "exact_web_prompt")
            inputs.append(artifact)
            prompt_path = Path(artifact["path"])
            response_path = prompt_path.with_name("response_inbox.txt")
            write_durable(response_path, b"")  # Mutable inbox; submitted text is preserved in the raw log.
        self.audit.checkpoint()  # Durable input manifest BEFORE any provider call.
        reused = self._reuse(role, system_prompt, user_prompt, json_schema, payload, repair, tags)
        if reused is not None:
            return reused
        attempt = 0
        feedback = self.recovery_feedback
        while spec.manual or attempt < self.max_attempts:
            attempt += 1
            attempt_tags = {**tags, "attempt": attempt}
            def event_logger(event, **details):
                if event.endswith("error"):
                    self.audit.error(event, **attempt_tags, **details)
                else:
                    self.audit.log("requests.jsonl", event, **attempt_tags, **details)
            def raw_logger(kind, text):
                self.audit.log("raw_responses.jsonl", kind, **attempt_tags, raw_text=text,
                               text_sha256=sha256(text.encode("utf-8")))
            def artifact_writer(name, data):
                if Path(name).name != name:
                    raise ValueError("Transport artifact names must be simple filenames")
                self.audit.artifact(f"requests/{call_id}/attempt_{attempt}/{name}", data, "transport_input")
                self.audit.checkpoint()
            request = ProviderRequest(call_id, attempt, self.query_id, self.audit.arm, role,
                                      spec.requested_model, system_prompt, user_prompt, json_schema,
                                      self.seed if spec.seed_supported else None, self.timeout_seconds,
                                      prompt_path, response_path, event_logger, raw_logger, artifact_writer, feedback)
            self.audit.log("requests.jsonl", "provider_attempt", **attempt_tags,
                           inputs=inputs, effective_seed=request.seed,
                           timeout_seconds=None if spec.manual else request.timeout_seconds)
            try:
                response = spec.request(request)  # ONE attempt; absolutely no JSON parsing here.
            except ManualRunAborted as error:
                if error.partial is not None:
                    self._raw(error.partial, attempt_tags)
                self.audit.error("manual_aborted", error, **attempt_tags)
                raise
            except Exception as error:
                if isinstance(error, ProviderFailure) and error.raw_response is not None:
                    self._raw(error.raw_response, attempt_tags)
                retry = isinstance(error, ProviderFailure) and error.retryable and attempt < self.max_attempts
                self.audit.error("provider_error", error, **attempt_tags, will_retry=retry)
                if not retry:
                    raise
                delay = (error.retry_after_seconds if error.retry_after_seconds is not None
                         else 2 ** (attempt - 1))
                delay = max(0.0, min(30.0, float(delay)))
                self.audit.error("provider_retry", **attempt_tags, next_attempt=attempt + 1,
                                 delay_seconds=delay, reason=str(error))
                self.sleeper(delay)
                continue
            if not isinstance(response, RawProviderResponse):
                # Retain even a mistakenly returned plain string before rejecting its contract.
                if isinstance(response, str):
                    self._raw(RawProviderResponse(response), attempt_tags)
                error = TypeError("request() must return RawProviderResponse, without parsing it")
                self.audit.error("transport_contract_error", error, **attempt_tags)
                raise error
            self._raw(response, attempt_tags)  # fsync completes before decode().
            try:
                decoded = spec.decode(response)
                if not isinstance(decoded, DecodedResponse) or not isinstance(decoded.text, str):
                    raise TypeError("decode() must return DecodedResponse with exact model text")
            except Exception as error:
                event = "json_parsing_error" if isinstance(error, json.JSONDecodeError) else "provider_decode_error"
                self.audit.error(event, error, **attempt_tags, location="provider_envelope")
                raise
            self.audit.log("raw_responses.jsonl", "model_text", **attempt_tags,
                           raw_text=decoded.text, text_sha256=sha256(decoded.text.encode("utf-8")))
            observation = {**attempt_tags, "timestamp_utc": utc_now(),
                           "reported_model": decoded.reported_model,
                           "backend_revision": decoded.backend_revision,
                           "backend_revision_exposed": decoded.backend_revision is not None,
                           "provider_request_id": decoded.provider_request_id,
                           "finish_reason": decoded.finish_reason,
                           "identity_source": decoded.identity_source,
                           "human_model_label": decoded.human_model_label,
                           "provider_verified": not spec.manual}
            self.audit.manifest["models"][role]["observed_calls"].append(observation)
            self.audit.checkpoint()
            identity_ok = (
                decoded.identity_source == "human_attested_web_ui"
                and isinstance(decoded.human_model_label, str)
                and _matches_web_model(role, decoded.human_model_label)
            ) if spec.manual else decoded.reported_model == spec.requested_model
            if response.exit_code != 0 or not identity_ok:
                error = ProviderFailure("Provider failed or actual model differs from the requested model; no fallback")
                self.audit.error("model_or_transport_verification_error", error, **attempt_tags,
                                 reported_model=decoded.reported_model, exit_code=response.exit_code)
                raise error
            try:
                self._validate_model_text(decoded.text, role, payload, repair)
            except ValueError as error:
                event = ("json_parsing_error" if isinstance(error, json.JSONDecodeError)
                         else "pydantic_validation_error" if isinstance(error, ValidationError)
                         else "repair_constraint_error")
                self.audit.error(event, error, **attempt_tags, location="model_output")
                if spec.manual:
                    self.audit.error("manual_response_retry", **attempt_tags, next_attempt=attempt + 1,
                                     reason=str(error), automatic_model_request=False)
                    (spec.validation_feedback or print)(error)
                    continue
                if spec.retry_invalid_output and attempt < self.max_attempts:
                    self.audit.error("schema_response_retry", **attempt_tags, next_attempt=attempt + 1,
                                     reason=str(error), automatic_model_request=True)
                    # The new stateless CLI session needs the rejected answer
                    # to make a targeted correction instead of recreating it.
                    # This is explicitly prior model output, never source data.
                    feedback = self._correction_feedback(decoded.text, error, payload)
                    continue
                # Return the EXACT response. The pure pipeline owns validation
                # failure handling, including its conservative repair rollback.
            return decoded.text
        raise RuntimeError("No provider attempt was made")


def billing_policy(providers):
    return ("gemini_api_only_gpt_claude_subscription_no_fallback"
            if providers["model_c"].billing_mode == "developer_api"
            else "subscription_only_no_api_key_fallback")


def validate_providers(providers: dict[str, ProviderSpec]) -> None:
    if set(providers) != set(PASS_NUMBERS):
        raise ValueError("Provider factory must supply model_a, model_b, and model_c")
    for role, spec in providers.items():
        api_exception = (isinstance(spec, ProviderSpec) and role == "model_c"
                         and spec.billing_mode == "developer_api" and spec.settings.get("api_authorized") is True
                         and spec.transport_name == "GeminiDeveloperAPI")
        if not isinstance(spec, ProviderSpec) or (spec.billing_mode != "subscription" and not api_exception):
            raise ValueError(f"{role} must supply a subscription-only ProviderSpec")
        if not spec.requested_model or not spec.transport_name or not spec.transport_version:
            raise ValueError(f"{role} must declare model and transport versions")
        json_bytes(spec.settings)  # Reject unrecordable/non-JSON protocol settings.


def run_pilot(*, questions_path: Path, pool_path: Path, corpus_path: Path,
              output_root: Path, providers: dict[str, ProviderSpec] | None = None,
              prepare_only: bool = False, max_attempts: int = 3,
              max_repair_rounds: int = 1, seed: int | None = None,
              timeout_seconds: float = 600, resume_failed: bool = False,
              arms: tuple[str, ...] = tuple(ARMS),
              query_ids: tuple[str, ...] = QUERY_IDS) -> dict[str, str]:
    """Run selected questions; retain failed trials and continue other trials."""
    query_ids = validate_query_ids(query_ids)
    if providers is None:
        raise ValueError("Supply audited subscription providers; the CLI configures account transports by default")
    validate_providers(providers)
    if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
        raise ValueError("max_attempts must be from 1 through 10")
    if type(max_repair_rounds) is not int or not 0 <= max_repair_rounds <= 3:
        raise ValueError("max_repair_rounds must be from 0 through 3")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= ARMS.keys():
        raise ValueError("arms must contain E1 and/or E2 without duplicates")
    if resume_failed:
        if prepare_only:
            raise ValueError("--resume-failed cannot be combined with --prepare-only")
        # Verify every selected arm before modifying any; other arms can run
        # independently in their own directories and are never touched here.
        audits = {arm: ArmAudit.reopen(output_root / arm, providers) for arm in arms}
        if len({a.run_id for a in audits.values()}) != 1:
            raise ValueError("E1/E2 must belong to the same original run")
        for audit in audits.values():
            if audit.manifest["query_ids"] != list(query_ids):
                raise ValueError("Recovery must preserve the original query selection and order")
            if (audit.manifest["retry_policy"]["max_attempts"] != max_attempts
                    or audit.manifest["max_repair_rounds"] != max_repair_rounds
                    or audit.manifest["randomness"]["requested_provider_seed"] != seed):
                raise ValueError("Recovery must preserve retry/repair limits and seed")
    else:
        for arm in arms:
            if (output_root / arm).exists():
                raise FileExistsError(f"Refusing to overwrite {output_root / arm}; choose a new output root or --resume-failed")
        run_id = uuid4().hex
        audits = {arm: ArmAudit(output_root / arm, arm, providers, seed=seed,
                                max_attempts=max_attempts, max_repair_rounds=max_repair_rounds,
                                run_id=run_id, query_ids=query_ids) for arm in arms}
        try:
            inputs = load_pilot_inputs(questions_path, pool_path, corpus_path, query_ids)
            for audit in audits.values():
                snapshot_inputs(audit, inputs, providers)
        except Exception as error:
            for audit in audits.values():
                audit.error("input_preflight_error", error, query_id=None, pass_number=None)
                audit.manifest["requested_input_paths"] = {
                    "questions": str(questions_path.resolve()), "pool": str(pool_path.resolve()),
                    "corpus": str(corpus_path.resolve()),
                }
                audit.finish("blocked_input")
            raise
    if prepare_only:
        for audit in audits.values():
            audit.finish("prepared_only")
        return {arm: "prepared_only" for arm in arms}
    for arm, audit in audits.items():
        pending_query_ids = [q for q in query_ids if not resume_failed
                     or audit.groups["queries"].get(q, {}).get("status") != "completed"]
        if not pending_query_ids:
            continue
        call_prefix = audit.begin_recovery(providers, pending_query_ids) if resume_failed else ""
        audit.manifest["status"] = "running"
        for query_id in pending_query_ids:
            adapter = AuditedLLMAdapter(audit, providers, query_id, max_attempts=max_attempts,
                                        seed=seed, timeout_seconds=timeout_seconds, call_prefix=call_prefix,
                                        reuse_validated=resume_failed)
            audit.manifest["queries"][query_id] = {"status": "running", "started_at_utc": utc_now()}
            audit.checkpoint()
            previous_configuration = qgcg._CONFIGURATION.get()
            qgcg.configure_qgcg(adapter, max_repair_rounds=max_repair_rounds)
            try:
                # Both fresh and recovery runs consume the archived exact bytes.
                question = (audit.directory / f"inputs/{query_id}/question.txt").read_bytes().decode("utf-8")
                page_index = json.loads((audit.directory / f"inputs/{query_id}/pages.json").read_bytes())
                pages = [{"page_id": row["page_id"],
                          "text": (audit.directory / row["relative_path"]).read_bytes().decode("utf-8")}
                         for row in page_index]
                result = qgcg.build_qgcg(question, pages)
                # The core intentionally catches some repair errors. Mirror them
                # to the audit log instead of losing them at the adapter boundary.
                for record in result["repair_history"]:
                    if record.get("validation_error"):
                        tags = adapter.last_call or {"query_id": query_id, "pass_number": 2}
                        audit.error("repair_validation_error", **tags,
                                    round=record["round"], message=record["validation_error"],
                                    last_call=adapter.last_call)
                audit.groups["queries"][query_id] = {"status": "completed", "result": result}
                audit.manifest["queries"][query_id].update(
                    status="completed", review_status=result["review_status"], finished_at_utc=utc_now())
            except Exception as error:
                tags = adapter.last_call or {"query_id": query_id, "pass_number": None}
                audit.error("pipeline_error", error, **tags)
                failure = {"status": "failed", "error_type": type(error).__name__, "message": str(error)}
                audit.groups["queries"][query_id] = failure
                audit.manifest["queries"][query_id].update(**failure, finished_at_utc=utc_now())
            except BaseException as error:
                # Preserve interruptions without pretending the trial completed.
                audit.error("interrupted", error, query_id=query_id, pass_number=None)
                audit.manifest["queries"][query_id].update(status="interrupted", finished_at_utc=utc_now())
                audit.groups["queries"][query_id] = {"status": "interrupted", "message": str(error)}
                for pending in audits.values():
                    if pending.manifest["finished_at_utc"] is None:
                        pending.finish("interrupted")
                raise
            finally:
                # Restore an embedding caller's configuration without changing
                # the pure module's configure_qgcg(adapter) public interface.
                qgcg._CONFIGURATION.set(previous_configuration)
            audit.checkpoint()
        statuses = Counter(query["status"] for query in audit.groups["queries"].values())
        audit.finish("completed" if statuses["completed"] == len(query_ids)
                     else "failed" if statuses["failed"] == len(query_ids) else "partial_failure")
    return {arm: audit.manifest["status"] for arm, audit in audits.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--questions", type=Path, default=Path("benchmark/questions.csv"))
    parser.add_argument("--pool", type=Path, default=Path("benchmark/qrels/consensus_qrels_v2.csv"))
    parser.add_argument("--corpus", type=Path, default=Path("benchmark/fair_v2/corpus.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("benchmark/qgcg_pilot"))
    parser.add_argument("--query-ids", nargs="+", default=QUERY_IDS,
                        help="Questions to run; defaults to Q2 Q3. Stress checks: Q1 Q6 Q7")
    parser.add_argument("--transport", choices=("subscription", "manual"), default="subscription")
    parser.add_argument("--gemini-transport", choices=("subscription", "api"), default="subscription",
                        help="Explicit Gemini-only developer API opt-in; GPT/Claude still use subscription CLIs")
    parser.add_argument("--gemini-model", default="gemini-3.1-pro-low",
                        choices=("gemini-3.1-pro-low", "gemini-3.1-pro-high"))
    parser.add_argument("--check-accounts", action="store_true", help="Check subscription logins/model access without inference")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze inputs and manifests; no model calls")
    parser.add_argument("--resume-failed", action="store_true",
                        help="Retry only unfinished queries in a finalized run, using verified frozen inputs")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=tuple(ARMS),
                        help="Experimental arms to execute/recover; defaults to both E1 and E2")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-repair-rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args()
    def resolve(path: Path) -> Path:
        return (path if path.is_absolute() else args.data_root / path).resolve()
    provider_preflight_complete = False
    try:
        if args.transport == "manual" and args.gemini_transport == "api":
            raise ValueError("Gemini API mode requires automatic subscription transports for GPT/Claude")
        if args.transport == "manual":
            if not args.prepare_only:
                _pyperclip()
            providers = hitl_providers()
        else:
            if str(WORKSPACE) not in sys.path:
                sys.path.insert(0, str(WORKSPACE))
            if args.gemini_transport == "api":
                from scripts.qgcg_gemini_api import mixed_providers
                providers = mixed_providers(args.data_root.resolve(), resolve(args.output_root))
            else:
                from scripts.qgcg_subscription_transports import subscription_providers
                providers = subscription_providers(args.data_root.resolve(), gemini_model=args.gemini_model)
        provider_preflight_complete = True
        if args.check_accounts:
            print(json.dumps({role: {"requested_model": spec.requested_model, **spec.settings}
                              for role, spec in providers.items()}, ensure_ascii=False, indent=2))
            return 0
        statuses = run_pilot(
            questions_path=resolve(args.questions), pool_path=resolve(args.pool),
            corpus_path=resolve(args.corpus), output_root=resolve(args.output_root),
            providers=providers, prepare_only=args.prepare_only, max_attempts=args.max_attempts,
            max_repair_rounds=args.max_repair_rounds, seed=args.seed,
            timeout_seconds=args.timeout_seconds,
            resume_failed=args.resume_failed,
            arms=tuple(args.arms),
            query_ids=tuple(args.query_ids),
        )
        print(json.dumps(statuses, ensure_ascii=False))
        return 0 if all(s in {"completed", "prepared_only"} for s in statuses.values()) else 1
    except KeyboardInterrupt as error:
        print(f"Pilot interrupted: {error}. Existing prompts, responses, and manifests are preserved.", file=sys.stderr)
        return 130
    except Exception as error:
        if not provider_preflight_complete:
            # Account/model preflight happens before ArmAudit exists. Preserve
            # its failure separately; never fabricate groups or inference logs.
            directory = resolve(args.output_root) / "preflight"
            record = {"timestamp_utc": utc_now(), "event": "provider_preflight_error",
                      "runner_version": RUNNER_VERSION, "query_ids": args.query_ids,
                      "arms": list(args.arms), "inference_started": False,
                      "billing_policy": "gemini_api_only_gpt_claude_subscription_no_fallback" if args.gemini_transport == "api" else "subscription_only_no_api_key_fallback",
                      "error_type": type(error).__name__, "message": str(error),
                      "details": getattr(error, "preflight_details", {}),
                      "requested_models": {**DEFAULT_MODELS, "model_c": "gemini-3.1-pro-preview" if args.gemini_transport == "api" else args.gemini_model},
                      "runner_sha256": file_sha256(Path(__file__).resolve()),
                      "next_step": "Restore requested subscription model access or explicitly authorize a separately labeled model change."}
            write_durable(directory / (uuid4().hex + ".json"), json_bytes(record))
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Ensure externally imported ProviderSpec has the same identity as this module.
    sys.modules.setdefault("scripts.run_qgcg_pilot", sys.modules[__name__])
    raise SystemExit(main())
