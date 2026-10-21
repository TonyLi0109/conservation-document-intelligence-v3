"""Official account-authenticated CLI transports for the audited QGCG pilot.

No API keys, token extraction, browser scraping, or model/billing fallback.
Codex uses ChatGPT login; Claude uses the existing claude.ai account profile;
Antigravity uses the Windows keyring login with extra-credit fallback disabled.
Raw CLI stdout/stderr and final messages are saved before response decoding.
The capture boundary is the official CLI, not its private HTTP implementation.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time

from scripts.run_qgcg_pilot import (
    DecodedResponse, ProviderFailure, ProviderRequest, ProviderSpec,
    RawProviderResponse, file_sha256, json_bytes,
)


BLOCKED_ENV = {
    "OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE",
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CONFIG_DIR", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_GCA",
    "GEMINI_API_BASE_URL", "GOOGLE_GEMINI_BASE_URL", "GEMINI_CLI_HOME", "GEMINI_SYSTEM_MD",
    "GEMINI_CLI_SYSTEM_SETTINGS_PATH", "GEMINI_CLI_SYSTEM_DEFAULTS_PATH", "NODE_OPTIONS",
}
AGENT_NAME = "qgcg-benchmark"
AGENT_HEADER = """---
name: qgcg-benchmark
description: Isolated QGCG evidence verifier
tools: [finish]
mainAgent: true
subagent: false
inheritMcp: false
commandExecutionPolicy: "off"
mcpServers: []
skills: []
plugins: []
---
Submit the final structured JSON through the finish tool immediately.
The finish tool only submits the response; use no other tools or external sources.
Return the requested output inside the enclosing schema's response field.

"""
GEMINI_SETTINGS = {
    "useG1Credits": False, "toolPermission": "request-review",
    "permissions": {"deny": ["read_file(*)", "write_file(*)", "command(*)",
                              "unsandboxed(*)", "read_url(*)", "execute_url(*)", "mcp(*)"]},
}


def account_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items()
            if key.upper() not in BLOCKED_ENV
            and not key.upper().startswith(("AGY_", "ANTIGRAVITY_", "CASCADE_"))}


def _text(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""


def _capture(request: ProviderRequest, kind: str, text: str) -> None:
    if request.raw_logger:
        request.raw_logger(kind, text)


def _artifact(request: ProviderRequest, name: str, value: bytes) -> None:
    if request.artifact_writer:
        request.artifact_writer(name, value)


def _record_request(request: ProviderRequest, command: list[str], stdin: str, settings: dict) -> None:
    _artifact(request, "command.json", json_bytes(command))
    _artifact(request, "stdin.txt", stdin.encode("utf-8"))
    _artifact(request, "protocol_settings.json", json_bytes(settings))


def _user_input(request: ProviderRequest) -> str:
    if not request.validation_feedback:
        return request.user_prompt
    return (request.user_prompt + "\n\nLOCAL VALIDATION ERROR FROM PREVIOUS ATTEMPT:\n"
            + request.validation_feedback
            + "\nReturn corrected raw JSON only, starting with { or [. Do not use Markdown code fences, "
              "introductory text, or trailing commentary. Preserve all original source constraints.")


def _run(request: ProviderRequest, command: list[str], *, cwd: Path,
         environment: dict[str, str], stdin: str, final_path: Path | None = None) -> RawProviderResponse:
    try:
        result = subprocess.run(command, input=stdin.encode("utf-8"), cwd=cwd, env=environment,
                                capture_output=True, timeout=request.timeout_seconds, check=False)
    except subprocess.TimeoutExpired as error:
        raw = RawProviderResponse(_text(error.stdout), _text(error.stderr), exit_code=-1)
        raise ProviderFailure("Subscription CLI timed out; no billing fallback", raw_response=raw) from error
    final_text = final_path.read_bytes().decode("utf-8") if final_path and final_path.is_file() else None
    return RawProviderResponse(_text(result.stdout), _text(result.stderr), result.returncode,
                               final_text=final_text, capture_method="official_account_cli")


def _check(command: list[str], environment: dict[str, str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, env=environment, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=30, check=False)


def json_member_text(text: str, keys: tuple[str, ...]) -> str:
    """Extract a JSON value's original lexical text without re-serializing it."""
    decoder = json.JSONDecoder()
    for wanted in keys:
        position = len(text) - len(text.lstrip())
        if text[position:position + 1] != "{":
            raise ValueError(f"Expected object containing {wanted}")
        position += 1
        while True:
            while position < len(text) and text[position].isspace():
                position += 1
            key, position = decoder.raw_decode(text, position)
            while text[position].isspace():
                position += 1
            if text[position] != ":":
                raise ValueError("Invalid JSON member separator")
            position += 1
            while text[position].isspace():
                position += 1
            start = position
            _, position = decoder.raw_decode(text, position)
            if key == wanted:
                text = text[start:position]
                break
            while text[position].isspace():
                position += 1
            if text[position] != ",":
                raise ValueError(f"Missing JSON member {wanted}")
            position += 1
    return text


class CodexSubscription:
    def __init__(self):
        self.executable = shutil.which("codex")
        if not self.executable:
            raise ProviderFailure("Codex CLI missing; ChatGPT account transport unavailable")

    def preflight(self) -> dict:
        with tempfile.TemporaryDirectory(prefix="qgcg_codex_auth_") as folder:
            result = _check([self.executable, "-c", 'forced_login_method="chatgpt"', "login", "status"],
                            account_environment(), Path(folder))
            if result.returncode or "logged in using chatgpt" not in (result.stdout + result.stderr).lower():
                raise ProviderFailure("Run codex login with ChatGPT; API-key login is not allowed")
            version = _check([self.executable, "--version"], account_environment(), Path(folder))
        if version.returncode:
            raise ProviderFailure("Cannot establish Codex CLI version")
        return {"authentication": "ChatGPT account", "cli_version": version.stdout.strip(),
                "binary_sha256": file_sha256(Path(self.executable))}

    def request(self, request: ProviderRequest) -> RawProviderResponse:
        with tempfile.TemporaryDirectory(prefix="qgcg_codex_", ignore_cleanup_errors=True) as folder:
            workspace = Path(folder)
            answer = workspace / "answer.txt"
            command = [self.executable, "--no-daemon", "exec", "--ignore-user-config", "--ephemeral",
                       "--skip-git-repo-check", "--sandbox", "read-only", "--cd", str(workspace),
                       "--model", request.model, "--color", "never",
                       "-c", 'forced_login_method="chatgpt"', "-c", 'model_provider="openai"',
                       "-c", 'approval_policy="never"', "-c", 'web_search="disabled"',
                       "-c", 'model_reasoning_effort="low"', "--output-last-message", str(answer)]
            for feature in ("shell_tool", "unified_exec", "apps", "browser_use", "computer_use",
                            "image_generation", "multi_agent", "hooks", "skill_search"):
                command.extend(["--disable", feature])
            command.append("-")
            stdin = (request.system_prompt + "\n\nUSER INPUT JSON:\n" + _user_input(request)
                     + "\nOUTPUT JSON SCHEMA:\n" + json_bytes(request.json_schema).decode("utf-8"))
            _record_request(request, command, stdin, {"authentication": "chatgpt", "reasoning_effort": "low",
                                                      "tools": "disabled", "api_fallback": False})
            return _run(request, command, cwd=workspace, environment=account_environment(),
                        stdin=stdin, final_path=answer)

    @staticmethod
    def decode(response: RawProviderResponse) -> DecodedResponse:
        match = re.search(r"^model:\s*(\S+)", response.stderr or "", re.MULTILINE)
        if not match or response.final_text is None:
            raise ProviderFailure("Codex did not return a final message and selected-model header; inspect saved stderr")
        return DecodedResponse(response.final_text, match.group(1), identity_source="official_cli_selected_model")


class ClaudeSubscription:
    def __init__(self, data_root: Path):
        tools = data_root / ".benchmark-tools"
        package = tools / "cli/node_modules/@anthropic-ai/claude-code"
        metadata = json.loads((package / "package.json").read_text(encoding="utf-8"))
        entry = metadata["bin"]
        script = (package / (entry["claude"] if isinstance(entry, dict) else entry)).resolve()
        if script.suffix == ".exe":
            self.command = [str(script)]
        else:
            node = next(tools.glob("node-*-win-x64/node.exe"), None)
            if node is None:
                raise ProviderFailure("Project Claude CLI requires its existing Node runtime")
            self.command = [str(node.resolve()), str(script)]
        self.profile = (tools / "accounts/claude").resolve()

    def environment(self) -> dict[str, str]:
        env = account_environment()
        env["CLAUDE_CONFIG_DIR"] = str(self.profile)
        env["NO_COLOR"] = "1"
        return env

    def preflight(self) -> dict:
        result = _check(self.command + ["auth", "status"], self.environment(), self.profile)
        state = json.loads(result.stdout)
        if result.returncode or not state.get("loggedIn") or state.get("authMethod") != "claude.ai":
            raise ProviderFailure("Claude must be signed in through its claude.ai subscription, not an API key")
        if state.get("subscriptionType") not in {"pro", "max"}:
            raise ProviderFailure("Claude account is not reported as a Pro/Max subscription")
        version = _check(self.command + ["--version"], self.environment(), self.profile)
        if version.returncode:
            raise ProviderFailure("Cannot establish Claude CLI version")
        return {"authentication": "claude.ai", "subscription_type": state["subscriptionType"],
                "cli_version": version.stdout.strip(), "binary_sha256": file_sha256(Path(self.command[-1]))}

    def request(self, request: ProviderRequest) -> RawProviderResponse:
        with tempfile.TemporaryDirectory(prefix="qgcg_claude_", ignore_cleanup_errors=True) as folder:
            workspace = Path(folder)
            settings = {"disableAllHooks": True, "forceLoginMethod": "claudeai"}
            settings_path = workspace / "settings.json"
            settings_path.write_bytes(json_bytes(settings))
            system = request.system_prompt + "\n\nOUTPUT JSON SCHEMA:\n" + json_bytes(request.json_schema).decode("utf-8")
            command = self.command + ["--safe-mode", "--restricted", "-p", "--model", request.model,
                       "--output-format", "json", "--json-schema", json_bytes(request.json_schema).decode("utf-8"),
                       "--tools", "", "--disallowedTools", "mcp__*",
                       "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                       "--setting-sources", "", "--settings", str(settings_path),
                       "--no-session-persistence", "--max-turns", "2", "--system-prompt", system]
            _artifact(request, "settings.json", json_bytes(settings))
            _record_request(request, command, _user_input(request),
                            {"authentication": "claude.ai", "profile": str(self.profile), "api_fallback": False,
                             "structured_output": "json_schema", "max_turns": 2,
                             "turn_note": "Reported num_turns also includes CLI-managed structured-output submissions/retries; external tools disabled"})
            return _run(request, command, cwd=workspace, environment=self.environment(), stdin=_user_input(request))

    @staticmethod
    def decode(response: RawProviderResponse) -> DecodedResponse:
        result = json.loads(response.raw_text)
        # --max-turns limits the main loop. The CLI's own structured-output
        # submissions/retries also count in num_turns, so it is not a second
        # enforcement of that limit. Preserve it in raw logs without rejecting
        # successful, schema-valid results because of the extra format turns.
        turns = result.get("num_turns")
        if result.get("is_error") or result.get("error") or type(turns) is not int or turns < 1:
            raise ProviderFailure("Claude account request failed or lacks valid turn metadata; inspect saved response")
        models = list(result.get("modelUsage", {}))
        if len(models) != 1 or not isinstance(result.get("structured_output"), dict):
            raise ProviderFailure("Claude response lacks exact model usage or schema-constrained output")
        # Extract the original JSON member lexically after its enclosing raw
        # provider response has been fsynced; never strip fences or reserialize.
        text = json_member_text(response.raw_text, ("structured_output",))
        return DecodedResponse(text, models[0], provider_request_id=result.get("session_id"),
                               identity_source="official_cli_model_usage")


class GeminiSubscription:
    def __init__(self, data_root: Path):
        executable = data_root / ".benchmark-tools/antigravity/agy.exe"
        if not executable.is_file():
            raise ProviderFailure("The existing Antigravity account CLI is unavailable")
        self.executable = str(executable.resolve())

    @staticmethod
    def isolated_profile(root: Path):
        profile = root / "profile"
        settings_path = profile / ".gemini/antigravity-cli/settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_bytes(json_bytes(GEMINI_SETTINGS))
        env = account_environment()
        env["USERPROFILE"] = str(profile)
        env["AGY_CLI_DISABLE_AUTO_UPDATE"] = "true"
        return profile, env

    def preflight(self) -> dict:
        with tempfile.TemporaryDirectory(prefix="qgcg_gemini_auth_", ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            _, env = self.isolated_profile(root)
            models = _check([self.executable, "models"], env, root)
            version = _check([self.executable, "--version"], env, root)
        if models.returncode or version.returncode or not models.stdout.strip():
            raise ProviderFailure("Gemini account model access failed; no API fallback")
        return {"authentication": "official account keyring", "cli_version": version.stdout.strip(),
                "binary_sha256": file_sha256(Path(self.executable)), "useG1Credits": False,
                "available_models": [line.split()[0] for line in models.stdout.splitlines() if line.strip()]}

    def request(self, request: ProviderRequest) -> RawProviderResponse:
        with tempfile.TemporaryDirectory(prefix="qgcg_gemini_", ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            profile, env = self.isolated_profile(root)
            workspace = root / "workspace"
            workspace.mkdir()
            agent = AGENT_HEADER + request.system_prompt + "\n"
            agent_path = profile / f".gemini/config/agents/{AGENT_NAME}.md"
            agent_path.parent.mkdir(parents=True)
            agent_path.write_text(agent, encoding="utf-8", newline="")
            inner = deepcopy(request.json_schema)
            definitions = inner.pop("$defs", {})
            schema = {"type": "object", "additionalProperties": False,
                      "properties": {"response": inner}, "required": ["response"], "$defs": definitions}
            schema_path = root / "schema.json"
            schema_path.write_bytes(json_bytes(schema))
            command = [self.executable, "--input-format", "stream-json", "--output-format", "stream-json",
                       "--model", request.model, "--agent", AGENT_NAME, "--disable-slash-commands",
                       "--json-schema", str(schema_path), "--print-timeout", f"{int(request.timeout_seconds)}s",
                       "--log-file", str(root / "cli.log")]
            stdin = json.dumps({"event": "user", "message": {"content": _user_input(request)}}, ensure_ascii=False) + "\n"
            _artifact(request, "agent.md", agent.encode("utf-8"))
            _artifact(request, "schema.json", json_bytes(schema))
            _record_request(request, command, stdin, GEMINI_SETTINGS)
            return self._stream(request, command, workspace, env, stdin)

    @staticmethod
    def _stream(request, command, workspace, env, stdin):
        # Persist each line BEFORE parsing the init/control events. Release the
        # source data only after model, workspace, and finish-only agent match.
        lines: list[str] = []
        deadline = time.monotonic() + request.timeout_seconds + 15
        with tempfile.TemporaryFile("w+b") as errors:
            process = subprocess.Popen(command, cwd=workspace, env=env, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=errors)
            events = queue.Queue()
            def reader():
                try:
                    for line in iter(process.stdout.readline, b""):
                        events.put(line)
                finally:
                    events.put(None)
            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            initialized = False
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ProviderFailure("Gemini subscription response timed out")
                    try:
                        raw = events.get(timeout=remaining)
                    except queue.Empty as error:
                        raise ProviderFailure("Gemini subscription response timed out") from error
                    if raw is None:
                        break
                    line = raw.decode("utf-8")
                    lines.append(line)
                    _capture(request, "provider_stream", line)
                    event = json.loads(line)
                    if not initialized:
                        init = event.get("init", {})
                        if (event.get("event") != "init" or init.get("model") != request.model
                                or init.get("agent") != AGENT_NAME or init.get("permission_mode") != "request-review"
                                or Path(init.get("cwd", "")).resolve() != workspace.resolve()):
                            raise ProviderFailure("Gemini startup identity/isolation mismatch; source input withheld")
                        initialized = True
                        process.stdin.write(stdin.encode("utf-8"))
                        process.stdin.close()
                code = process.wait(timeout=max(1, deadline - time.monotonic()))
                errors.seek(0)
                return RawProviderResponse("".join(lines), _text(errors.read()), code,
                                           capture_method="official_account_cli_stream")
            except Exception as error:
                errors.seek(0)
                partial = RawProviderResponse("".join(lines), _text(errors.read()), exit_code=-1)
                raise ProviderFailure(str(error), raw_response=partial) from error
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                thread.join(timeout=5)
                process.stdout.close()

    @staticmethod
    def decode(response: RawProviderResponse) -> DecodedResponse:
        initialized = None
        result = None
        for line in response.raw_text.splitlines():
            event = json.loads(line)
            kind = event.get("event")
            if result is not None:
                raise ProviderFailure("Gemini emitted unexpected events after the final result")
            if kind == "init" and initialized is None:
                initialized = event
            elif kind == "step_update" and initialized is not None:
                step = event.get("step_update", {})
                finish = step.get("tool_name") == "finish" and step.get("tool_info", {}).get("name", "finish") == "finish"
                if (step.get("subagent_info") or step.get("conversation_id") != initialized.get("conversation_id")
                        or (not finish and (step.get("step_type") not in {"user_input", "agent_response", "finish", "checkpoint"}
                                            or step.get("tool_name") or step.get("tool_info")))):
                    raise ProviderFailure("Gemini used unexpected external tools or conversation state")
            elif kind == "result" and initialized is not None:
                result = event["result"]
                if (result.get("status") != "SUCCESS" or result.get("num_turns") != 1
                        or result.get("conversation_id") != initialized.get("conversation_id")):
                    raise ProviderFailure("Gemini did not return a successful independent single-turn result")
                result_text = json_member_text(line, ("result", "structured_output", "response"))
            else:
                raise ProviderFailure("Unexpected Gemini CLI event")
        if initialized is None or result is None:
            raise ProviderFailure("Gemini stream has no verified final result")
        return DecodedResponse(result_text, initialized["init"]["model"],
                               provider_request_id=initialized.get("conversation_id"),
                               identity_source="official_cli_selected_model")


def subscription_providers(data_root: Path, *, gemini_model: str,
                           include_gemini: bool = True) -> dict[str, ProviderSpec]:
    """Read-only account/version preflight; no inference until the runner calls request."""
    clients = {"model_a": CodexSubscription(), "model_b": ClaudeSubscription(data_root)}
    if include_gemini:
        clients["model_c"] = GeminiSubscription(data_root)
    models = {"model_a": "gpt-5.6-sol", "model_b": "claude-sonnet-5", "model_c": gemini_model}
    specs = {}
    for role, client in clients.items():
        state = client.preflight()
        if role == "model_c" and gemini_model not in state["available_models"]:
            error = ProviderFailure(f"Gemini subscription does not expose {gemini_model}; no automatic model substitution")
            error.preflight_details = {"role": role, "requested_model": gemini_model, **state}
            raise error
        state.update(api_fallback=False, capture_boundary="official_cli_stdout_stderr_and_final_text",
                     internal_http_retry_visibility="only retries exposed in CLI diagnostics",
                     backend_revision_exposed=False)
        specs[role] = ProviderSpec(models[role], client.request, client.decode, type(client).__name__,
                                   state["cli_version"], settings=state,
                                   implementation_files=(Path(__file__).resolve(),), retry_invalid_output=True)
    return specs
