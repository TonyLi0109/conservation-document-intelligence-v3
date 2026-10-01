"""Generate page candidates with three frontier LLM systems.

The corpus is too large to send in one request reliably. This runner packs pages
into bounded prompts, asks each model to select candidates from every batch, and
recursively reduces the candidate set until a final global top-k ranking fits.

Live example:
    python scripts/run_llm_retrievers.py

Protocol-only smoke test without API keys:
    python scripts/run_llm_retrievers.py --mock

Model defaults can be overridden with LLM_A_MODEL, LLM_B_MODEL, and LLM_C_MODEL
or with repeated --model SYSTEM=MODEL arguments.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
import json
import math
import logging
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Protocol
import sys

from dotenv import load_dotenv

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
load_dotenv(REPOSITORY_ROOT / ".env.local", override=False)

from scripts.run_conventional_retrievers import Query, load_queries


LOGGER = logging.getLogger("run_llm_retrievers")
DEFAULT_CORPUS = REPOSITORY_ROOT / "benchmark" / "corpus_pages.jsonl"
DEFAULT_QUESTIONS = REPOSITORY_ROOT / "benchmark" / "questions.csv"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "benchmark" / "candidate_runs"
DEFAULT_TRACE_DIR = REPOSITORY_ROOT / "benchmark" / "llm_traces"
TOP_K = 20
DEFAULT_PROMPT_BUDGET = 90_000
DEFAULT_PAGE_CHARS = 6_000
DEFAULT_TIMEOUT = 120.0
DEFAULT_RETRIES = 3
DEFAULT_MODELS = {
    "llm_a": os.getenv("LLM_A_MODEL", "openai/gpt-5.6-sol"),
    "llm_b": os.getenv("LLM_B_MODEL", "anthropic/claude-sonnet-5"),
    "llm_c": os.getenv("LLM_C_MODEL", "gemini/gemini-3.1-pro-preview"),
    "gpt_6_luna": os.getenv("GPT_6_LUNA_MODEL", "openai/gpt-6-luna"),
    "claude_haiku": os.getenv("CLAUDE_HAIKU_MODEL", "anthropic/claude-haiku-4-5-20251001"),
    "gemini_flash": os.getenv("GEMINI_FLASH_MODEL", "gemini/gemini-3.7-flash"),
}
PAGE_ID_PATTERN = re.compile(r"\bDOC\d{3}_P\d{4}\b", re.IGNORECASE)
_CANDIDATE_RETRIEVAL_PROMPT_TEMPLATE = """
You are an expert evidence retrieval system for a Missouri conservation document corpus. Your task is to identify and rank exactly __TOP_K__ page-level evidence items that support answering the user's query.

**STRICT RULES:**

1. **Corpus Only:** Use ONLY the provided `corpus_pages` text. Do not use outside knowledge.
2. **Direct Evidence:** Select pages that contain concrete information, data, or explicit statements useful for constructing an answer, not just pages that mention the query keywords.
3. **Scope Constraints:** Strictly preserve geographic and temporal scope. Do not transfer external/national examples into a Missouri context, and do not infer missing publication dates.
4. **Exactly __TOP_K__ Pages:** You MUST output EXACTLY __TOP_K__ candidate pages. Under no circumstances should you output fewer than __TOP_K__. If you cannot find __TOP_K__ highly relevant pages, you MUST append marginally relevant, contextual, or tangentially related pages until you reach exactly __TOP_K__ candidates. Do not stop early.
5. **Output Format:** You MUST output ONLY a valid JSON array of objects. Do not include markdown code blocks (like ```json), conversational text, or explanations outside the JSON.

**Input Query:**
{query}

**Available Corpus Pages:**
{corpus_context}

**JSON Output Schema (Array of Objects):**
[
{
"page_id": "DOC036_P0106",
"doc_id": "DOC036",
"pdf_page": 106,
"rationale": "Briefly extract the specific sentence or data point that answers the query.",
"supported_facets": ["Briefly list 1-2 aspects of the query this page answers"]
}
]
"""

CANDIDATE_RETRIEVAL_PROMPT = _CANDIDATE_RETRIEVAL_PROMPT_TEMPLATE.replace(
    "__TOP_K__", str(TOP_K)
)


def configure_top_k(top_k: int) -> None:
    """Set an isolated run depth and regenerate the exact-count prompt."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    global TOP_K, CANDIDATE_RETRIEVAL_PROMPT
    TOP_K = top_k
    CANDIDATE_RETRIEVAL_PROMPT = _CANDIDATE_RETRIEVAL_PROMPT_TEMPLATE.replace(
        "__TOP_K__", str(top_k)
    )

# Hierarchical retrieval uses smaller, disjoint batches before the final global
# top-20 call. Asking those internal calls for 20 items can be impossible (a
# batch may contain fewer than 20 pages) and can prevent the pool from
# shrinking. This instruction is used only for internal reduction; the final
# candidate run always uses CANDIDATE_RETRIEVAL_PROMPT above.
BATCH_REDUCTION_DIRECTIVE = """

**Internal Reduction Instruction:**
This is an internal candidate-reduction batch. Return EXACTLY {target} distinct
candidate page objects from the pages supplied in this batch. Do not stop early.
"""


class LLMRetrievalError(RuntimeError):
    """Raised when an LLM candidate run cannot be completed safely."""


@dataclass(frozen=True)
class CorpusPage:
    page_id: str
    doc_id: str
    pdf_page: int
    page_text: str


class CompletionClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: float,
        max_tokens: int,
    ) -> str:
        """Return the model's response text."""


class LiteLLMClient:
    """Small adapter around LiteLLM's provider-neutral completion API."""

    def __init__(self) -> None:
        self._last_request_at: dict[str, float] = {}

    def _throttle_model(self, model: str) -> None:
        """Apply provider-specific spacing without changing other model calls."""
        normalized = model.casefold()
        settings = (
            ("gpt-6-luna", "GPT_6_LUNA_MIN_INTERVAL", 7.5),
            ("claude-haiku-4-5", "CLAUDE_HAIKU_MIN_INTERVAL", 1.0),
            ("gemini-3.7-flash", "GEMINI_FLASH_MIN_INTERVAL", 2.5),
        )
        matched = next((item for item in settings if item[0] in normalized), None)
        if matched is None:
            return
        key, environment_name, default = matched
        interval = float(os.getenv(environment_name, str(default)))
        if interval < 0:
            raise ValueError(f"{environment_name} cannot be negative")
        now = time.monotonic()
        previous = self._last_request_at.get(key)
        if previous is not None:
            time.sleep(max(0.0, previous + interval - now))
        self._last_request_at[key] = time.monotonic()

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: float,
        max_tokens: int,
    ) -> str:
        try:
            import litellm
        except ImportError as error:
            raise LLMRetrievalError(
                "LiteLLM is required for live calls. Install requirements.txt "
                "or run with --mock."
            ) from error

        messages = [{"role": "user", "content": user_prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        self._throttle_model(model)
        response = litellm.completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            timeout=timeout,
            num_retries=0,
        )
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise LLMRetrievalError("LLM response did not contain message content") from error
        return coerce_response_text(content)


class MockLLMClient:
    """Deterministic protocol mock; it does not represent frontier retrieval."""

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: float,
        max_tokens: int,
    ) -> str:
        del model, system_prompt, timeout, max_tokens
        try:
            context = user_prompt.split(
                "**Available Corpus Pages:**\n", 1
            )[1].split("\n\n**JSON Output Schema", 1)[0]
            records = [
                json.loads(line) for line in context.splitlines() if line.strip()
            ]
        except (IndexError, json.JSONDecodeError) as error:
            raise LLMRetrievalError("Mock could not parse corpus context") from error
        return json.dumps(
            [
                {
                    "page_id": item["page_id"],
                    "doc_id": item["doc_id"],
                    "pdf_page": item["pdf_page"],
                    "rationale": "Mock protocol result.",
                    "supported_facets": ["mock"],
                }
                for item in records[:TOP_K]
            ]
        )


def coerce_response_text(content: Any) -> str:
    """Normalize string or content-block responses from provider SDKs."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(getattr(block, "text", None), str):
                parts.append(block.text)
        if parts:
            return "\n".join(parts)
    raise LLMRetrievalError("LLM response content was not text")


def load_corpus_pages(path: Path) -> list[CorpusPage]:
    """Load required page fields and reject malformed or duplicate records."""

    if not path.is_file():
        raise LLMRetrievalError(f"Corpus JSONL does not exist: {path}")
    pages: list[CorpusPage] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise LLMRetrievalError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
            page_id = item.get("page_id")
            doc_id = item.get("doc_id")
            pdf_page = item.get("pdf_page")
            page_text = item.get("page_text")
            if (
                not isinstance(page_id, str)
                or not PAGE_ID_PATTERN.fullmatch(page_id)
                or not isinstance(doc_id, str)
                or type(pdf_page) is not int
                or pdf_page < 1
                or not isinstance(page_text, str)
            ):
                raise LLMRetrievalError(
                    f"Malformed page record at {path}:{line_number}"
                )
            if page_id in seen:
                raise LLMRetrievalError(f"Duplicate page_id: {page_id}")
            seen.add(page_id)
            pages.append(
                CorpusPage(
                    page_id=page_id,
                    doc_id=doc_id,
                    pdf_page=pdf_page,
                    page_text=page_text,
                )
            )
    if not pages:
        raise LLMRetrievalError(f"Corpus contains no pages: {path}")
    return pages


def render_page(page: CorpusPage, max_page_chars: int) -> str:
    """Render bounded page metadata and text for a selection prompt."""

    if max_page_chars < 1:
        raise ValueError("max_page_chars must be positive")
    text = page.page_text
    truncated = len(text) > max_page_chars
    if truncated:
        text = text[:max_page_chars]
    payload = {
        "page_id": page.page_id,
        "doc_id": page.doc_id,
        "pdf_page": page.pdf_page,
        "page_text": text,
    }
    if truncated:
        payload["text_truncated"] = True
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def pack_pages(
    pages: list[CorpusPage],
    *,
    prompt_budget: int,
    max_page_chars: int,
) -> list[list[CorpusPage]]:
    """Pack pages deterministically within the approximate character budget."""

    if prompt_budget < 2_000:
        raise ValueError("prompt_budget must be at least 2000 characters")
    rendered = [(page, render_page(page, max_page_chars)) for page in pages]
    batches: list[list[CorpusPage]] = []
    current: list[CorpusPage] = []
    current_chars = 0
    for page, text in rendered:
        size = len(text) + 1
        if current and current_chars + size > prompt_budget:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(page)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def _extract_json_value(raw: str) -> Any | None:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def parse_page_ids(
    raw: str,
    *,
    allowed_ids: set[str],
    limit: int,
) -> list[str]:
    """Parse JSON first, then recover page IDs from imperfect model output."""

    if limit < 1:
        raise ValueError("limit must be positive")
    value = _extract_json_value(raw)
    candidates: list[Any] = []
    if isinstance(value, list):
        candidates = value
    elif isinstance(value, dict):
        for key in ("page_ids", "results", "candidates", "pages"):
            if isinstance(value.get(key), list):
                candidates = value[key]
                break

    recovered: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, str):
            recovered.append(candidate)
        elif isinstance(candidate, dict) and isinstance(candidate.get("page_id"), str):
            recovered.append(candidate["page_id"])
    if not recovered:
        recovered = PAGE_ID_PATTERN.findall(raw)

    output: list[str] = []
    seen: set[str] = set()
    for candidate in recovered:
        page_id = candidate.upper()
        if page_id in allowed_ids and page_id not in seen:
            output.append(page_id)
            seen.add(page_id)
            if len(output) == limit:
                break
    return output


SYSTEM_PROMPT = ""


def build_user_prompt(
    query: Query,
    pages: list[CorpusPage],
    *,
    max_page_chars: int,
    target: int = TOP_K,
    final_selection: bool = True,
) -> str:
    """Render the standardized retrieval prompt without rescanning inserted text."""

    corpus_context = "\n".join(
        render_page(page, max_page_chars) for page in pages
    )
    values = {
        "query": query.text,
        "corpus_context": corpus_context,
    }
    prompt = re.sub(
        r"\{(query|corpus_context)\}",
        lambda match: values[match.group(1)],
        CANDIDATE_RETRIEVAL_PROMPT,
    )
    if not final_selection:
        final_directive = (
            f"4. **Exactly {TOP_K} Pages:** You MUST output EXACTLY {TOP_K} candidate pages. "
            f"Under no circumstances should you output fewer than {TOP_K}. If you cannot "
            f"find {TOP_K} highly relevant pages, you MUST append marginally relevant, "
            f"contextual, or tangentially related pages until you reach exactly {TOP_K} "
            "candidates. Do not stop early."
        )
        prompt = prompt.replace(
            final_directive,
            BATCH_REDUCTION_DIRECTIVE.format(target=target).strip(),
        )
    return prompt


def warn_short_response(*, model: str, query_id: str, actual: int, expected: int) -> None:
    """Emit the required conspicuous warning for an early-stopped response."""

    LOGGER.warning(
        "\033[1mWARNING: Model %s returned only %d candidates for Query %s. "
        "Expected exactly %d.\033[0m",
        model,
        actual,
        query_id,
        expected,
    )


def append_trace(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        output.write("\n")


def request_candidates(
    *,
    client: CompletionClient,
    model: str,
    system: str,
    query: Query,
    pages: list[CorpusPage],
    target: int,
    max_page_chars: int,
    timeout: float,
    retries: int,
    trace_path: Path | None,
    pass_number: int,
    batch_number: int,
    allow_empty: bool = False,
    final_selection: bool = False,
) -> list[str]:
    """Call one model batch and incrementally repair early-stopped output."""

    del allow_empty  # Exact-count validation supersedes the legacy empty-batch mode.
    allowed = {page.page_id for page in pages}
    prompt = build_user_prompt(
        query,
        pages,
        max_page_chars=max_page_chars,
        target=target,
        final_selection=final_selection,
    )
    collected: list[str] = []
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        raw = ""
        remaining = target - len(collected)
        attempt_prompt = prompt
        if collected:
            selected = set(collected)
            repair_pages = [page for page in pages if page.page_id not in selected]
            attempt_prompt = build_user_prompt(
                query,
                repair_pages,
                max_page_chars=max_page_chars,
                target=remaining,
                final_selection=False,
            )
            attempt_prompt += (
                "\n\n**MANDATORY EARLY-STOP REPAIR:**\n"
                f"The prior response supplied {len(collected)} of {target} required "
                f"distinct pages. The already selected pages have been removed. "
                f"Return EXACTLY {remaining} additional candidate page objects "
                "from the remaining pages above. Output only the JSON array."
            )
        try:
            raw = client.complete(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=attempt_prompt,
                timeout=timeout,
                max_tokens=max(4096, target * 128),
            )
            parsed = parse_page_ids(
                raw,
                allowed_ids=allowed.difference(collected),
                limit=remaining,
            )
            collected.extend(parsed)
            if len(collected) == target:
                append_trace(
                    trace_path,
                    {
                        "system": system,
                        "model": model,
                        "query_id": query.query_id,
                        "pass": pass_number,
                        "batch": batch_number,
                        "attempt": attempt,
                        "candidate_count": len(pages),
                        "requested": target,
                        "response": raw,
                        "parsed_page_ids": parsed,
                        "accumulated_page_ids": collected,
                        "error": None,
                    },
                )
                return collected

            warn_short_response(
                model=model,
                query_id=query.query_id,
                actual=len(collected),
                expected=target,
            )
            last_error = LLMRetrievalError(
                f"Response accumulation reached {len(collected)} distinct valid "
                f"candidates; expected exactly {target}"
            )
            append_trace(
                trace_path,
                {
                    "system": system,
                    "model": model,
                    "query_id": query.query_id,
                    "pass": pass_number,
                    "batch": batch_number,
                    "attempt": attempt,
                    "candidate_count": len(pages),
                    "requested": target,
                    "response": raw,
                    "parsed_page_ids": parsed,
                    "accumulated_page_ids": collected,
                    "error": f"{type(last_error).__name__}: {last_error}",
                },
            )
        except Exception as error:
            last_error = error
            append_trace(
                trace_path,
                {
                    "system": system,
                    "model": model,
                    "query_id": query.query_id,
                    "pass": pass_number,
                    "batch": batch_number,
                    "attempt": attempt,
                    "candidate_count": len(pages),
                    "requested": target,
                    "response": raw,
                    "parsed_page_ids": [],
                    "accumulated_page_ids": collected,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        if attempt < retries:
            time.sleep(2 ** (attempt - 1))

    raise LLMRetrievalError(
        f"{system}/{query.query_id} failed after {retries} attempts: {last_error}"
    ) from last_error


def request_candidates_with_split(
    *,
    client: CompletionClient,
    model: str,
    system: str,
    query: Query,
    pages: list[CorpusPage],
    target: int,
    max_page_chars: int,
    timeout: float,
    retries: int,
    trace_path: Path | None,
    pass_number: int,
    batch_number: int,
    final_selection: bool = False,
) -> list[str]:
    """Request an exact quota, splitting a persistently blank large batch."""

    try:
        return request_candidates(
            client=client,
            model=model,
            system=system,
            query=query,
            pages=pages,
            target=target,
            max_page_chars=max_page_chars,
            timeout=timeout,
            retries=retries,
            trace_path=trace_path,
            pass_number=pass_number,
            batch_number=batch_number,
            final_selection=final_selection,
        )
    except LLMRetrievalError as error:
        # Splitting can repair a structurally short/blank response. Provider
        # failures (authentication, billing, quotas, timeouts) must propagate
        # unchanged instead of multiplying the same failed API call.
        if "Response accumulation reached" not in str(error):
            raise
        if len(pages) <= target or len(pages) < 2:
            raise
        midpoint = len(pages) // 2
        partitions = (pages[:midpoint], pages[midpoint:])
        left_target = min(len(partitions[0]), max(1, target // 2))
        right_target = target - left_target
        if right_target > len(partitions[1]):
            shift = right_target - len(partitions[1])
            left_target += shift
            right_target -= shift
        quotas = (left_target, right_target)
        LOGGER.warning(
            "Splitting failed %s/%s batch %d (%d pages) into quotas %d + %d",
            system,
            query.query_id,
            batch_number,
            len(pages),
            left_target,
            right_target,
        )
        output: list[str] = []
        for split_index, (partition, quota) in enumerate(
            zip(partitions, quotas), start=1
        ):
            if quota == 0:
                continue
            output.extend(
                request_candidates_with_split(
                    client=client,
                    model=model,
                    system=system,
                    query=query,
                    pages=partition,
                    target=quota,
                    max_page_chars=max_page_chars,
                    timeout=timeout,
                    retries=retries,
                    trace_path=trace_path,
                    pass_number=pass_number,
                    batch_number=batch_number * 10 + split_index,
                    final_selection=False,
                )
            )
        if len(output) != target:
            raise LLMRetrievalError(
                f"{system}/{query.query_id} split repair returned {len(output)}; "
                f"expected {target}"
            )
        return output


def generate_query_candidates(
    *,
    client: CompletionClient,
    model: str,
    system: str,
    query: Query,
    corpus: list[CorpusPage],
    prompt_budget: int,
    max_page_chars: int,
    timeout: float,
    retries: int,
    trace_path: Path | None,
) -> list[str]:
    """Recursively reduce the corpus and return a globally ordered top 20."""

    page_by_id = {page.page_id: page for page in corpus}
    current = list(corpus)
    pass_number = 0
    while True:
        pass_number += 1
        if pass_number > 20:
            raise LLMRetrievalError(
                f"{system}/{query.query_id} exceeded 20 reduction passes"
            )

        if len(current) <= TOP_K:
            overhead = 2_000
            per_page = max(
                200,
                min(max_page_chars, (prompt_budget - overhead) // max(1, len(current))),
            )
            return request_candidates_with_split(
                client=client,
                model=model,
                system=system,
                query=query,
                pages=current,
                target=len(current),
                max_page_chars=per_page,
                timeout=timeout,
                retries=retries,
                trace_path=trace_path,
                pass_number=pass_number,
                batch_number=1,
                final_selection=True,
            )[:TOP_K]

        batches = pack_pages(
            current,
            prompt_budget=prompt_budget,
            max_page_chars=max_page_chars,
        )
        selected_ids: list[str] = []
        # Internal batches are a recall-oriented shortlist, not the final query
        # output. Allocate enough candidates across all batches to feed the
        # final exact-20 reduction without forcing every corpus shard to emit
        # 20 pages. This avoids dozens of unnecessary repair calls on large
        # corpora while retaining candidates from every disjoint shard.
        per_batch_target = max(2, math.ceil(TOP_K / len(batches)))
        batch_targets = [min(len(batch), per_batch_target) for batch in batches]
        deficit = max(0, TOP_K - sum(batch_targets))
        while deficit:
            for index, batch in enumerate(batches):
                if batch_targets[index] < len(batch):
                    batch_targets[index] += 1
                    deficit -= 1
                    if deficit == 0:
                        break

        for batch_number, (batch, target) in enumerate(
            zip(batches, batch_targets), start=1
        ):
            selected_ids.extend(
                request_candidates_with_split(
                    client=client,
                    model=model,
                    system=system,
                    query=query,
                    pages=batch,
                    target=target,
                    max_page_chars=max_page_chars,
                    timeout=timeout,
                    retries=retries,
                    trace_path=trace_path,
                    pass_number=pass_number,
                    batch_number=batch_number,
                    final_selection=len(batches) == 1,
                )
            )

        selected_ids = list(dict.fromkeys(selected_ids))
        if not selected_ids:
            raise LLMRetrievalError(
                f"{system}/{query.query_id} reduction produced no candidates"
            )
        if len(selected_ids) >= len(current):
            raise LLMRetrievalError(
                f"{system}/{query.query_id} candidate reduction did not shrink"
            )
        current = [page_by_id[page_id] for page_id in selected_ids]
        if len(batches) == 1:
            ranking = selected_ids[:TOP_K]
            if len(ranking) < TOP_K:
                warn_short_response(
                    model=model,
                    query_id=query.query_id,
                    actual=len(ranking),
                    expected=TOP_K,
                )
                raise LLMRetrievalError(
                    f"{system}/{query.query_id} final ranking has {len(ranking)} "
                    f"candidates; expected exactly {TOP_K}"
                )
            return ranking


def write_run(
    path: Path,
    queries: list[Query],
    rankings: dict[str, list[str]],
    *,
    system: str,
) -> int:
    """Atomically write one frontier LLM candidate run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for query in queries:
                page_ids = rankings.get(query.query_id)
                if not page_ids:
                    raise LLMRetrievalError(
                        f"Missing candidates for query {query.query_id}"
                    )
                if len(page_ids) != TOP_K:
                    warn_short_response(
                        model=system,
                        query_id=query.query_id,
                        actual=len(page_ids),
                        expected=TOP_K,
                    )
                    raise LLMRetrievalError(
                        f"Refusing to write {system}/{query.query_id}: got "
                        f"{len(page_ids)} candidates; expected exactly {TOP_K}"
                    )
                for rank, page_id in enumerate(page_ids, start=1):
                    output.write(
                        json.dumps(
                            {
                                "query_id": query.query_id,
                                "page_id": page_id,
                                "rank": rank,
                                "system": system,
                            },
                            separators=(",", ":"),
                        )
                    )
                    output.write("\n")
                    count += 1
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return count


def build_run_signature(
    *,
    model: str,
    system: str,
    queries: list[Query],
    corpus: list[CorpusPage],
    prompt_budget: int,
    max_page_chars: int,
) -> str:
    """Fingerprint every input that can change a candidate run."""

    digest = hashlib.sha256()
    header = {
        "schema": 1,
        "model": model,
        "system": system,
        "queries": [[query.query_id, query.text] for query in queries],
        "prompt_budget": prompt_budget,
        "max_page_chars": max_page_chars,
        "prompt": CANDIDATE_RETRIEVAL_PROMPT,
    }
    digest.update(json.dumps(header, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for page in corpus:
        digest.update(b"\0")
        digest.update(page.page_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(page.page_text.encode("utf-8"))
    return digest.hexdigest()


def _valid_ranking(page_ids: Any, allowed_ids: set[str]) -> list[str] | None:
    expected = min(TOP_K, len(allowed_ids))
    if not isinstance(page_ids, list) or len(page_ids) != expected:
        return None
    normalized: list[str] = []
    for value in page_ids:
        if not isinstance(value, str):
            return None
        page_id = value.upper()
        if page_id not in allowed_ids or page_id in normalized:
            return None
        normalized.append(page_id)
    return normalized


def write_checkpoint(
    path: Path,
    *,
    signature: str,
    model: str,
    system: str,
    rankings: dict[str, list[str]],
) -> None:
    """Atomically persist completed queries so paid work survives interruption."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "signature": signature,
        "model": model,
        "system": system,
        "rankings": rankings,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    *,
    signature: str,
    model: str,
    system: str,
    query_ids: set[str],
    allowed_ids: set[str],
) -> dict[str, list[str]]:
    """Load only a checkpoint proven to match the current run inputs."""

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise LLMRetrievalError(f"Unreadable checkpoint {path}: {error}") from error
    if (
        payload.get("schema_version") != 1
        or payload.get("signature") != signature
        or payload.get("model") != model
        or payload.get("system") != system
    ):
        LOGGER.warning("Ignoring stale checkpoint with a different run signature: %s", path)
        return {}
    rankings: dict[str, list[str]] = {}
    raw_rankings = payload.get("rankings")
    if not isinstance(raw_rankings, dict):
        raise LLMRetrievalError(f"Malformed checkpoint rankings: {path}")
    for query_id, page_ids in raw_rankings.items():
        ranking = _valid_ranking(page_ids, allowed_ids)
        if query_id in query_ids and ranking is not None:
            rankings[query_id] = ranking
    return rankings


def load_existing_run(
    path: Path,
    *,
    system: str,
    query_ids: set[str],
    allowed_ids: set[str],
) -> dict[str, list[str]]:
    """Recover completed query rankings from an existing candidate run."""

    if not path.is_file():
        return {}
    grouped: dict[str, list[tuple[int, str]]] = {}
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise LLMRetrievalError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if record.get("system") != system:
                continue
            query_id = record.get("query_id")
            page_id = record.get("page_id")
            rank = record.get("rank")
            if query_id in query_ids and isinstance(page_id, str) and type(rank) is int:
                grouped.setdefault(query_id, []).append((rank, page_id.upper()))
    rankings: dict[str, list[str]] = {}
    for query_id, rows in grouped.items():
        rows.sort()
        if [rank for rank, _ in rows] != list(range(1, len(rows) + 1)):
            continue
        ranking = _valid_ranking([page_id for _, page_id in rows], allowed_ids)
        if ranking is not None:
            rankings[query_id] = ranking
    return rankings


def recover_rankings_from_trace(
    path: Path,
    *,
    model: str,
    system: str,
    query_ids: set[str],
    allowed_ids: set[str],
) -> dict[str, list[str]]:
    """Conservatively recover legacy traces whose next query proves completion."""

    if not path.is_file():
        return {}
    last_success: dict[str, list[str]] = {}
    explicitly_complete: dict[str, list[str]] = {}
    completed_by_transition: set[str] = set()
    previous_query: str | None = None
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed trace line %s:%d", path, line_number)
                continue
            if record.get("system") != system or record.get("model") != model:
                continue
            query_id = record.get("query_id")
            if query_id not in query_ids:
                continue
            if previous_query is not None and query_id != previous_query:
                completed_by_transition.add(previous_query)
            previous_query = query_id
            if record.get("event") == "query_complete":
                ranking = _valid_ranking(record.get("ranked_page_ids"), allowed_ids)
                if ranking is not None:
                    explicitly_complete[query_id] = ranking
                continue
            if record.get("error") is None:
                ranking = _valid_ranking(record.get("parsed_page_ids"), allowed_ids)
                if ranking is not None:
                    last_success[query_id] = ranking

    recovered = dict(explicitly_complete)
    for query_id in completed_by_transition:
        ranking = last_success.get(query_id)
        if ranking is not None:
            recovered.setdefault(query_id, ranking)
    return recovered


def run_system(
    *,
    client: CompletionClient,
    model: str,
    system: str,
    queries: list[Query],
    corpus: list[CorpusPage],
    output_dir: Path,
    trace_dir: Path | None,
    prompt_budget: int,
    max_page_chars: int,
    timeout: float,
    retries: int,
    resume: bool = False,
) -> int:
    trace_path = trace_dir / f"{system}.jsonl" if trace_dir else None
    output_path = output_dir / f"{system}.jsonl"
    checkpoint_path = output_dir / ".checkpoints" / f"{system}.json"
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_ids = {query.query_id for query in queries}
    allowed_ids = {page.page_id for page in corpus}
    signature = build_run_signature(
        model=model,
        system=system,
        queries=queries,
        corpus=corpus,
        prompt_budget=prompt_budget,
        max_page_chars=max_page_chars,
    )
    rankings: dict[str, list[str]] = {}
    if resume:
        rankings.update(
            load_checkpoint(
                checkpoint_path,
                signature=signature,
                model=model,
                system=system,
                query_ids=query_ids,
                allowed_ids=allowed_ids,
            )
        )
        for query_id, ranking in load_existing_run(
            output_path,
            system=system,
            query_ids=query_ids,
            allowed_ids=allowed_ids,
        ).items():
            rankings.setdefault(query_id, ranking)
        if trace_path:
            recovered = recover_rankings_from_trace(
                trace_path,
                model=model,
                system=system,
                query_ids=query_ids,
                allowed_ids=allowed_ids,
            )
            for query_id, ranking in recovered.items():
                rankings.setdefault(query_id, ranking)
        if rankings:
            LOGGER.info(
                "%s: resumed %d/%d completed queries: %s",
                system,
                len(rankings),
                len(queries),
                ", ".join(query.query_id for query in queries if query.query_id in rankings),
            )
            write_checkpoint(
                checkpoint_path,
                signature=signature,
                model=model,
                system=system,
                rankings=rankings,
            )
    else:
        if trace_path:
            trace_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)

    for index, query in enumerate(queries, start=1):
        if query.query_id in rankings:
            LOGGER.info(
                "%s (%s): query %d/%d %s [checkpoint]",
                system,
                model,
                index,
                len(queries),
                query.query_id,
            )
            continue
        LOGGER.info(
            "%s (%s): query %d/%d %s",
            system,
            model,
            index,
            len(queries),
            query.query_id,
        )
        ranking = generate_query_candidates(
            client=client,
            model=model,
            system=system,
            query=query,
            corpus=corpus,
            prompt_budget=prompt_budget,
            max_page_chars=max_page_chars,
            timeout=timeout,
            retries=retries,
            trace_path=trace_path,
        )
        if len(ranking) != TOP_K:
            warn_short_response(
                model=model,
                query_id=query.query_id,
                actual=len(ranking),
                expected=TOP_K,
            )
            raise LLMRetrievalError(
                f"{system}/{query.query_id} returned {len(ranking)} candidates; "
                f"expected exactly {TOP_K}"
            )
        rankings[query.query_id] = ranking
        append_trace(
            trace_path,
            {
                "event": "query_complete",
                "system": system,
                "model": model,
                "query_id": query.query_id,
                "ranked_page_ids": ranking,
                "error": None,
            },
        )
        write_checkpoint(
            checkpoint_path,
            signature=signature,
            model=model,
            system=system,
            rankings=rankings,
        )

    return write_run(output_path, queries, rankings, system=system)

def parse_model_overrides(values: list[str]) -> dict[str, str]:
    models = dict(DEFAULT_MODELS)
    for value in values:
        system, separator, model = value.partition("=")
        if not separator or system not in models or not model.strip():
            raise ValueError(
                "--model must be SYSTEM=MODEL where SYSTEM is a configured retrieval system"
            )
        models[system] = model.strip()
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate top-k page candidates with configured frontier LLMs."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="SYSTEM=MODEL",
        help="Override a model; may be repeated.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=tuple(DEFAULT_MODELS),
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument("--prompt-budget", type=int, default=DEFAULT_PROMPT_BUDGET)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-page-chars", type=int, default=DEFAULT_PAGE_CHARS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Exercise orchestration without API calls; outputs are not benchmark results.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed queries from checkpoints, candidate runs, or traces.",
    )
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
        if args.retries < 1:
            raise ValueError("retries must be positive")
        configure_top_k(args.top_k)
        models = parse_model_overrides(args.model)
        corpus = load_corpus_pages(args.corpus)
        queries = load_queries(args.questions)
        client: CompletionClient = MockLLMClient() if args.mock else LiteLLMClient()
        if args.mock:
            LOGGER.warning(
                "Mock mode validates orchestration only; generated files are not "
                "frontier LLM benchmark results."
            )
        for system in args.systems:
            count = run_system(
                client=client,
                model=models[system],
                system=system,
                queries=queries,
                corpus=corpus,
                output_dir=args.output_dir.resolve(),
                trace_dir=args.trace_dir.resolve() if args.trace_dir else None,
                prompt_budget=args.prompt_budget,
                max_page_chars=args.max_page_chars,
                timeout=args.timeout,
                retries=args.retries,
                resume=args.resume,
            )
            LOGGER.info("Wrote %d records to %s.jsonl", count, system)
    except (OSError, ValueError, LLMRetrievalError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
