"""Run three independent, blind page-level relevance judges.

Each judge sees only the query, canonical page ID, catalog title, and frozen page
text. Candidate provenance, retrieval ranks, and selection counts are discarded
before prompt construction.

Example:
    python scripts/run_llm_judges.py
    python scripts/run_llm_judges.py --mock
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Protocol

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
load_dotenv(REPOSITORY_ROOT / ".env.local", override=False)

from scripts.run_conventional_retrievers import Query, load_queries


LOGGER = logging.getLogger("run_llm_judges")
DEFAULT_POOL = REPOSITORY_ROOT / "benchmark" / "pools" / "candidate_pool.jsonl"
DEFAULT_QUESTIONS = REPOSITORY_ROOT / "benchmark" / "questions.csv"
DEFAULT_CORPUS = REPOSITORY_ROOT / "benchmark" / "corpus_pages.jsonl"
DEFAULT_CATALOG = REPOSITORY_ROOT / "data" / "source_catalog.csv"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "benchmark" / "judgments"
DEFAULT_MODELS = {
    "judge_a": os.getenv("JUDGE_A_MODEL", "openai/gpt-5.6-sol"),
    "judge_b": os.getenv("JUDGE_B_MODEL", "anthropic/claude-sonnet-5"),
    "judge_c": os.getenv("JUDGE_C_MODEL", "gemini/gemini-3.1-pro-preview"),
}
PAGE_ID_PATTERN = re.compile(r"^DOC\d{3}_P\d{4}$")
CandidateKey = tuple[str, str]

JUDGING_PROMPT_VERSION = "core-evidence-v2-2026-09-30"

JUDGE_SYSTEM_PROMPT = """You are judging whether a candidate PDF page provides evidence for answering a specific question.

Inputs:
- Question
- PAGE_ID
- Document title
- Complete extracted text of the candidate page

Judge the page against the FULL information requirements of the question, not only topical similarity or keyword overlap.

Use the following relevance grades:

4 = ESSENTIAL / CORE EVIDENCE
    The page contains evidence that is indispensable for answering at least one major required component of a complete, high-quality answer. A question may have multiple core pages when different pages support different indispensable components. Do NOT assume there should be only one best/core page.

3 = HIGHLY RELEVANT
    The page provides substantive evidence that clearly strengthens the answer, but the answer could still cover all essential components without this page.

2 = SUPPORTING
    The page provides useful supporting, corroborating, explanatory, or contextual evidence.

1 = MARGINALLY RELEVANT
    The page is related to the topic but adds little usable evidence for answering the question.

0 = IRRELEVANT
    The page provides no meaningful evidence for answering the question.

Core-evidence instructions:
1. A page can receive grade 4 even if another page contains the same or similar critical fact. Do not lower a page merely because equivalent evidence exists elsewhere.
2. If the page is grade 4, identify the major answer component or evidence facet that makes it indispensable (for example: historical status, quantitative outcome, 2015 framework, 2022 framework, partner organization, regulatory mechanism, funding structure).
3. If the page appears to duplicate substantially the same critical evidence as another candidate page, assign a redundancy-group description so equivalent pages can later be clustered.
4. A page that is merely very informative or topically rich is not automatically grade 4. Grade 4 should be reserved for evidence needed to satisfy an important requirement of the question.
5. For comparison questions, consider whether the page provides indispensable evidence for one side of the comparison. For multi-document or multi-facet questions, consider which required facet the page supports.

Return a structured record with:
- question_id
- page_id
- grade (0-4)
- core_facet (short label; null if grade < 4)
- redundancy_group_hint (short description; null if not applicable)
- rationale (1-3 sentences)"""


class JudgeError(RuntimeError):
    """Raised when judging inputs or model outputs are invalid."""


@dataclass(frozen=True)
class Candidate:
    query_id: str
    page_id: str


@dataclass(frozen=True)
class PageEvidence:
    page_id: str
    doc_id: str
    pdf_page: int
    document_title: str
    page_text: str


@dataclass(frozen=True)
class Judgment:
    query_id: str
    page_id: str
    grade: int
    core_facet: str | None
    redundancy_group_hint: str | None
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.query_id,
            "page_id": self.page_id,
            "grade": self.grade,
            "core_facet": self.core_facet,
            "redundancy_group_hint": self.redundancy_group_hint,
            "rationale": self.rationale,
        }


class AsyncCompletionClient(Protocol):
    async def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: float,
        max_tokens: int,
    ) -> str:
        """Return one model response as text."""


class AsyncRequestRateLimiter:
    """Space request starts evenly to respect a provider RPM quota."""

    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._interval = 60.0 / requests_per_minute
        self._next_start = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            delay = self._next_start - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_start = loop.time() + self._interval


def provider_retry_delay(error: Exception) -> float | None:
    """Extract short RetryInfo delays; long daily waits remain fail-fast."""

    message = str(error)
    seconds_match = re.search(
        r'retryDelay["\s:]+"?(\d+(?:\.\d+)?)s',
        message,
        re.IGNORECASE,
    )
    if seconds_match:
        seconds = float(seconds_match.group(1))
        return seconds + 1.0 if seconds <= 120 else None
    duration_match = re.search(
        r'retry in (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s',
        message,
        re.IGNORECASE,
    )
    if duration_match:
        seconds = (
            float(duration_match.group(1) or 0) * 3600
            + float(duration_match.group(2) or 0) * 60
            + float(duration_match.group(3))
        )
        return seconds + 1.0 if seconds <= 120 else None
    return None

class LiteLLMAsyncClient:
    """Provider-neutral asynchronous LiteLLM adapter."""

    async def complete(
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
            raise JudgeError(
                "LiteLLM is required for live judging. Install requirements.txt "
                "or run with --mock."
            ) from error

        options: dict[str, Any] = {}
        if model.endswith("gpt-5.6-sol"):
            options["reasoning_effort"] = "none"
        elif model.startswith("gemini/"):
            # Gemini 3 uses reasoning tokens from the output budget. Keep its
            # required thinking at the lowest level and require a JSON object.
            options["reasoning_effort"] = "none"
            options["response_format"] = {"type": "json_object"}
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            timeout=timeout,
            num_retries=0,
            **options,
        )
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise JudgeError("Judge response omitted message content") from error
        return coerce_text(content)


class MockJudgeClient:
    """Deterministic protocol mock; its grades are not benchmark judgments."""

    async def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: float,
        max_tokens: int,
    ) -> str:
        del model, system_prompt, timeout, max_tokens
        query_id = re.search(r"^Question ID: (.+)$", user_prompt, re.MULTILINE)
        page_id = re.search(r"^Page ID: (.+)$", user_prompt, re.MULTILINE)
        if not query_id or not page_id:
            raise JudgeError("Mock could not parse blind judging prompt")
        return json.dumps(
            {
                "question_id": query_id.group(1).strip(),
                "page_id": page_id.group(1).strip(),
                "grade": 2,
                "core_facet": None,
                "redundancy_group_hint": None,
                "rationale": "Mock judgment for protocol validation.",
            }
        )


def coerce_text(content: Any) -> str:
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
    raise JudgeError("Judge response content was not text")


def load_candidates(path: Path) -> list[Candidate]:
    """Load only blind candidate identity; intentionally discard all provenance."""

    if not path.is_file():
        raise JudgeError(f"Candidate pool does not exist: {path}")
    candidates: list[Candidate] = []
    seen: set[CandidateKey] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise JudgeError(f"{path}:{line_number}: invalid JSON: {error}") from error
            query_id = item.get("query_id") if isinstance(item, dict) else None
            page_id = item.get("page_id") if isinstance(item, dict) else None
            if not isinstance(query_id, str) or not query_id.strip():
                raise JudgeError(f"{path}:{line_number}: invalid query_id")
            if not isinstance(page_id, str) or not PAGE_ID_PATTERN.fullmatch(page_id):
                raise JudgeError(f"{path}:{line_number}: invalid page_id")
            key = (query_id.strip(), page_id)
            if key in seen:
                raise JudgeError(f"{path}:{line_number}: duplicate candidate {key}")
            seen.add(key)
            candidates.append(Candidate(*key))
    if not candidates:
        raise JudgeError(f"Candidate pool contains no records: {path}")
    return candidates


def load_document_titles(path: Path) -> dict[str, str]:
    """Load exact document titles from the frozen source catalog."""

    if not path.is_file():
        raise JudgeError(f"Source catalog does not exist: {path}")
    titles: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        id_field = "document_id" if "document_id" in fields else "doc_id"
        if id_field not in fields or "title" not in fields:
            raise JudgeError(
                f"{path} must contain document_id (or doc_id) and title columns"
            )
        for line_number, row in enumerate(reader, start=2):
            doc_id = (row.get(id_field) or "").strip()
            title = (row.get("title") or "").strip()
            if not doc_id or not title:
                raise JudgeError(f"{path}:{line_number}: blank doc_id or title")
            if doc_id in titles:
                raise JudgeError(f"{path}:{line_number}: duplicate doc_id {doc_id}")
            titles[doc_id] = title
    return titles


def load_candidate_pages(
    path: Path,
    candidates: list[Candidate],
    titles: dict[str, str],
) -> dict[str, PageEvidence]:
    """Load exact text only for pages present in the candidate pool."""

    required = {candidate.page_id for candidate in candidates}
    found: dict[str, PageEvidence] = {}
    if not path.is_file():
        raise JudgeError(f"Corpus JSONL does not exist: {path}")
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise JudgeError(f"{path}:{line_number}: invalid JSON: {error}") from error
            page_id = item.get("page_id") if isinstance(item, dict) else None
            if page_id not in required:
                continue
            doc_id = item.get("doc_id")
            pdf_page = item.get("pdf_page")
            page_text = item.get("page_text")
            if (
                not isinstance(doc_id, str)
                or type(pdf_page) is not int
                or pdf_page < 1
                or not isinstance(page_text, str)
            ):
                raise JudgeError(f"{path}:{line_number}: malformed candidate page")
            if page_id in found:
                raise JudgeError(f"{path}:{line_number}: duplicate page_id {page_id}")
            title = titles.get(doc_id)
            if title is None:
                raise JudgeError(f"No catalog title for {doc_id} ({page_id})")
            found[page_id] = PageEvidence(
                page_id=page_id,
                doc_id=doc_id,
                pdf_page=pdf_page,
                document_title=title,
                page_text=page_text,
            )
    missing = sorted(required - found.keys())
    if missing:
        preview = ", ".join(missing[:10])
        raise JudgeError(f"Corpus is missing {len(missing)} candidate pages: {preview}")
    return found


def build_blind_prompt(query: Query, page: PageEvidence) -> str:
    """Build a prompt containing no retrieval-system or selection metadata."""

    return (
        f"Question ID: {query.query_id}\n"
        f"Question: {query.text}\n\n"
        f"Page ID: {page.page_id}\n"
        f"Document Title: {page.document_title}\n"
        f"Page Text:\n{page.page_text}"
    )


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            return value[0]
        if isinstance(value, str) and value.strip() != cleaned:
            return _extract_json_object(value)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_judgment(raw: str, candidate: Candidate) -> Judgment:
    """Parse and strictly validate one V2 facet-aware model judgment."""

    value = _extract_json_object(raw)
    if value is None:
        raise JudgeError("Response was not a JSON object")
    if "question_id" not in value and "query_id" in value:
        value["question_id"] = value.pop("query_id")
    required = {
        "question_id",
        "page_id",
        "grade",
        "core_facet",
        "redundancy_group_hint",
        "rationale",
    }
    missing = required - set(value)
    if missing:
        raise JudgeError(f"Response omitted required fields: {sorted(missing)}")
    unexpected = set(value) - required
    if unexpected:
        raise JudgeError(f"Response included unexpected fields: {sorted(unexpected)}")
    if value["question_id"] != candidate.query_id:
        raise JudgeError("Response question_id did not match the request")
    if value["page_id"] != candidate.page_id:
        raise JudgeError("Response page_id did not match the request")
    grade = value["grade"]
    if type(grade) is not int or not 0 <= grade <= 4:
        raise JudgeError("Response grade must be an integer from 0 through 4")

    core_facet = value["core_facet"]
    redundancy_hint = value["redundancy_group_hint"]
    if grade == 4:
        if not isinstance(core_facet, str) or not core_facet.strip():
            raise JudgeError("Grade-4 response requires a non-empty core_facet")
        core_facet = core_facet.strip()
    elif core_facet is not None:
        raise JudgeError("core_facet must be null when grade is below 4")

    if redundancy_hint is not None:
        if not isinstance(redundancy_hint, str) or not redundancy_hint.strip():
            raise JudgeError("redundancy_group_hint must be a non-empty string or null")
        redundancy_hint = redundancy_hint.strip()
    rationale = value["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise JudgeError("Response rationale must be a non-empty string")
    return Judgment(
        query_id=candidate.query_id,
        page_id=candidate.page_id,
        grade=grade,
        core_facet=core_facet,
        redundancy_group_hint=redundancy_hint,
        rationale=rationale.strip(),
    )

async def judge_candidate(
    *,
    client: AsyncCompletionClient,
    semaphore: asyncio.Semaphore,
    model: str,
    judge_name: str,
    candidate: Candidate,
    query: Query,
    page: PageEvidence,
    timeout: float,
    retries: int,
    rate_limiter: AsyncRequestRateLimiter | None = None,
) -> Judgment:
    """Judge one candidate with bounded, format-aware retries."""

    prompt = build_blind_prompt(query, page)
    correction = (
        "\n\nFORMAT CORRECTION: Return exactly one JSON object, not an array or "
        "prose. Include only question_id, page_id, grade, core_facet, "
        "redundancy_group_hint, and rationale. Use null for core_facet when "
        "grade < 4 and null for redundancy_group_hint when not applicable."
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        raw = ""
        try:
            if rate_limiter is not None:
                await rate_limiter.acquire()
            async with semaphore:
                raw = await client.complete(
                    model=model,
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                    user_prompt=prompt if attempt == 1 else prompt + correction,
                    timeout=timeout,
                    max_tokens=2048,
                )
            return parse_judgment(raw, candidate)
        except Exception as error:
            last_error = error
            preview = re.sub(r"\s+", " ", raw).strip()[:400]
            LOGGER.warning(
                "%s %s/%s attempt %d failed: %s%s",
                judge_name,
                candidate.query_id,
                candidate.page_id,
                attempt,
                error,
                f"; response_preview={preview!r}" if preview else "",
            )
            if attempt < retries:
                retry_delay = provider_retry_delay(error)
                await asyncio.sleep(
                    retry_delay if retry_delay is not None else 2 ** (attempt - 1)
                )
    raise JudgeError(
        f"{judge_name} failed for {candidate.query_id}/{candidate.page_id} "
        f"after {retries} attempts: {last_error}"
    ) from last_error

def atomic_write_judgments(path: Path, judgments: list[Judgment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            for judgment in judgments:
                output.write(
                    json.dumps(
                        judgment.as_dict(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_judge_signature(
    *,
    judge_name: str,
    model: str,
    candidates: list[Candidate],
    queries: dict[str, Query],
    pages: dict[str, PageEvidence],
) -> str:
    """Fingerprint every input that affects a judge run."""

    digest = hashlib.sha256()
    digest.update(judge_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(JUDGE_SYSTEM_PROMPT.encode("utf-8"))
    for candidate in candidates:
        query = queries[candidate.query_id]
        page = pages[candidate.page_id]
        payload = [
            candidate.query_id,
            candidate.page_id,
            query.text,
            page.document_title,
            page.page_text,
        ]
        digest.update(b"\0")
        digest.update(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def _load_judgment_rows(
    path: Path,
    *,
    candidates: dict[CandidateKey, Candidate],
) -> dict[CandidateKey, Judgment]:
    if not path.is_file():
        return {}
    judgments: dict[CandidateKey, Judgment] = {}
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise JudgeError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise JudgeError(f"{path}:{line_number}: judgment must be an object")
            key = (value.get("question_id", value.get("query_id")), value.get("page_id"))
            candidate = candidates.get(key)
            if candidate is None:
                raise JudgeError(f"{path}:{line_number}: unknown judgment key {key}")
            judgment = parse_judgment(json.dumps(value), candidate)
            if key in judgments:
                raise JudgeError(f"{path}:{line_number}: duplicate judgment {key}")
            judgments[key] = judgment
    return judgments


def write_judge_checkpoint(
    path: Path,
    *,
    signature: str,
    judge_name: str,
    model: str,
    candidates: list[Candidate],
    judgments: dict[CandidateKey, Judgment],
) -> None:
    """Atomically save completed items in canonical candidate order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "signature": signature,
        "judge_name": judge_name,
        "model": model,
        "judgments": [
            judgments[(candidate.query_id, candidate.page_id)].as_dict()
            for candidate in candidates
            if (candidate.query_id, candidate.page_id) in judgments
        ],
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


def load_judge_checkpoint(
    path: Path,
    *,
    signature: str,
    judge_name: str,
    model: str,
    candidates: dict[CandidateKey, Candidate],
) -> dict[CandidateKey, Judgment]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise JudgeError(f"Unreadable judge checkpoint {path}: {error}") from error
    if (
        payload.get("schema_version") != 2
        or payload.get("signature") != signature
        or payload.get("judge_name") != judge_name
        or payload.get("model") != model
    ):
        LOGGER.warning("Ignoring stale judge checkpoint: %s", path)
        return {}
    rows = payload.get("judgments")
    if not isinstance(rows, list):
        raise JudgeError(f"Malformed judge checkpoint: {path}")
    judgments: dict[CandidateKey, Judgment] = {}
    for index, value in enumerate(rows, start=1):
        if not isinstance(value, dict):
            raise JudgeError(f"Malformed judgment {index} in checkpoint {path}")
        key = (value.get("question_id", value.get("query_id")), value.get("page_id"))
        candidate = candidates.get(key)
        if candidate is None or key in judgments:
            raise JudgeError(f"Invalid judgment key {key} in checkpoint {path}")
        judgments[key] = parse_judgment(json.dumps(value), candidate)
    return judgments


async def run_judge(
    *,
    client: AsyncCompletionClient,
    judge_name: str,
    model: str,
    candidates: list[Candidate],
    queries: dict[str, Query],
    pages: dict[str, PageEvidence],
    output_path: Path,
    concurrency: int,
    timeout: float,
    retries: int,
    resume: bool = False,
    requests_per_minute: float | None = None,
) -> int:
    """Run a judge with bounded concurrency and atomic batch checkpoints."""

    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    candidate_map = {
        (candidate.query_id, candidate.page_id): candidate for candidate in candidates
    }
    if len(candidate_map) != len(candidates):
        raise JudgeError("Candidate list contains duplicate query-page keys")
    for candidate in candidates:
        if candidate.query_id not in queries:
            raise JudgeError(f"No question text for {candidate.query_id}")

    signature = build_judge_signature(
        judge_name=judge_name,
        model=model,
        candidates=candidates,
        queries=queries,
        pages=pages,
    )
    checkpoint_path = output_path.parent / ".checkpoints" / f"{judge_name}.json"
    completed: dict[CandidateKey, Judgment] = {}
    if resume:
        completed.update(
            load_judge_checkpoint(
                checkpoint_path,
                signature=signature,
                judge_name=judge_name,
                model=model,
                candidates=candidate_map,
            )
        )
        for key, judgment in _load_judgment_rows(
            output_path,
            candidates=candidate_map,
        ).items():
            completed.setdefault(key, judgment)
        if completed:
            LOGGER.info(
                "%s: resumed %d/%d completed judgments",
                judge_name,
                len(completed),
                len(candidates),
            )
            write_judge_checkpoint(
                checkpoint_path,
                signature=signature,
                judge_name=judge_name,
                model=model,
                candidates=candidates,
                judgments=completed,
            )
    else:
        checkpoint_path.unlink(missing_ok=True)

    pending = [
        candidate
        for candidate in candidates
        if (candidate.query_id, candidate.page_id) not in completed
    ]
    semaphore = asyncio.Semaphore(concurrency)
    rate_limiter = (
        AsyncRequestRateLimiter(requests_per_minute)
        if requests_per_minute is not None
        else None
    )
    if requests_per_minute is not None:
        LOGGER.info(
            "%s throttled to %.1f requests/minute",
            judge_name,
            requests_per_minute,
        )
    batch_size = concurrency * 4
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        results = await asyncio.gather(
            *(
                judge_candidate(
                    client=client,
                    semaphore=semaphore,
                    model=model,
                    judge_name=judge_name,
                    candidate=candidate,
                    query=queries[candidate.query_id],
                    page=pages[candidate.page_id],
                    timeout=timeout,
                    retries=retries,
                    rate_limiter=rate_limiter,
                )
                for candidate in batch
            ),
            return_exceptions=True,
        )
        failures: list[BaseException] = []
        for candidate, result in zip(batch, results):
            if isinstance(result, BaseException):
                failures.append(result)
            else:
                completed[(candidate.query_id, candidate.page_id)] = result
        if completed:
            write_judge_checkpoint(
                checkpoint_path,
                signature=signature,
                judge_name=judge_name,
                model=model,
                candidates=candidates,
                judgments=completed,
            )
        if failures:
            first = failures[0]
            raise JudgeError(
                f"{judge_name} stopped with {len(failures)} failed judgment(s); "
                f"{len(completed)}/{len(candidates)} are checkpointed. First: {first}"
            ) from first
        LOGGER.info(
            "%s progress: %d/%d judgments checkpointed",
            judge_name,
            len(completed),
            len(candidates),
        )

    ordered = [
        completed[(candidate.query_id, candidate.page_id)] for candidate in candidates
    ]
    atomic_write_judgments(output_path, ordered)
    return len(ordered)

def parse_model_overrides(values: list[str]) -> dict[str, str]:
    models = dict(DEFAULT_MODELS)
    for value in values:
        judge, separator, model = value.partition("=")
        if not separator or judge not in models or not model.strip():
            raise ValueError(
                "--model must be JUDGE=MODEL where JUDGE is judge_a, judge_b, or judge_c"
            )
        models[judge] = model.strip()
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run three blind, independent page-level relevance judges."
    )
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--judges",
        nargs="+",
        choices=tuple(DEFAULT_MODELS),
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="JUDGE=MODEL",
    )
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--gemini-rpm",
        type=float,
        default=20.0,
        help="Throttle Gemini judges below the project RPM quota.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Validate orchestration without API calls; grades are not benchmark data.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed judgments from validated outputs or checkpoints.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    if args.retries < 1:
        raise ValueError("retries must be positive")
    if args.gemini_rpm <= 0:
        raise ValueError("gemini-rpm must be positive")
    candidates = load_candidates(args.pool)
    query_list = load_queries(args.questions)
    queries = {query.query_id: query for query in query_list}
    titles = load_document_titles(args.catalog)
    pages = load_candidate_pages(args.corpus, candidates, titles)
    models = parse_model_overrides(args.model)
    client: AsyncCompletionClient = (
        MockJudgeClient() if args.mock else LiteLLMAsyncClient()
    )
    if args.mock:
        LOGGER.warning(
            "Mock mode validates orchestration only; grades are not judgments."
        )

    for judge_name in args.judges:
        LOGGER.info(
            "Running %s (%s) over %d candidates",
            judge_name,
            models[judge_name],
            len(candidates),
        )
        count = await run_judge(
            client=client,
            judge_name=judge_name,
            model=models[judge_name],
            candidates=candidates,
            queries=queries,
            pages=pages,
            output_path=args.output_dir.resolve() / f"{judge_name}.jsonl",
            concurrency=args.concurrency,
            timeout=args.timeout,
            retries=args.retries,
            resume=args.resume,
            requests_per_minute=(
                args.gemini_rpm
                if models[judge_name].startswith("gemini/")
                else None
            ),
        )
        LOGGER.info("Wrote %d judgments to %s.jsonl", count, judge_name)
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )
    try:
        return asyncio.run(async_main(args))
    except (JudgeError, OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
