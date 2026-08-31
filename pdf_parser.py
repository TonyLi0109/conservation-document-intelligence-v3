"""Production PDF extraction and page-aware chunking for V3.

The parser depends only on ``pypdf`` and V3's canonical data contract. It has no
catalog, database, retrieval, or legacy-repository dependencies.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from config import SETTINGS
from data_models import KnowledgeArtifact


LOGGER = logging.getLogger(__name__)
TARGET_WORDS = SETTINGS.chunking.target_words
OVERLAP_WORDS = SETTINGS.chunking.overlap_words
WORD_PATTERN = re.compile(r"\S+")
PAGE_EXTRACTION_TIMEOUT_SECONDS = SETTINGS.chunking.page_timeout_seconds
PAGE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9 .:_-]{1,32}$", re.IGNORECASE)


@dataclass(slots=True)
class DocumentParseReport:
    """Mutable ingestion telemetry, separate from canonical evidence objects."""

    total_pages: int = 0
    extracted_pages: int = 0
    empty_pages: list[int] = field(default_factory=list)
    failed_pages: list[int] = field(default_factory=list)
    fallback_pages: list[int] = field(default_factory=list)
    chunk_count: int = 0


def normalize_text(text: str) -> str:
    """Normalize extracted text while preserving meaningful line boundaries.

    NFKC normalization standardizes compatibility characters. NUL bytes,
    platform-specific line endings, repeated horizontal whitespace, and repeated
    blank lines are removed because they are common PDF extraction artifacts.
    """

    normalized_text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    normalized_text = normalized_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        re.sub(r"[\t \u00a0]+", " ", line).strip()
        for line in normalized_text.split("\n")
    ]

    normalized_lines: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        normalized_lines.append(line)
        previous_blank = is_blank
    return "\n".join(normalized_lines).strip()


def _pdf_page_labels(reader: PdfReader, page_count: int) -> list[str | None]:
    """Read standard PDF logical page labels without guessing from body text."""

    try:
        raw_labels = reader.page_labels
    except Exception as error:
        LOGGER.warning("Could not read PDF PageLabels: %s", error)
        return [None] * page_count
    if not isinstance(raw_labels, list) or len(raw_labels) != page_count:
        return [None] * page_count
    labels: list[str | None] = []
    for physical_page, raw_label in enumerate(raw_labels, start=1):
        label = str(raw_label).strip() if raw_label is not None else ""
        if not label or not PAGE_LABEL_PATTERN.fullmatch(label):
            labels.append(None)
            continue
        # A label identical to the physical index carries no additional logical
        # information, so omit it and keep the citation concise.
        labels.append(None if label == str(physical_page) else label)
    return labels


def read_pdf_page_labels(pdf_path: str | Path) -> dict[str, str]:
    """Return trusted physical-page to logical-label mappings from PDF metadata."""

    path = Path(pdf_path)
    reader = PdfReader(path)
    page_count = len(reader.pages)
    return {
        str(index): label
        for index, label in enumerate(_pdf_page_labels(reader, page_count), start=1)
        if label is not None
    }


def _run_page_worker(source: Path, page_index: int, mode: str) -> subprocess.CompletedProcess[str] | None:
    """Run one isolated extraction backend and return its bounded result."""

    command = [sys.executable, str(Path(__file__).resolve()), mode,
               str(source.resolve()), str(page_index)]
    try:
        return subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=PAGE_EXTRACTION_TIMEOUT_SECONDS, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        LOGGER.warning("Page worker %s failed for page %s of %s: %s", mode,
                       page_index + 1, source, error)
        return None


def _extract_page_in_subprocess(source: Path, page_index: int) -> tuple[str | None, bool]:
    """Extract one page outside the ingestion process.

    Some malformed PDF text matrices can trigger a native access violation in
    ``pypdf`` rather than a catchable Python exception. A dedicated process is a
    hard fault boundary: its non-zero exit code causes only that page to be
    skipped, while the canonical ingestion process remains alive.
    """

    primary = _run_page_worker(source, page_index, "--extract-page")
    if primary is not None and primary.returncode == 0:
        return primary.stdout, False
    fallback = _run_page_worker(source, page_index, "--extract-page-fitz")
    if fallback is not None and fallback.returncode == 0:
        LOGGER.info("Recovered page %s of %s with PyMuPDF", page_index + 1, source.name)
        return fallback.stdout, True
    detail = fallback.stderr.strip().splitlines() if fallback is not None else []
    if not detail and primary is not None:
        detail = primary.stderr.strip().splitlines()
    summary = detail[-1] if detail else "all extraction backends failed"
    LOGGER.warning("Skipped unreadable page %s from %s: %s", page_index + 1, source, summary)
    return None, False


def _page_tokens(page_count: int, source: Path, report: DocumentParseReport) -> list[tuple[str, int]]:
    """Safely extract words paired with physical one-based page numbers."""

    tokens: list[tuple[str, int]] = []
    empty_pages: list[int] = []
    failed_pages: list[int] = []
    try:
        for page_index in range(page_count):
            page_number = page_index + 1
            if page_number == 1 or page_number % 10 == 0 or page_number == page_count:
                LOGGER.info(
                    "Extracting page %s/%s from %s",
                    page_number,
                    page_count,
                    source.name,
                )
            raw_text, used_fallback = _extract_page_in_subprocess(source, page_index)
            if raw_text is None:
                failed_pages.append(page_number)
                continue
            if used_fallback:
                report.fallback_pages.append(page_number)
            page_text = normalize_text(raw_text)
            if not page_text:
                empty_pages.append(page_number)
                continue
            report.extracted_pages += 1
            tokens.extend(
                (match.group(0), page_number)
                for match in WORD_PATTERN.finditer(page_text)
            )
    except Exception as error:
        LOGGER.warning("Could not enumerate pages in %s: %s", source, error)
        return []

    if empty_pages:
        LOGGER.info(
            "Skipped %s empty page(s) in %s: %s",
            len(empty_pages),
            source,
            ", ".join(str(page) for page in empty_pages[:10]),
        )
    if failed_pages:
        LOGGER.info(
            "Skipped %s unreadable page(s) in %s",
            len(failed_pages),
            source,
        )
    report.empty_pages.extend(empty_pages)
    report.failed_pages.extend(failed_pages)
    return tokens


def parse_pdf_to_artifacts(
    pdf_path: str,
    document_id: str,
    title: str,
    report: DocumentParseReport | None = None,
) -> Iterator[KnowledgeArtifact]:
    """Yield validated, overlapping chunks from a PDF.

    Chunks contain at most 750 whitespace-delimited words and overlap the prior
    chunk by 100 words. Words retain their source-page association while windows
    are formed across page boundaries. A cross-page chunk is assigned the page
    number of its first word, satisfying the V3 provenance rule.

    Missing files, malformed PDFs, encrypted/unreadable documents, empty pages,
    and individual page extraction errors are logged and produce no artifact for
    the affected content rather than terminating a corpus ingestion run.
    """

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")

    path = Path(pdf_path)
    if not path.is_file():
        LOGGER.warning("PDF does not exist or is not a file: %s", path)
        return
    if path.suffix.casefold() != ".pdf":
        LOGGER.warning("Skipping non-PDF input: %s", path)
        return

    try:
        reader = PdfReader(path)
    except Exception as error:
        LOGGER.warning("Could not open PDF %s: %s", path, error)
        return

    try:
        page_count = len(reader.pages)
    except Exception as error:
        LOGGER.warning("Could not determine page count for %s: %s", path, error)
        return

    printed_labels = _pdf_page_labels(reader, page_count)

    parse_report = report if report is not None else DocumentParseReport()
    parse_report.total_pages = page_count
    tokens = _page_tokens(page_count, path, parse_report)
    if not tokens:
        LOGGER.warning("PDF produced no readable text: %s", path)
        return

    start = 0
    while start < len(tokens):
        end = min(start + TARGET_WORDS, len(tokens))
        window = tokens[start:end]
        chunk_text = " ".join(word for word, _ in window)
        starting_page = window[0][1]
        try:
            yield KnowledgeArtifact(
                document_id=document_id,
                title=title,
                page_number=str(starting_page),
                original_text_chunk=chunk_text,
                printed_page_label=printed_labels[starting_page - 1],
            )
            parse_report.chunk_count += 1
        except (TypeError, ValueError) as error:
            LOGGER.warning(
                "Skipping invalid chunk beginning on page %s of %s: %s",
                starting_page,
                path,
                error,
            )

        if end == len(tokens):
            break
        start = end - OVERLAP_WORDS


def parse_text_to_artifacts(
    text_path: str,
    document_id: str,
    title: str,
) -> Iterator[KnowledgeArtifact]:
    """Yield validated chunks from a non-paginated legacy web-text source.

    Text sources have no physical page boundary, so every artifact uses the
    canonical ``Web`` location. Cleanup and word-window parameters intentionally
    match PDF ingestion so retrieval behavior remains consistent across source
    types.
    """

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")

    path = Path(text_path)
    if not path.is_file():
        LOGGER.warning("Text source does not exist or is not a file: %s", path)
        return
    if path.suffix.casefold() != ".txt":
        LOGGER.warning("Skipping non-text input: %s", path)
        return

    try:
        raw_text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as error:
        LOGGER.warning("Could not read text source %s: %s", path, error)
        return

    cleaned_text = normalize_text(raw_text)
    words = [match.group(0) for match in WORD_PATTERN.finditer(cleaned_text)]
    if not words:
        LOGGER.warning("Text source produced no readable text: %s", path)
        return

    start = 0
    while start < len(words):
        end = min(start + TARGET_WORDS, len(words))
        chunk_text = " ".join(words[start:end])
        try:
            yield KnowledgeArtifact(
                document_id=document_id,
                title=title,
                page_number="Web",
                original_text_chunk=chunk_text,
            )
        except (TypeError, ValueError) as error:
            LOGGER.warning("Skipping invalid text chunk from %s: %s", path, error)

        if end == len(words):
            break
        start = end - OVERLAP_WORDS


def _extract_page_worker(pdf_path: str, page_index: int) -> int:
    """Internal subprocess entry point; stdout is UTF-8 extracted text bytes."""

    try:
        reader = PdfReader(pdf_path)
        text = reader.pages[page_index].extract_text() or ""
        # Write bytes explicitly: Windows Store Python may configure stdout as
        # GBK even when the parent requests UTF-8 decoding.
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.flush()
        return 0
    except Exception as error:
        # The parent consumes only a short sanitized diagnostic from stderr.
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        return 1


def _extract_page_fitz_worker(pdf_path: str, page_index: int) -> int:
    """Fallback worker using PyMuPDF for pages pypdf cannot safely decode."""

    try:
        import fitz

        with fitz.open(pdf_path) as document:
            text = document.load_page(page_index).get_text("text") or ""
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.flush()
        return 0
    except Exception as error:
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        return 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] in {"--extract-page", "--extract-page-fitz"}:
        try:
            requested_page = int(sys.argv[3])
        except ValueError:
            raise SystemExit(2)
        worker = _extract_page_worker if sys.argv[1] == "--extract-page" else _extract_page_fitz_worker
        raise SystemExit(worker(sys.argv[2], requested_page))
