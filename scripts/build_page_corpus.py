"""Build the frozen CDIRP page-level IR benchmark corpus.

Example:
    python scripts/build_page_corpus.py path/to/frozen_sources

The default contract requires exactly one DOC001 through DOC036 source. Physical
PDF pages are preserved; frozen TXT web snapshots become one synthetic page with
pdf_page=1. Output replacement is atomic: an unreadable source leaves any prior
dataset untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from collections.abc import Iterator

try:
    import pymupdf as fitz
except ImportError:  # Compatibility with older PyMuPDF releases.
    try:
        import fitz  # type: ignore[no-redef]
    except ImportError as error:  # pragma: no cover - exercised only without dependency
        raise SystemExit(
            "PyMuPDF is required. Install project dependencies with "
            "python -m pip install -r requirements.txt."
        ) from error


LOGGER = logging.getLogger("build_page_corpus")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "benchmark" / "corpus_pages.jsonl"
SOURCE_NAME = re.compile(
    r"^(DOC(?P<number>\d{3}))(?:_.+)?\.(?P<extension>pdf|txt)$",
    re.IGNORECASE,
)
HASH_BLOCK_BYTES = 1024 * 1024


class CorpusExtractionError(RuntimeError):
    """Raised when the frozen corpus cannot be reproduced completely."""


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of the exact file bytes."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(HASH_BLOCK_BYTES), b""):
                digest.update(block)
    except OSError as error:
        raise CorpusExtractionError(f"Could not hash {path}: {error}") from error
    return digest.hexdigest()


def normalize_page_text(text: str) -> str:
    """Apply deterministic text cleanup without changing page membership.

    This matches the project's canonical PDF normalization: Unicode NFKC, LF
    line endings, removal of NUL bytes, collapsed horizontal whitespace, and at
    most one consecutive blank line. An image-only or otherwise blank page is
    retained with an empty string.
    """

    normalized = unicodedata.normalize("NFKC", text).replace("\x00", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        re.sub(r"[\t \u00a0]+", " ", line).strip()
        for line in normalized.split("\n")
    ]
    output: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        output.append(line)
        previous_blank = is_blank
    return "\n".join(output).strip()


def discover_frozen_sources(
    input_dir: Path,
    *,
    expected_documents: int = 36,
) -> list[tuple[str, Path]]:
    """Validate and return frozen PDF/TXT sources in document-ID order."""

    if expected_documents < 1 or expected_documents > 999:
        raise ValueError("expected_documents must be between 1 and 999")
    if not input_dir.is_dir():
        raise CorpusExtractionError(f"Input directory does not exist: {input_dir}")

    source_paths = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in {".pdf", ".txt"}
        ),
        key=lambda path: path.name.casefold(),
    )
    malformed = [
        path.name for path in source_paths if SOURCE_NAME.fullmatch(path.name) is None
    ]
    if malformed:
        raise CorpusExtractionError(
            "Source filenames must start with DOC### and end in .pdf or .txt; "
            "invalid: " + ", ".join(malformed)
        )

    by_id: dict[str, Path] = {}
    for path in source_paths:
        match = SOURCE_NAME.fullmatch(path.name)
        assert match is not None
        doc_id = match.group(1).upper()
        if doc_id in by_id:
            raise CorpusExtractionError(
                f"Duplicate document ID {doc_id}: {by_id[doc_id].name}, {path.name}"
            )
        by_id[doc_id] = path

    expected_ids = {f"DOC{number:03d}" for number in range(1, expected_documents + 1)}
    actual_ids = set(by_id)
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise CorpusExtractionError(
            f"Frozen corpus must contain exactly {expected_documents} documents "
            f"(DOC001..DOC{expected_documents:03d}); " + "; ".join(details)
        )

    return [(doc_id, by_id[doc_id]) for doc_id in sorted(by_id)]


def extract_pdf_pages(
    doc_id: str,
    pdf_path: Path,
    file_checksum: str,
) -> Iterator[dict[str, object]]:
    """Yield one JSON-compatible record per physical PDF page."""

    try:
        document = fitz.open(pdf_path)
    except Exception as error:
        raise CorpusExtractionError(f"Could not open {pdf_path}: {error}") from error

    try:
        if document.needs_pass:
            raise CorpusExtractionError(f"Encrypted PDF requires a password: {pdf_path}")
        page_count = document.page_count
        if page_count < 1:
            raise CorpusExtractionError(f"PDF contains no pages: {pdf_path}")

        for page_index in range(page_count):
            pdf_page = page_index + 1
            try:
                page = document.load_page(page_index)
                raw_text = page.get_text("text", sort=False) or ""
            except Exception as error:
                raise CorpusExtractionError(
                    f"Could not extract {doc_id} PDF page {pdf_page}: {error}"
                ) from error
            yield {
                "page_id": f"{doc_id}_P{pdf_page:04d}",
                "doc_id": doc_id,
                "pdf_page": pdf_page,
                "page_text": normalize_page_text(raw_text),
                "file_checksum": file_checksum,
            }
    finally:
        document.close()


def extract_text_snapshot(
    doc_id: str,
    source_path: Path,
    file_checksum: str,
) -> Iterator[dict[str, object]]:
    """Yield a frozen text/web snapshot as one synthetic page.

    TXT sources do not contain physical PDF pagination. Assigning pdf_page=1
    keeps the required benchmark schema explicit and avoids fabricating page
    boundaries within a web snapshot.
    """

    try:
        raw_text = source_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise CorpusExtractionError(
            f"Could not read text snapshot {source_path}: {error}"
        ) from error
    yield {
        "page_id": f"{doc_id}_P0001",
        "doc_id": doc_id,
        "pdf_page": 1,
        "page_text": normalize_page_text(raw_text),
        "file_checksum": file_checksum,
    }


def extract_source_pages(
    doc_id: str,
    source_path: Path,
    file_checksum: str,
) -> Iterator[dict[str, object]]:
    """Dispatch extraction according to the frozen source type."""

    if source_path.suffix.casefold() == ".pdf":
        yield from extract_pdf_pages(doc_id, source_path, file_checksum)
    elif source_path.suffix.casefold() == ".txt":
        yield from extract_text_snapshot(doc_id, source_path, file_checksum)
    else:
        raise CorpusExtractionError(f"Unsupported source type: {source_path}")


def build_page_corpus(
    input_dir: Path,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    expected_documents: int = 36,
) -> tuple[int, int]:
    """Build the JSONL dataset atomically and return document/page counts."""

    documents = discover_frozen_sources(
        input_dir.resolve(), expected_documents=expected_documents
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    page_count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            for doc_id, pdf_path in documents:
                checksum = sha256_file(pdf_path)
                LOGGER.info("Extracting %s from %s", doc_id, pdf_path.name)
                for record in extract_source_pages(doc_id, pdf_path, checksum):
                    destination.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    )
                    destination.write("\n")
                    page_count += 1
            destination.flush()
            os.fsync(destination.fileno())

        if page_count < len(documents):
            raise CorpusExtractionError(
                f"Only {page_count} pages were extracted from {len(documents)} documents"
            )
        os.replace(temporary_path, output_path)
        temporary_path = None
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return len(documents), page_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen DOC### PDF/TXT sources into a deterministic page-level JSONL "
            "dataset with parent-file SHA-256 checksums."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing one frozen source for each DOC001 through DOC036.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--expected-documents",
        type=int,
        default=36,
        help="Expected contiguous DOC IDs beginning at DOC001 (default: 36).",
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
        documents, pages = build_page_corpus(
            args.input_dir,
            args.output,
            expected_documents=args.expected_documents,
        )
    except (CorpusExtractionError, OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1

    LOGGER.info(
        "Wrote %s pages from %s documents to %s",
        pages,
        documents,
        args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
