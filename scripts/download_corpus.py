"""Copy or download the legacy 35-document corpus into V3.

The legacy repository is treated as read-only. For each metadata row this script
first copies an existing ``local_file`` and otherwise downloads a catalog URL.
PDF responses are stored as PDF files; HTML sources are converted to plain text,
matching the legacy corpus' mix of ``pdf`` and ``html_text`` documents.

Example::

    python new-V3/scripts/download_corpus.py \
        --legacy-root conservation-document-intelligence-main/conservation-document-intelligence-main
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import shutil
import sys
import unicodedata
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


V3_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = V3_ROOT.parent
DEFAULT_LEGACY_ROOT = (
    WORKSPACE_ROOT
    / "conservation-document-intelligence-main"
    / "conservation-document-intelligence-main"
)
DEFAULT_SECONDARY_ROOT = (
    WORKSPACE_ROOT / "conservation-intelligence-master" / "conservation-intelligence-master"
)
DEFAULT_OUTPUT_DIR = V3_ROOT / "data" / "raw"
HF_SPACE_RAW_ROOT = (
    "https://huggingface.co/spaces/shanged/conservation-intelligence/resolve/main/data/raw"
)
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}
FINAL_URL_PATTERN = re.compile(r"final URL\s+(https?://\S+)", re.IGNORECASE)


@dataclass(frozen=True)
class FetchResult:
    """Outcome for one catalog record."""

    status: str
    document_id: str
    destination: Path | None = None
    detail: str = ""


class _ReadableHTMLParser(HTMLParser):
    """Small dependency-free HTML-to-text extractor for legacy web sources."""

    _ignored_tags = {"script", "style", "noscript", "svg", "nav", "footer"}
    _block_tags = {
        "article", "blockquote", "br", "div", "h1", "h2", "h3", "h4",
        "h5", "h6", "header", "li", "main", "p", "section", "table", "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in self._ignored_tags:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in self._block_tags:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in self._block_tags:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._parts.append(data)

    def text(self) -> str:
        lines = (re.sub(r"\s+", " ", line).strip() for line in "".join(self._parts).splitlines())
        return "\n\n".join(line for line in lines if line)


def _slug(value: str, *, max_length: int = 90) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_")
    return (slug or "Untitled")[:max_length].rstrip("_")


def _build_session(retries: int) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _safe_legacy_path(legacy_root: Path, catalog_value: str) -> Path | None:
    if not catalog_value.strip():
        return None
    root = legacy_root.resolve()
    candidate = (root / Path(catalog_value)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _candidate_urls(row: dict[str, str]) -> list[str]:
    urls: list[str] = []
    note_match = FINAL_URL_PATTERN.search(row.get("notes", ""))
    values = [row.get(field, "") for field in ("resolved_url", "url", "original_url")]
    if note_match:
        # The shanged catalog keeps representative DocumentCloud URLs in notes.
        values.insert(0, note_match.group(1).rstrip(".;,)"))
    for raw_value in values:
        value = raw_value.strip()
        if value and value not in urls and urlparse(value).scheme in {"http", "https"}:
            urls.append(value)
    return urls


def _existing_destination(output_dir: Path, document_id: str) -> Path | None:
    """Find either V3's logical name or a legacy ``DOCxxx.ext`` name."""
    prefix = _slug(document_id, max_length=20)
    candidates = [output_dir / f"{prefix}.pdf", output_dir / f"{prefix}.txt"]
    candidates.extend(sorted(output_dir.glob(f"{prefix}_*.pdf")))
    candidates.extend(sorted(output_dir.glob(f"{prefix}_*.txt")))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _html_to_text(content: bytes, encoding: str | None) -> str:
    decoded = content.decode(encoding or "utf-8", errors="replace")
    parser = _ReadableHTMLParser()
    parser.feed(decoded)
    parser.close()
    return parser.text()


def _write_atomically(destination: Path, content: bytes) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(destination)


def _copy_local(source: Path, output_dir: Path, stem: str) -> Path:
    suffix = source.suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        with source.open("rb") as handle:
            suffix = ".pdf" if handle.read(5) == b"%PDF-" else ".txt"
    destination = output_dir / f"{stem}{suffix}"
    temporary = destination.with_suffix(destination.suffix + ".part")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return destination


def _download(
    session: requests.Session,
    urls: list[str],
    output_dir: Path,
    stem: str,
    *,
    timeout: float,
    max_bytes: int,
    expected_pdf: bool,
) -> tuple[Path, str]:
    errors: list[str] = []
    for url in urls:
        try:
            request_headers = dict(BROWSER_HEADERS)
            if urlparse(url).hostname == "huggingface.co" and os.environ.get("HF_TOKEN"):
                request_headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
            response = session.get(
                url,
                headers=request_headers,
                timeout=timeout,
                allow_redirects=True,
            )
            response.raise_for_status()
            content = response.content
            if not content:
                raise ValueError("empty response")
            if len(content) > max_bytes:
                raise ValueError(f"response exceeds {max_bytes:,} bytes")

            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            is_pdf = content.startswith(b"%PDF-")
            if expected_pdf and not is_pdf:
                raise ValueError(
                    f"expected PDF but received {content_type or 'unknown content type'}"
                )
            if is_pdf:
                suffix = ".pdf"
                output = content
            else:
                text = _html_to_text(content, response.encoding)
                if len(text) < 200:
                    raise ValueError("web page produced less than 200 characters of useful text")
                suffix = ".txt"
                output = text.encode("utf-8")

            destination = output_dir / f"{stem}{suffix}"
            _write_atomically(destination, output)
            return destination, response.url
        except (requests.RequestException, OSError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errors) if errors else "catalog contains no usable URL")


def fetch_record(
    row: dict[str, str],
    *,
    legacy_root: Path,
    secondary_root: Path | None,
    secondary_row: dict[str, str] | None,
    output_dir: Path,
    session: requests.Session,
    timeout: float,
    max_bytes: int,
    overwrite: bool,
) -> FetchResult:
    document_id = row.get("doc_id", "").strip()
    title = row.get("title", "").strip()
    if not document_id or not title:
        return FetchResult("failed", document_id or "UNKNOWN", detail="missing doc_id or title")

    stem = f"{_slug(document_id, max_length=20)}_{_slug(title)}"
    if not overwrite:
        existing = _existing_destination(output_dir, document_id)
        if existing:
            return FetchResult("skipped", document_id, existing, "already exists")

    source = _safe_legacy_path(legacy_root, row.get("local_file", ""))
    try:
        if source and source.is_file() and source.stat().st_size > 0:
            destination = _copy_local(source, output_dir, stem)
            return FetchResult("copied", document_id, destination, str(source))

        if secondary_root and secondary_row:
            secondary_source = _safe_legacy_path(
                secondary_root, secondary_row.get("local_file", "")
            )
            if secondary_source and secondary_source.is_file() and secondary_source.stat().st_size > 0:
                destination = _copy_local(secondary_source, output_dir, stem)
                return FetchResult("copied", document_id, destination, str(secondary_source))

        urls = _candidate_urls(row)
        if secondary_row:
            urls.extend(url for url in _candidate_urls(secondary_row) if url not in urls)
            secondary_suffix = (
                Path(secondary_row.get("local_file", "")).suffix.lower() or ".pdf"
            )
            hf_url = f"{HF_SPACE_RAW_ROOT}/{document_id}{secondary_suffix}?download=true"
            if hf_url not in urls:
                urls.append(hf_url)

        destination, resolved_url = _download(
            session,
            urls,
            output_dir,
            stem,
            timeout=timeout,
            max_bytes=max_bytes,
            expected_pdf=(
                row.get("file_type", "").strip().lower() == "pdf"
                or bool(
                    secondary_row
                    and secondary_row.get("file_type", "").strip().lower() == "pdf"
                )
            ),
        )
        return FetchResult("downloaded", document_id, destination, resolved_url)
    except (OSError, RuntimeError, ValueError) as exc:
        return FetchResult("failed", document_id, detail=str(exc))


def fetch_corpus(args: argparse.Namespace) -> int:
    legacy_root = args.legacy_root.resolve()
    secondary_root = args.secondary_root.resolve() if args.secondary_root else None
    metadata_path = (args.metadata or legacy_root / "data" / "metadata.csv").resolve()
    output_dir = args.output.resolve()
    if not metadata_path.is_file():
        print(f"ERROR: Legacy metadata file does not exist: {metadata_path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        print(f"ERROR: Metadata file contains no records: {metadata_path}", file=sys.stderr)
        return 2

    secondary_records: dict[str, dict[str, str]] = {}
    if secondary_root:
        secondary_metadata = secondary_root / "data" / "metadata.csv"
        if secondary_metadata.is_file():
            with secondary_metadata.open("r", encoding="utf-8-sig", newline="") as handle:
                secondary_records = {
                    row.get("doc_id", "").strip(): row for row in csv.DictReader(handle)
                }
        else:
            print(f"WARNING: Secondary metadata not found: {secondary_metadata}", file=sys.stderr)

    session = _build_session(args.retries)
    results: list[FetchResult] = []
    for index, row in enumerate(records, start=1):
        document_id = row.get("doc_id", "UNKNOWN")
        print(f"[{index:02d}/{len(records):02d}] {document_id}: processing...", flush=True)
        result = fetch_record(
            row,
            legacy_root=legacy_root,
            secondary_root=secondary_root,
            secondary_row=secondary_records.get(document_id),
            output_dir=output_dir,
            session=session,
            timeout=args.timeout,
            max_bytes=args.max_mb * 1024 * 1024,
            overwrite=args.overwrite,
        )
        results.append(result)
        location = f" -> {result.destination}" if result.destination else ""
        print(f"    {result.status.upper()}{location}: {result.detail}", flush=True)

    counts = {status: sum(result.status == status for result in results) for status in {
        "copied", "downloaded", "skipped", "failed"
    }}
    print(
        "Summary: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)),
        flush=True,
    )
    return 1 if counts["failed"] else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy/download the legacy conservation corpus into V3.")
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument(
        "--secondary-root",
        type=Path,
        default=DEFAULT_SECONDARY_ROOT,
        help="Local shanged repository used for fallback metadata/files.",
    )
    parser.add_argument("--metadata", type=Path, help="Override the legacy metadata.csv path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-mb", type=int, default=150, help="Maximum response size per document.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing V3 corpus files.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(fetch_corpus(_parse_args()))
