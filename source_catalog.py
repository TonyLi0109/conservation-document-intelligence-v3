"""V3-owned loader for trusted document-level source metadata."""

from __future__ import annotations

import csv
from pathlib import Path

from config import SETTINGS
from data_models import DocumentSource


def load_source_catalog(
    path: str | Path = SETTINGS.storage.source_catalog_path,
) -> dict[str, DocumentSource]:
    """Load and strictly validate the independent V3 source registry."""

    catalog_path = Path(path)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"V3 source catalog does not exist: {catalog_path}")
    sources: dict[str, DocumentSource] = {}
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            document_id = (row.get("document_id") or "").strip().upper()
            if document_id in sources:
                raise ValueError(f"Duplicate {document_id} at catalog line {line_number}")
            source = DocumentSource(
                document_id=document_id,
                title=(row.get("title") or "").strip(),
                year=(row.get("year") or "").strip(),
                agency=(row.get("agency") or "").strip(),
                topic=(row.get("topic") or "").strip(),
                source_url=(row.get("source_url") or "").strip() or None,
                resolved_url=(row.get("resolved_url") or "").strip() or None,
                local_filename=(row.get("local_filename") or "").strip(),
                file_type=(row.get("file_type") or "").strip().lower(),
            )
            sources[document_id] = source
    if not sources:
        raise ValueError("V3 source catalog is empty")
    return sources
