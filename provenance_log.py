"""Machine-readable claim provenance logging with fail-open I/O behavior."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import threading
from typing import Iterable

from data_models import ClaimValidation
from pipeline_tracer import capture

LOGGER = logging.getLogger(__name__)
_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class ClaimProvenanceRecord:
    """Professor Shang's required per-claim provenance record."""

    claim_id: str
    claim_text: str
    doc_id: str | None
    printed_page: str | None
    pdf_page: str | None
    supporting_evidence_span: str | None
    validation_status: str


def _records_for_validation(item: ClaimValidation) -> list[ClaimProvenanceRecord]:
    sources = item.sources or (None,)
    spans = item.supporting_spans or (None,)
    records: list[ClaimProvenanceRecord] = []
    for source in sources:
        matching = (
            tuple(span for span in spans if source and span and span in source.original_text_chunk)
            or (spans if source is None else (None,))
        )
        for span in matching:
            records.append(ClaimProvenanceRecord(
                claim_id=item.claim_id,
                claim_text=item.claim_text,
                doc_id=source.document_id if source else None,
                printed_page=source.printed_page_label if source else None,
                pdf_page=source.page_number if source else None,
                supporting_evidence_span=span,
                validation_status=item.status.value,
            ))
    return records


class ProvenanceLogger:
    """Emit JSON Lines without allowing observability failures to alter answers.

    Set V3_PROVENANCE_LOG to a private local JSONL path. Records are also
    captured by PipelineTracer whenever request tracing is active.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        configured = path if path is not None else os.environ.get("V3_PROVENANCE_LOG")
        self.path = Path(configured) if configured else None

    def emit(self, validations: Iterable[ClaimValidation]) -> list[dict[str, object]]:
        records = [asdict(record) for item in validations
                   for record in _records_for_validation(item)]
        for record in records:
            capture("claim_provenance", record)
        if not self.path or not records:
            return records
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _WRITE_LOCK, self.path.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        except OSError:
            LOGGER.exception("Could not append claim provenance log")
        return records
