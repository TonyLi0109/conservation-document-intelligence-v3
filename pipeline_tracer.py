"""Opt-in, request-local pipeline snapshots. No provider calls or ranking changes."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from functools import wraps
import json
import logging
import os
from pathlib import Path
import threading
from uuid import uuid4

STAGES = (
    "original_query", "rewritten_query", "initial_retrieval", "reranked_evidence",
    "llm_context_payload", "draft_answer", "validation_results", "final_output",
)
_current = ContextVar("pipeline_tracer", default=None)


def _encode(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Trace requires an explicit JSON adapter for {type(value).__name__}")


class PipelineTracer:
    """One instance per request. Export errors never replace the pipeline result.

    Values are copied at capture time. Repeated stages retain every event;
    stages holds the latest snapshot. Unvisited stages are explicitly marked.
    Full prompts/text are intentionally retained: use a local private directory.
    """

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.trace_id = uuid4().hex
        self.path = self.output_dir / f"{self.trace_id}.json"
        self.data = {"schema_version": 1, "trace_id": self.trace_id,
                     "started_at": datetime.now(timezone.utc).isoformat(),
                     "stages": {s: {"status": "not_reached", "value": None} for s in STAGES},
                     "events": [], "errors": []}
        self._lock = threading.RLock()
        self._search_cursor = 0

    def capture(self, stage, value):
        try:
            if stage == 'initial_retrieval':
                value = {**value, 'raw_searches': [e for e in self.data['events'][self._search_cursor:]
                         if e['stage'] in {'dense_search', 'lexical_search'}]}
                self._search_cursor = len(self.data['events'])
            snapshot = json.loads(json.dumps(value, default=_encode, ensure_ascii=False, allow_nan=False))
            with self._lock:
                event = {"sequence": len(self.data["events"]), "stage": stage,
                         "at": datetime.now(timezone.utc).isoformat(), "value": snapshot}
                self.data["events"].append(event)
                if stage in STAGES:
                    self.data["stages"][stage] = {"status": "captured", "value": snapshot}
        except Exception as error:
            with self._lock:
                self.data["errors"].append({"stage": stage, "error": str(error)})
            logging.warning("Pipeline trace snapshot failed for %s", stage)

    def __enter__(self):
        self._token = _current.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        _current.reset(self._token)
        if exc is not None:
            self.capture("pipeline_exception", {"type": exc_type.__name__, "message": str(exc)})
        self.data["finished_at"] = datetime.now(timezone.utc).isoformat()
        for record in self.data["stages"].values():
            if record["status"] == "not_reached":
                record["reason"] = "This route did not reach this hook; inspect route/exception events."
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(self.data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            os.replace(temporary, self.path)
        except Exception:
            logging.exception("Could not export pipeline trace %s", self.trace_id)
        return False


def active():
    return _current.get() is not None


def capture(stage, value):
    tracer = _current.get()
    if tracer is not None:
        tracer.capture(stage, value)


def trace_pipeline(function):
    """Trace existing backend calls using V3_TRACE_DIR or an explicit context."""
    @wraps(function)
    def wrapped(question, *args, **kwargs):
        def run():
            capture("original_query", question)
            result = function(question, *args, **kwargs)
            from output_formatter import normalize_user_facing_answer
            answer = normalize_user_facing_answer(
                result[0], has_validated_findings=bool(result[2])
            )
            result = (answer, result[1], result[2])
            capture("final_output", {"answer": result[0], "preamble": result[1], "sources": result[2]})
            return result
        if active():
            return run()
        directory = os.environ.get("V3_TRACE_DIR")
        if directory:
            with PipelineTracer(directory):
                return run()
        return run()
    return wrapped
