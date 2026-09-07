"""Browser-owned conversation archives; no shared visitor history on the server."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import streamlit.components.v1 as components


archive_component = components.declare_component(
    "v3_conversation_archive", path=str(Path(__file__).parent / "components" / "conversation_archive")
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_conversation(book: dict) -> str:
    identifier, timestamp = str(uuid4()), now()
    book["conversations"][identifier] = {
        "conversation_id": identifier, "title": "New conversation",
        "created_at": timestamp, "updated_at": timestamp, "messages": [],
    }
    book["active_id"] = identifier
    return identifier


def empty_book() -> dict:
    book = {"version": 1, "conversations": {}, "active_id": ""}
    new_conversation(book)
    return book


def append_message(book: dict, identifier: str, message: dict) -> None:
    conversation = book["conversations"][identifier]
    timestamp = now()
    if message["role"] == "user" and not conversation["messages"]:
        conversation["title"] = " ".join(message["content"].split())[:70]
    conversation["messages"].append({**message, "timestamp": timestamp})
    conversation["updated_at"] = timestamp


def export_book(book: dict) -> dict:
    conversations = {}
    for identifier, conversation in book["conversations"].items():
        messages = []
        for message in conversation["messages"]:
            references = [{"document_id": source.document_id, "page_number": source.page_number,
                           "sha256": hashlib.sha256(source.original_text_chunk.encode()).hexdigest()}
                          for source in message.get("sources", [])]
            # Preserve unresolved references across corpus replacement, without
            # treating browser-supplied source metadata as canonical evidence.
            references += message.get("unavailable_source_refs", [])
            messages.append({**{k: v for k, v in message.items() if k not in {"sources", "unavailable_source_refs"}},
                             "source_refs": references})
        conversations[identifier] = {**conversation, "messages": messages}
    return {**book, "conversations": conversations}


def restore_book(payload: object, store) -> dict:
    if payload is None:
        return empty_book()
    if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(payload.get("conversations"), dict):
        raise ValueError("Unrecognized conversation archive")
    if len(json.dumps(payload)) > 5_000_000:
        raise ValueError("Conversation archive exceeds the supported size")
    book = {"version": 1, "conversations": {}, "active_id": payload.get("active_id")}
    cache = {}
    for identifier, thread in payload["conversations"].items():
        if not isinstance(thread, dict) or thread.get("conversation_id") != identifier:
            raise ValueError("Invalid conversation identity")
        if not all(isinstance(thread.get(k), str) for k in ("title", "created_at", "updated_at")):
            raise ValueError("Invalid conversation metadata")
        if not isinstance(thread.get("messages"), list):
            raise ValueError("Invalid message history")
        messages = []
        for message in thread["messages"]:
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"} or not isinstance(message.get("content"), str):
                raise ValueError("Invalid message")
            restored = {k: message[k] for k in ("role", "content", "timestamp", "preamble", "context") if k in message}
            sources, missing = [], []
            for reference in message.get("source_refs", []):
                key = (reference["document_id"], reference["page_number"], reference["sha256"])
                if key not in cache:
                    cache[key] = store.resolve_source_reference(*key)
                if cache[key] is None:
                    missing.append(reference)
                else:
                    sources.append(cache[key])
            restored.update(sources=sources, unavailable_source_refs=missing)
            messages.append(restored)
        book["conversations"][identifier] = {**thread, "messages": messages}
    if not book["conversations"]:
        new_conversation(book)
    elif book["active_id"] not in book["conversations"]:
        book["active_id"] = next(iter(book["conversations"]))
    return book
