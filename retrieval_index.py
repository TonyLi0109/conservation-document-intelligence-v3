"""Persisted lexical candidate indexes over unmodified canonical evidence.

The indexes only return source identifiers. Scoring/fusion metadata must stay
outside ``KnowledgeArtifact``. FTS5 external-content tables avoid duplicating
the canonical text, while SQL triggers update postings in the same transaction
as ingestion. Query text is tokenized and quoted; user FTS syntax is never run.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from database import KnowledgeStore


INDEX_VERSION = "v3-fts5-1"
CHUNK_INDEX = "retrieval_chunks_fts"
DOCUMENT_INDEX = "retrieval_documents_fts"
MAX_QUERY_TERMS = 64


def _match_query(query_text: str) -> str:
    if not isinstance(query_text, str):
        raise TypeError("query_text must be a string")
    # Operators, column selectors, quotes, punctuation and wildcard characters
    # are input text, not instructions. unicode61 handles case/diacritics and
    # Porter stemming in the persisted index. Scope/alias policy belongs above.
    bounded = query_text[:8192].casefold()
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+", bounded)))[:MAX_QUERY_TERMS]
    # Keep an explicitly quoted multiword lookup as an additional phrase, with
    # token alternatives preserving recall. Even phrase contents are tokenized,
    # so embedded quotes/operators cannot become FTS instructions.
    phrases = [" ".join(re.findall(r"[^\W_]+", phrase))
               for phrase in re.findall(r'"([^"\n]{2,200})"', bounded)[:8]]
    terms.extend(phrase for phrase in phrases if " " in phrase)
    return " OR ".join('"' + term + '"' for term in dict.fromkeys(terms))


def _create_triggers(connection, table: str, index: str, row_key: str, columns: tuple[str, ...]) -> None:
    # All SQL identifiers here are module-owned constants, never user input.
    column_sql = ",".join(columns)
    new_values = ",".join("new." + column for column in columns)
    old_values = ",".join("old." + column for column in columns)
    insert = f"INSERT INTO {index}(rowid,{column_sql}) VALUES(new.{row_key},{new_values});"
    delete = f"INSERT INTO {index}({index},rowid,{column_sql}) VALUES('delete',old.{row_key},{old_values});"
    revision = "UPDATE retrieval_index_state SET revision=revision+1 WHERE singleton=1;"
    for event, body in (("INSERT", insert), ("DELETE", delete), ("UPDATE", delete + insert)):
        update_columns = " OF " + ",".join((row_key, *columns)) if event == "UPDATE" else ""
        connection.execute(
            f"CREATE TRIGGER IF NOT EXISTS {index}_{event.lower()} "
            f"AFTER {event}{update_columns} ON {table} BEGIN {body}{revision} END"
        )


def ensure_retrieval_index(store: KnowledgeStore, force: bool = False) -> None:
    """Build once, reopen cheaply, and maintain through canonical SQL writes.

    A savepoint makes initialization/rebuild atomic and preserves any outer
    transaction. A readiness flag is set only after an independently committed
    initialization, so rolling back a caller's first build cannot stale it.
    ``force`` rebuilds derived postings without changing source IDs or vectors.
    """
    with store._lock:
        connection = store.connection
        if getattr(store, "_v3_retrieval_index_ready", None) == INDEX_VERSION and not force:
            return
        connection.execute("SAVEPOINT retrieval_index_setup")
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS retrieval_index_state ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                "index_version TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0, "
                "build_count INTEGER NOT NULL DEFAULT 0)"
            )
            connection.execute("INSERT OR IGNORE INTO retrieval_index_state(singleton,index_version) VALUES(1,'')")
            existing = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?,?)",
                (CHUNK_INDEX, DOCUMENT_INDEX),
            )}
            connection.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {CHUNK_INDEX} USING fts5("
                "title,original_text_chunk,document_id UNINDEXED, "
                "content='knowledge_artifacts',content_rowid='artifact_id', "
                "tokenize='porter unicode61')"
            )
            connection.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {DOCUMENT_INDEX} USING fts5("
                "title,agency,topic,year,document_id UNINDEXED, "
                "content='documents',content_rowid='rowid', "
                "tokenize='porter unicode61')"
            )
            _create_triggers(connection, "knowledge_artifacts", CHUNK_INDEX, "artifact_id",
                             ("title", "original_text_chunk", "document_id"))
            _create_triggers(connection, "documents", DOCUMENT_INDEX, "rowid",
                             ("title", "agency", "topic", "year", "document_id"))
            version = connection.execute("SELECT index_version FROM retrieval_index_state WHERE singleton=1").fetchone()[0]
            if force or version != INDEX_VERSION or len(existing) != 2:
                for index in (CHUNK_INDEX, DOCUMENT_INDEX):
                    connection.execute(f"INSERT INTO {index}({index}) VALUES('rebuild')")
                connection.execute(
                    "UPDATE retrieval_index_state SET index_version=?,build_count=build_count+1 WHERE singleton=1",
                    (INDEX_VERSION,),
                )
            connection.execute("RELEASE SAVEPOINT retrieval_index_setup")
        except Exception:
            connection.execute("ROLLBACK TO SAVEPOINT retrieval_index_setup")
            connection.execute("RELEASE SAVEPOINT retrieval_index_setup")
            raise
        if not connection.in_transaction:
            store._v3_retrieval_index_ready = INDEX_VERSION


def lexical_candidates(
    store: KnowledgeStore,
    query_text: str,
    top_k: int = 48,
    document_ids: Sequence[str] | None = None,
) -> list[int]:
    """BM25-ranked canonical artifact IDs, optionally within selected documents.

    The optional scope is parameterized and applied before LIMIT, allowing a
    long report to recover relevant interior chunks without loading its text.
    BM25 is an ordinal source for later fusion, not a calibrated confidence.
    """
    query = _match_query(query_text)
    if not query or top_k <= 0:
        return []
    scope = ""
    parameters: list[object] = [query]
    if document_ids is not None:
        if isinstance(document_ids, (str, bytes)):
            raise TypeError("document_ids must be a sequence of document IDs")
        docids = list(dict.fromkeys(document_ids))
        if not docids:
            return []
        if any(not isinstance(docid, str) for docid in docids):
            raise TypeError("document_ids must contain strings")
        scope = f" AND a.document_id IN ({','.join('?' for _ in docids)})"
        parameters.extend(docids)
    parameters.append(top_k)
    with store._lock:
        ensure_retrieval_index(store)
        return [int(row[0]) for row in store.connection.execute(
            f"SELECT {CHUNK_INDEX}.rowid FROM {CHUNK_INDEX} "
            f"JOIN knowledge_artifacts a ON a.artifact_id={CHUNK_INDEX}.rowid "
            f"WHERE {CHUNK_INDEX} MATCH ?{scope} "
            f"ORDER BY bm25({CHUNK_INDEX}),{CHUNK_INDEX}.rowid LIMIT ?",
            parameters,
        )]


def document_candidates(store: KnowledgeStore, query_text: str, top_k: int = 8) -> list[str]:
    """Find candidate reports by title/agency/topic/year independently of size.

    Metadata matches seed document-scoped chunk retrieval; they never establish
    evidence for an answer by themselves. This function adds no topic aliases
    and makes no lifecycle/currentness decisions.
    """
    query = _match_query(query_text)
    if not query or top_k <= 0:
        return []
    with store._lock:
        ensure_retrieval_index(store)
        return [str(row[0]) for row in store.connection.execute(
            f"SELECT d.document_id FROM {DOCUMENT_INDEX} "
            f"JOIN documents d ON d.rowid={DOCUMENT_INDEX}.rowid "
            f"WHERE {DOCUMENT_INDEX} MATCH ? "
            f"ORDER BY bm25({DOCUMENT_INDEX}),d.document_id LIMIT ?",
            (query, top_k),
        )]


def index_stats(store: KnowledgeStore) -> dict[str, object]:
    """Cheap diagnostic counts from postings, not a scan of canonical text."""
    with store._lock:
        ensure_retrieval_index(store)
        state = store.connection.execute(
            "SELECT index_version,revision,build_count FROM retrieval_index_state WHERE singleton=1"
        ).fetchone()
        return {
            "index_version": state[0], "revision": int(state[1]), "build_count": int(state[2]),
            "indexed_chunks": int(store.connection.execute(f"SELECT COUNT(*) FROM {CHUNK_INDEX}_docsize").fetchone()[0]),
            "indexed_documents": int(store.connection.execute(f"SELECT COUNT(*) FROM {DOCUMENT_INDEX}_docsize").fetchone()[0]),
        }
