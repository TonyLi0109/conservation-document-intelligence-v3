"""Persistent canonical SQLite storage with a NumPy exact-vector cache."""

from __future__ import annotations

import math
import hashlib
import json
import re
import shutil
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from config import SETTINGS
from data_models import DocumentSource, KnowledgeArtifact, is_knowledge_artifact


Embedding = Sequence[float]
DEFAULT_DATABASE_PATH = SETTINGS.storage.database_path
WIKI_ENTITY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Agency": (
        "Missouri Department of Conservation",
        "U.S. Army Corps of Engineers",
        "U.S. Department of the Interior",
        "U.S. Fish and Wildlife Service",
    ),
    "Habitat": ("Forest", "Marsh", "Wetland"),
    "Location": ("Great Lakes", "Missouri"),
    "Species": ("Invasive carp", "Silver carp", "Zebra mussel"),
    "Threat": ("Climate change", "Habitat loss", "Invasive species"),
}
KEYWORD_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)
DOCUMENT_QUERY_STOPWORDS = {
    "what", "which", "documents", "document", "sources", "source",
    "reports", "report", "plans", "plan", "discuss", "discusses",
    "mention", "mentions", "relevant", "most", "are", "is", "to",
    "the", "a", "an", "about", "find", "identify", "list",
}


def prepare_runtime_database(source_path: Path, runtime_root: Path) -> Path:
    """Copy the bundled read-only corpus to a writable runtime location.

    Streamlit Community Cloud mounts repository files read-only. The canonical
    database remains the deployment seed, while Wiki refreshes and other runtime
    writes use this per-process copy.
    """

    source = Path(source_path)
    destination_root = Path(runtime_root)
    if not source.is_file():
        raise FileNotFoundError(f"Bundled corpus database does not exist: {source}")
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / source.name
    shutil.copy2(source, destination)
    return destination


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create separate canonical provenance and embedding-cache tables."""
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS knowledge_artifacts (
            artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL CHECK (length(trim(document_id)) > 0),
            title TEXT NOT NULL CHECK (length(trim(title)) > 0),
            page_number TEXT NOT NULL CHECK (length(trim(page_number)) > 0),
            original_text_chunk TEXT NOT NULL CHECK (length(trim(original_text_chunk)) > 0),
            printed_page_label TEXT
        );
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            year TEXT NOT NULL DEFAULT '',
            agency TEXT NOT NULL DEFAULT '',
            topic TEXT NOT NULL DEFAULT '',
            source_url TEXT,
            resolved_url TEXT,
            local_filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS vector_embeddings (
            artifact_id INTEGER PRIMARY KEY,
            dimension INTEGER NOT NULL CHECK (dimension > 0),
            embedding BLOB NOT NULL,
            FOREIGN KEY (artifact_id) REFERENCES knowledge_artifacts(artifact_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS staged_knowledge_artifacts (
            artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL CHECK (length(trim(document_id)) > 0),
            title TEXT NOT NULL CHECK (length(trim(title)) > 0),
            page_number TEXT NOT NULL CHECK (length(trim(page_number)) > 0),
            original_text_chunk TEXT NOT NULL CHECK (length(trim(original_text_chunk)) > 0),
            printed_page_label TEXT
        );
        CREATE TABLE IF NOT EXISTS staged_vector_embeddings (
            artifact_id INTEGER PRIMARY KEY,
            dimension INTEGER NOT NULL CHECK (dimension > 0),
            embedding BLOB NOT NULL,
            FOREIGN KEY (artifact_id) REFERENCES staged_knowledge_artifacts(artifact_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_document_page
            ON knowledge_artifacts (document_id, page_number);
        CREATE TABLE IF NOT EXISTS compiled_knowledge (
            knowledge_id TEXT PRIMARY KEY,
            concept_key TEXT NOT NULL UNIQUE,
            concept_title TEXT NOT NULL,
            summary TEXT NOT NULL,
            generation_method TEXT NOT NULL,
            generation_version TEXT NOT NULL,
            model_name TEXT NOT NULL,
            generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS compiled_facts (
            knowledge_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            fact_text TEXT NOT NULL,
            PRIMARY KEY (knowledge_id, ordinal),
            FOREIGN KEY (knowledge_id) REFERENCES compiled_knowledge(knowledge_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS compiled_relationships (
            knowledge_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            entity_name TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            PRIMARY KEY (knowledge_id, ordinal),
            FOREIGN KEY (knowledge_id) REFERENCES compiled_knowledge(knowledge_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS compiled_relationship_evidence (
            knowledge_id TEXT NOT NULL,
            relationship_ordinal INTEGER NOT NULL,
            artifact_id INTEGER NOT NULL,
            exact_span TEXT NOT NULL,
            PRIMARY KEY (knowledge_id, relationship_ordinal),
            FOREIGN KEY (knowledge_id, relationship_ordinal)
                REFERENCES compiled_relationships(knowledge_id, ordinal) ON DELETE CASCADE,
            FOREIGN KEY (artifact_id) REFERENCES knowledge_artifacts(artifact_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS compiled_evidence (
            knowledge_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            artifact_id INTEGER NOT NULL,
            exact_span TEXT NOT NULL,
            PRIMARY KEY (knowledge_id, ordinal),
            FOREIGN KEY (knowledge_id) REFERENCES compiled_knowledge(knowledge_id) ON DELETE CASCADE,
            FOREIGN KEY (artifact_id) REFERENCES knowledge_artifacts(artifact_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_compiled_evidence_artifact
            ON compiled_evidence (artifact_id);
        CREATE TABLE IF NOT EXISTS document_ingestion_reports (
            document_id TEXT PRIMARY KEY,
            file_sha256 TEXT NOT NULL,
            total_pages INTEGER NOT NULL,
            extracted_pages INTEGER NOT NULL,
            empty_pages TEXT NOT NULL,
            failed_pages TEXT NOT NULL,
            fallback_pages TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        );
        """
    )
    for table in ("knowledge_artifacts", "staged_knowledge_artifacts"):
        columns = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if "printed_page_label" not in columns:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN printed_page_label TEXT"
            )
    connection.commit()


def _validated_vector(embedding: Embedding, expected_dimension: int | None = None) -> np.ndarray:
    """Return a finite, non-zero float32 vector of the expected dimension."""
    if isinstance(embedding, (str, bytes)) or not isinstance(embedding, (Sequence, np.ndarray)):
        raise TypeError("embedding must be a sequence of numbers")
    try:
        vector = np.asarray(embedding, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise TypeError("embedding must contain only numbers") from error
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("embedding must be a non-empty one-dimensional vector")
    if expected_dimension is not None and vector.size != expected_dimension:
        raise ValueError(f"embedding dimension {vector.size} does not match {expected_dimension}")
    if not np.isfinite(vector).all():
        raise ValueError("embedding values must be finite")
    if float(np.linalg.norm(vector)) == 0.0:
        raise ValueError("embedding must not be a zero vector")
    return vector


class ExactVectorStore:
    """Vectorized cosine index loaded from the persistent SQLite cache."""
    def __init__(self) -> None:
        self._ids = np.empty(0, dtype=np.int64)
        self._matrix = np.empty((0, 0), dtype=np.float32)
        self._lock = threading.RLock()

    @property
    def dimension(self) -> int | None:
        return int(self._matrix.shape[1]) if self._matrix.size else None

    def replace(self, artifact_ids: Sequence[int], embeddings: Sequence[Embedding]) -> None:
        if len(artifact_ids) != len(embeddings):
            raise ValueError("artifact IDs and embeddings must have equal lengths")
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("artifact IDs must be unique")
        with self._lock:
            if not embeddings:
                self._ids = np.empty(0, dtype=np.int64)
                self._matrix = np.empty((0, 0), dtype=np.float32)
                return
            vectors: list[np.ndarray] = []
            dimension: int | None = None
            for embedding in embeddings:
                vector = _validated_vector(embedding, dimension)
                dimension = int(vector.size)
                vectors.append(vector)
            matrix = np.vstack(vectors)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
            self._ids = np.asarray(artifact_ids, dtype=np.int64)
            self._matrix = matrix

    def search(self, query_embedding: Embedding, top_k: int) -> list[int]:
        """Return IDs ranked using vectorized cosine similarity."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        with self._lock:
            if not self._matrix.size:
                return []
            query = _validated_vector(query_embedding, self.dimension)
            query /= np.linalg.norm(query)
            scores = np.dot(self._matrix, query)
            count = min(top_k, int(scores.size))
            candidates = np.argpartition(scores, -count)[-count:]
            ranked = candidates[np.argsort(-scores[candidates], kind="stable")]
            return [int(self._ids[index]) for index in ranked]


class KnowledgeStore:
    """Persistent provenance repository with an eagerly loaded vector cache."""
    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH, *, vector_store: ExactVectorStore | None = None) -> None:
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.vector_store = vector_store or ExactVectorStore()
        self._lock = threading.RLock()
        initialize_schema(self.connection)
        self._load_vector_cache()

    @property
    def artifact_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM knowledge_artifacts").fetchone()[0])

    def list_documents(self) -> list[dict[str, object]]:
        """Return presentation-safe corpus summaries without exposing vectors."""

        with self._lock:
            rows = self.connection.execute(
                """
                SELECT a.document_id, a.title,
                       COUNT(*) AS chunk_count,
                       COUNT(DISTINCT a.page_number) AS indexed_pages,
                       COALESCE(d.year, '') AS year,
                       COALESCE(d.agency, '') AS agency,
                       COALESCE(d.topic, '') AS topic,
                       COALESCE(d.resolved_url, d.source_url, '') AS source_url
                FROM knowledge_artifacts AS a
                LEFT JOIN documents AS d ON d.document_id = a.document_id
                GROUP BY a.document_id, a.title, d.year, d.agency, d.topic,
                         d.resolved_url, d.source_url
                ORDER BY a.document_id
                """
            ).fetchall()
        return [
            {
                "Document ID": row["document_id"],
                "Title": row["title"],
                "Indexed pages": int(row["indexed_pages"]),
                "Chunks": int(row["chunk_count"]),
                "Year": row["year"],
                "Agency": row["agency"],
                "Topic": row["topic"],
                "Source URL": row["source_url"],
            }
            for row in rows
        ]

    def upsert_document_sources(self, sources: Sequence[DocumentSource]) -> int:
        """Synchronize trusted V3 catalog metadata without touching evidence text."""

        if any(not isinstance(source, DocumentSource) for source in sources):
            raise TypeError("sources must contain only DocumentSource objects")
        with self._lock, self.connection:
            self.connection.executemany(
                """
                INSERT INTO documents
                    (document_id, title, year, agency, topic, source_url,
                     resolved_url, local_filename, file_type, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(document_id) DO UPDATE SET
                    title=excluded.title, year=excluded.year,
                    agency=excluded.agency, topic=excluded.topic,
                    source_url=excluded.source_url,
                    resolved_url=excluded.resolved_url,
                    local_filename=excluded.local_filename,
                    file_type=excluded.file_type, updated_at=CURRENT_TIMESTAMP
                """,
                [
                    (s.document_id, s.title, s.year, s.agency, s.topic,
                     s.source_url, s.resolved_url, s.local_filename, s.file_type)
                    for s in sources
                ],
            )
        return len(sources)

    def record_ingestion_report(
        self,
        document_id: str,
        file_sha256: str,
        *,
        total_pages: int,
        extracted_pages: int,
        empty_pages: Sequence[int],
        failed_pages: Sequence[int],
        fallback_pages: Sequence[int],
        chunk_count: int,
    ) -> None:
        """Persist auditable source coverage separately from evidence content."""

        with self._lock, self.connection:
            self.connection.execute(
                """INSERT INTO document_ingestion_reports
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(document_id) DO UPDATE SET
                     file_sha256=excluded.file_sha256,
                     total_pages=excluded.total_pages,
                     extracted_pages=excluded.extracted_pages,
                     empty_pages=excluded.empty_pages,
                     failed_pages=excluded.failed_pages,
                     fallback_pages=excluded.fallback_pages,
                     chunk_count=excluded.chunk_count,
                     ingested_at=CURRENT_TIMESTAMP""",
                (document_id, file_sha256, total_pages, extracted_pages,
                 json.dumps(list(empty_pages)), json.dumps(list(failed_pages)),
                 json.dumps(list(fallback_pages)), chunk_count),
            )

    def set_document_page_labels(
        self, document_id: str, labels_by_physical_page: dict[str, str]
    ) -> int:
        """Backfill trusted PDF PageLabels without re-parsing or re-embedding text."""

        if not document_id.strip():
            raise ValueError("document_id is required")
        if any(not str(page).strip() or not str(label).strip()
               for page, label in labels_by_physical_page.items()):
            raise ValueError("page-label mappings must be non-empty strings")
        with self._lock, self.connection:
            self.connection.execute(
                "UPDATE knowledge_artifacts SET printed_page_label=NULL WHERE document_id=?",
                (document_id,),
            )
            updated = 0
            for physical_page, printed_label in labels_by_physical_page.items():
                cursor = self.connection.execute(
                    """UPDATE knowledge_artifacts SET printed_page_label=?
                       WHERE document_id=? AND page_number=?""",
                    (printed_label, document_id, physical_page),
                )
                updated += cursor.rowcount
        return updated

    def list_ingestion_reports(self) -> list[dict[str, object]]:
        """Return coverage metrics suitable for diagnostics and evaluation."""

        rows = self.connection.execute(
            "SELECT * FROM document_ingestion_reports ORDER BY document_id"
        ).fetchall()
        reports: list[dict[str, object]] = []
        for row in rows:
            total = int(row["total_pages"])
            extracted = int(row["extracted_pages"])
            reports.append(
                {
                    "Document ID": row["document_id"],
                    "Total pages": total,
                    "Extracted pages": extracted,
                    "Text coverage": extracted / total if total else 0.0,
                    "Processed coverage": (
                        (extracted + len(json.loads(row["empty_pages"]))) / total
                        if total else 0.0
                    ),
                    "Failed pages": len(json.loads(row["failed_pages"])),
                    "Fallback pages": len(json.loads(row["fallback_pages"])),
                    "Chunks": int(row["chunk_count"]),
                    "SHA-256": row["file_sha256"],
                    "Ingested at": row["ingested_at"],
                }
            )
        return reports

    def list_wiki_entities(self) -> dict[str, list[str]]:
        """Return curated legacy entities that occur in the canonical corpus.

        The legacy Wiki populated its dependent dropdowns from a generated
        ``wiki_pages`` table. V3 does not yet persist entity extraction output,
        so this method retains that proven vocabulary but exposes an entity only
        when its name occurs in V3-owned canonical text or titles.
        """

        available: dict[str, list[str]] = {}
        with self._lock:
            for entity_type, candidates in WIKI_ENTITY_CANDIDATES.items():
                matched: list[str] = []
                for entity in candidates:
                    exists = self.connection.execute(
                        """
                        SELECT 1
                        FROM knowledge_artifacts
                        WHERE INSTR(LOWER(original_text_chunk), LOWER(?)) > 0
                           OR INSTR(LOWER(title), LOWER(?)) > 0
                        LIMIT 1
                        """,
                        (entity, entity),
                    ).fetchone()
                    if exists is not None:
                        matched.append(entity)
                if matched:
                    available[entity_type] = matched
        return available

    def _load_vector_cache(self) -> None:
        rows = self.connection.execute(
            "SELECT artifact_id, dimension, embedding FROM vector_embeddings ORDER BY artifact_id"
        ).fetchall()
        ids: list[int] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            if vector.size != int(row["dimension"]):
                raise RuntimeError(f"cached embedding {row['artifact_id']} has invalid dimensions")
            ids.append(int(row["artifact_id"]))
            vectors.append(vector)
        self.vector_store.replace(ids, vectors)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def replace_corpus(self, artifacts: Sequence[KnowledgeArtifact], embeddings: Sequence[Embedding]) -> int:
        """Atomically replace canonical rows and their persisted vectors."""
        if len(artifacts) != len(embeddings):
            raise ValueError("artifacts and embeddings must have equal lengths")
        if any(not is_knowledge_artifact(item) for item in artifacts):
            raise TypeError("artifacts must contain only KnowledgeArtifact objects")
        vectors: list[np.ndarray] = []
        dimension: int | None = None
        for embedding in embeddings:
            vector = _validated_vector(embedding, dimension)
            dimension = int(vector.size)
            vectors.append(vector)

        with self._lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                self.connection.execute("DELETE FROM compiled_knowledge")
                self.connection.execute("DELETE FROM vector_embeddings")
                self.connection.execute("DELETE FROM knowledge_artifacts")
                artifact_ids: list[int] = []
                for artifact, vector in zip(artifacts, vectors, strict=True):
                    cursor = self.connection.execute(
                        """INSERT INTO knowledge_artifacts
                           (document_id, title, page_number, original_text_chunk, printed_page_label)
                           VALUES (?, ?, ?, ?, ?)""",
                        (artifact.document_id, artifact.title, artifact.page_number,
                         artifact.original_text_chunk, artifact.printed_page_label),
                    )
                    artifact_id = int(cursor.lastrowid)
                    artifact_ids.append(artifact_id)
                    self.connection.execute(
                        "INSERT INTO vector_embeddings (artifact_id, dimension, embedding) VALUES (?, ?, ?)",
                        (artifact_id, int(vector.size), vector.tobytes()),
                    )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
            self.vector_store.replace(artifact_ids, vectors)
        return len(artifacts)

    def begin_rebuild(self) -> None:
        """Clear only staging rows, leaving the current searchable corpus intact."""

        with self._lock, self.connection:
            self.connection.execute("DELETE FROM staged_vector_embeddings")
            self.connection.execute("DELETE FROM staged_knowledge_artifacts")

    def stage_batch(
        self,
        artifacts: Sequence[KnowledgeArtifact],
        embeddings: Sequence[Embedding],
    ) -> int:
        """Persist one bounded ingestion batch without growing process memory."""

        if len(artifacts) != len(embeddings):
            raise ValueError("artifacts and embeddings must have equal lengths")
        if not artifacts:
            return 0
        if any(not is_knowledge_artifact(item) for item in artifacts):
            raise TypeError("artifacts must contain only KnowledgeArtifact objects")
        expected_dimension: int | None = None
        staged_dimension = self.connection.execute(
            "SELECT dimension FROM staged_vector_embeddings LIMIT 1"
        ).fetchone()
        if staged_dimension is not None:
            expected_dimension = int(staged_dimension[0])
        vectors: list[np.ndarray] = []
        for embedding in embeddings:
            vector = _validated_vector(embedding, expected_dimension)
            expected_dimension = int(vector.size)
            vectors.append(vector)
        with self._lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                for artifact, vector in zip(artifacts, vectors, strict=True):
                    cursor = self.connection.execute(
                        """INSERT INTO staged_knowledge_artifacts
                           (document_id, title, page_number, original_text_chunk, printed_page_label)
                           VALUES (?, ?, ?, ?, ?)""",
                        (artifact.document_id, artifact.title, artifact.page_number,
                         artifact.original_text_chunk, artifact.printed_page_label),
                    )
                    self.connection.execute(
                        "INSERT INTO staged_vector_embeddings (artifact_id, dimension, embedding) VALUES (?, ?, ?)",
                        (int(cursor.lastrowid), int(vector.size), vector.tobytes()),
                    )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        return len(artifacts)

    def commit_rebuild(self) -> int:
        """Atomically promote the complete staged corpus and reload its vector index."""

        staged_count = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM staged_knowledge_artifacts"
            ).fetchone()[0]
        )
        if staged_count == 0:
            raise RuntimeError("cannot commit an empty staged corpus")
        with self._lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                self.connection.execute("DELETE FROM compiled_knowledge")
                self.connection.execute("DELETE FROM vector_embeddings")
                self.connection.execute("DELETE FROM knowledge_artifacts")
                self.connection.execute(
                    """
                    INSERT INTO knowledge_artifacts
                        (artifact_id, document_id, title, page_number,
                         original_text_chunk, printed_page_label)
                    SELECT artifact_id, document_id, title, page_number,
                           original_text_chunk, printed_page_label
                    FROM staged_knowledge_artifacts ORDER BY artifact_id
                    """
                )
                self.connection.execute(
                    """
                    INSERT INTO vector_embeddings (artifact_id, dimension, embedding)
                    SELECT artifact_id, dimension, embedding
                    FROM staged_vector_embeddings ORDER BY artifact_id
                    """
                )
                self.connection.execute("DELETE FROM staged_vector_embeddings")
                self.connection.execute("DELETE FROM staged_knowledge_artifacts")
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
            self._load_vector_cache()
        return staged_count

    def abort_rebuild(self) -> None:
        """Discard an incomplete staged rebuild while retaining the live index."""

        self.begin_rebuild()

    def ingest_chunk(self, artifact: KnowledgeArtifact, embedding: Embedding) -> int:
        """Append one artifact and embedding for incremental callers."""
        if not is_knowledge_artifact(artifact):
            raise TypeError("artifact must be a KnowledgeArtifact")
        vector = _validated_vector(embedding, self.vector_store.dimension)
        with self._lock:
            try:
                cursor = self.connection.execute(
                    """INSERT INTO knowledge_artifacts
                       (document_id, title, page_number, original_text_chunk, printed_page_label)
                       VALUES (?, ?, ?, ?, ?)""",
                    (artifact.document_id, artifact.title, artifact.page_number,
                     artifact.original_text_chunk, artifact.printed_page_label),
                )
                artifact_id = int(cursor.lastrowid)
                self.connection.execute(
                    "INSERT INTO vector_embeddings (artifact_id, dimension, embedding) VALUES (?, ?, ?)",
                    (artifact_id, int(vector.size), vector.tobytes()),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
            self._load_vector_cache()
        return artifact_id

    def _keyword_search(self, query_text: str, top_k: int) -> list[int]:
        """Rank chunks by safe, case-insensitive phrase and term occurrence."""

        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string for keyword search")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        normalized_query = " ".join(query_text.casefold().split())
        terms = list(dict.fromkeys(KEYWORD_PATTERN.findall(normalized_query)))
        if not terms:
            return []
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT artifact_id, document_id, page_number, title,
                       original_text_chunk
                FROM knowledge_artifacts
                """
            ).fetchall()

        ranked: list[tuple[float, str, str, int]] = []
        for row in rows:
            text = str(row["original_text_chunk"]).casefold()
            title = str(row["title"]).casefold()
            phrase_hits = text.count(normalized_query)
            term_hits = [text.count(term) for term in terms]
            title_hits = sum(title.count(term) for term in terms)
            covered_terms = sum(hit > 0 for hit in term_hits)
            if not phrase_hits and not covered_terms and not title_hits:
                continue
            # Exact phrases dominate; broad multi-term coverage outranks repeated
            # isolated words. log1p prevents boilerplate repetition dominating.
            score = (
                phrase_hits * 20.0
                + (covered_terms / len(terms)) * 8.0
                + sum(math.log1p(hit) for hit in term_hits)
                + title_hits * 3.0
            )
            ranked.append(
                (
                    score,
                    str(row["document_id"]),
                    str(row["page_number"]),
                    int(row["artifact_id"]),
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        return [item[3] for item in ranked[:top_k]]

    def retrieve_document_matches(
        self, query_text: str, top_k: int = 10
    ) -> list[KnowledgeArtifact]:
        """Run an explicit corpus-wide document listing with one result per source.

        Unlike chunk Top-K retrieval, this route scores every canonical chunk,
        aggregates matches by document, and cannot lose document diversity merely
        because one report contributes many highly similar chunks.
        """

        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be positive")
        normalized = " ".join(query_text.casefold().split())
        terms = [
            term for term in dict.fromkeys(KEYWORD_PATTERN.findall(normalized))
            if term not in DOCUMENT_QUERY_STOPWORDS
        ]
        if not terms:
            return []
        subject_phrase = " ".join(terms)
        rows = self.connection.execute(
            """SELECT artifact_id, document_id, title, original_text_chunk
               FROM knowledge_artifacts"""
        ).fetchall()
        best_by_document: dict[str, tuple[float, int, int]] = {}
        for row in rows:
            text = str(row["original_text_chunk"]).casefold()
            title = str(row["title"]).casefold()
            phrase_hits = text.count(subject_phrase)
            term_hits = [text.count(term) for term in terms]
            covered = sum(hit > 0 for hit in term_hits)
            title_covered = sum(term in title for term in terms)
            # Multi-term document listings require either the exact subject
            # phrase or all subject terms in one canonical evidence chunk.
            if not phrase_hits and covered < len(terms):
                continue
            score = (
                phrase_hits * 25.0
                + covered * 3.0
                + title_covered * 6.0
                + sum(math.log1p(hit) for hit in term_hits)
            )
            document_id = str(row["document_id"])
            previous = best_by_document.get(document_id)
            total_matches = (previous[2] if previous else 0) + max(phrase_hits, 1)
            candidate = (score, int(row["artifact_id"]), total_matches)
            if previous is None or score > previous[0]:
                best_by_document[document_id] = candidate
            else:
                best_by_document[document_id] = (
                    previous[0], previous[1], total_matches
                )
        ranked = sorted(
            best_by_document.items(),
            key=lambda item: (-item[1][0], -item[1][2], item[0]),
        )[:top_k]
        return self._artifacts_by_ranked_ids([value[1] for _, value in ranked])

    def _artifacts_by_ranked_ids(self, artifact_ids: Sequence[int]) -> list[KnowledgeArtifact]:
        """Rehydrate trusted canonical rows while preserving retrieval rank."""

        if not artifact_ids:
            return []
        placeholders = ", ".join("?" for _ in artifact_ids)
        with self._lock:
            rows = self.connection.execute(
                f"""SELECT a.artifact_id, a.document_id, a.title, a.page_number,
                            a.original_text_chunk,
                            a.printed_page_label,
                            COALESCE(d.resolved_url, d.source_url) AS source_url
                     FROM knowledge_artifacts AS a
                     LEFT JOIN documents AS d ON d.document_id = a.document_id
                     WHERE a.artifact_id IN ({placeholders})""",
                artifact_ids,
            ).fetchall()
        rows_by_id = {int(row["artifact_id"]): row for row in rows}
        missing = [item for item in artifact_ids if item not in rows_by_id]
        if missing:
            raise RuntimeError(f"vector cache references missing artifact IDs: {missing}")
        return [
            KnowledgeArtifact(
                document_id=rows_by_id[item]["document_id"],
                title=rows_by_id[item]["title"],
                page_number=rows_by_id[item]["page_number"],
                original_text_chunk=rows_by_id[item]["original_text_chunk"],
                source_url=rows_by_id[item]["source_url"],
                printed_page_label=rows_by_id[item]["printed_page_label"],
            )
            for item in artifact_ids
        ]

    def retrieve(
        self,
        query_embedding: Embedding | None,
        top_k: int,
        *,
        method: str = "semantic",
        query_text: str | None = None,
    ) -> list[KnowledgeArtifact]:
        """Retrieve canonical evidence using semantic or keyword ranking.

        Semantic retrieval retains the existing NumPy cosine index. Keyword
        retrieval requires ``query_text`` and never evaluates user input as SQL
        or as an unescaped regular expression.
        """

        normalized_method = method.strip().casefold().replace("_", " ")
        if normalized_method in {"semantic", "semantic search"}:
            if query_embedding is None:
                raise ValueError("query_embedding is required for semantic search")
            artifact_ids = self.vector_store.search(query_embedding, top_k)
        elif normalized_method in {"keyword", "keyword search"}:
            artifact_ids = self._keyword_search(query_text or "", top_k)
        else:
            raise ValueError("method must be 'semantic' or 'keyword'")
        return self._artifacts_by_ranked_ids(artifact_ids)

    def save_compiled_concept(
        self,
        topic_query: str,
        concept: dict[str, object],
        artifacts: dict[str, KnowledgeArtifact],
        *,
        model_name: str,
        generation_version: str,
        generation_method: str = "retrieval_grounded_llm",
    ) -> str:
        """Persist a validated derived knowledge item and its exact evidence links."""

        concept_key = " ".join(topic_query.casefold().split())
        knowledge_id = "KC-" + hashlib.sha256(concept_key.encode("utf-8")).hexdigest()[:16].upper()
        facts = concept.get("important_facts", [])
        relationships = concept.get("related_entities", [])
        evidence = concept.get("supporting_evidence", [])
        if not isinstance(facts, list) or not isinstance(relationships, list) or not isinstance(evidence, list):
            raise TypeError("compiled concept collections must be lists")

        resolved_evidence: list[tuple[int, str]] = []
        for item in evidence:
            if not isinstance(item, dict):
                raise TypeError("supporting evidence entries must be objects")
            handle, span = item.get("evidence_id"), item.get("exact_span")
            artifact = artifacts.get(str(handle))
            if artifact is None or not isinstance(span, str) or span not in artifact.original_text_chunk:
                raise ValueError("compiled evidence failed canonical verbatim validation")
            row = self.connection.execute(
                """SELECT artifact_id FROM knowledge_artifacts
                   WHERE document_id=? AND title=? AND page_number=?
                     AND original_text_chunk=? LIMIT 1""",
                (artifact.document_id, artifact.title, artifact.page_number,
                 artifact.original_text_chunk),
            ).fetchone()
            if row is None:
                raise ValueError(f"compiled evidence is not canonical: {handle}")
            resolved_evidence.append((int(row[0]), span))

        with self._lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                self.connection.execute(
                    """INSERT INTO compiled_knowledge
                       (knowledge_id, concept_key, concept_title, summary,
                        generation_method, generation_version, model_name, generated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                       ON CONFLICT(concept_key) DO UPDATE SET
                         concept_title=excluded.concept_title, summary=excluded.summary,
                         generation_method=excluded.generation_method,
                         generation_version=excluded.generation_version,
                         model_name=excluded.model_name, generated_at=CURRENT_TIMESTAMP""",
                    (knowledge_id, concept_key, str(concept["concept_title"]),
                     str(concept["summary"]), generation_method,
                     generation_version, model_name),
                )
                for table in (
                    "compiled_facts", "compiled_relationship_evidence",
                    "compiled_relationships", "compiled_evidence"
                ):
                    self.connection.execute(f"DELETE FROM {table} WHERE knowledge_id=?", (knowledge_id,))
                self.connection.executemany(
                    "INSERT INTO compiled_facts VALUES (?, ?, ?)",
                    [(knowledge_id, i, str(fact)) for i, fact in enumerate(facts, 1)],
                )
                self.connection.executemany(
                    "INSERT INTO compiled_relationships VALUES (?, ?, ?, ?)",
                    [(knowledge_id, i, str(item["entity_name"]), str(item["relationship_type"]))
                     for i, item in enumerate(relationships, 1)],
                )
                relationship_evidence: list[tuple[str, int, int, str]] = []
                for ordinal, item in enumerate(relationships, 1):
                    handle = str(item["evidence_id"])
                    span = str(item["exact_span"])
                    artifact = artifacts.get(handle)
                    if artifact is None or span not in artifact.original_text_chunk:
                        raise ValueError("relationship evidence failed canonical validation")
                    row = self.connection.execute(
                        """SELECT artifact_id FROM knowledge_artifacts
                           WHERE document_id=? AND title=? AND page_number=?
                             AND original_text_chunk=? LIMIT 1""",
                        (artifact.document_id, artifact.title, artifact.page_number,
                         artifact.original_text_chunk),
                    ).fetchone()
                    if row is None:
                        raise ValueError("relationship evidence is not canonical")
                    relationship_evidence.append(
                        (knowledge_id, ordinal, int(row[0]), span)
                    )
                self.connection.executemany(
                    "INSERT INTO compiled_relationship_evidence VALUES (?, ?, ?, ?)",
                    relationship_evidence,
                )
                self.connection.executemany(
                    "INSERT INTO compiled_evidence VALUES (?, ?, ?, ?)",
                    [(knowledge_id, i, artifact_id, span)
                     for i, (artifact_id, span) in enumerate(resolved_evidence, 1)],
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        return knowledge_id

    def list_compiled_concepts(self) -> list[dict[str, object]]:
        """Return compact metadata for reusable compiled knowledge items."""

        rows = self.connection.execute(
            """SELECT k.knowledge_id, k.concept_title, k.generation_version,
                      k.model_name, k.generated_at, COUNT(e.ordinal) AS evidence_count
               FROM compiled_knowledge k LEFT JOIN compiled_evidence e
                 ON e.knowledge_id=k.knowledge_id
               GROUP BY k.knowledge_id ORDER BY k.concept_title"""
        ).fetchall()
        return [dict(row) for row in rows]

    def get_compiled_concept(self, topic_query: str) -> dict[str, object] | None:
        """Rehydrate a reusable Wiki result and revalidate every stored span."""

        concept_key = " ".join(topic_query.casefold().split())
        knowledge = self.connection.execute(
            "SELECT * FROM compiled_knowledge WHERE concept_key=?", (concept_key,)
        ).fetchone()
        if knowledge is None:
            return None
        facts = self.connection.execute(
            "SELECT fact_text FROM compiled_facts WHERE knowledge_id=? ORDER BY ordinal",
            (knowledge["knowledge_id"],),
        ).fetchall()
        relationships = self.connection.execute(
            """SELECT r.entity_name, r.relationship_type, re.artifact_id,
                      re.exact_span
               FROM compiled_relationships r
               LEFT JOIN compiled_relationship_evidence re
                 ON re.knowledge_id=r.knowledge_id
                AND re.relationship_ordinal=r.ordinal
               WHERE r.knowledge_id=? ORDER BY r.ordinal""",
            (knowledge["knowledge_id"],),
        ).fetchall()
        evidence_rows = self.connection.execute(
            """SELECT e.exact_span, a.artifact_id, a.document_id, a.title,
                      a.page_number, a.original_text_chunk, a.printed_page_label,
                      COALESCE(d.resolved_url, d.source_url) AS source_url
               FROM compiled_evidence e
               JOIN knowledge_artifacts a ON a.artifact_id=e.artifact_id
               LEFT JOIN documents d ON d.document_id=a.document_id
               WHERE e.knowledge_id=? ORDER BY e.ordinal""",
            (knowledge["knowledge_id"],),
        ).fetchall()
        if not evidence_rows:
            return None

        handles_by_artifact: dict[int, str] = {}
        artifacts: dict[str, KnowledgeArtifact] = {}
        supporting_evidence: list[dict[str, str]] = []
        for row in evidence_rows:
            span = str(row["exact_span"])
            chunk = str(row["original_text_chunk"])
            if span not in chunk:
                raise RuntimeError(
                    f"Stored compiled evidence no longer validates for artifact {row['artifact_id']}"
                )
            artifact_id = int(row["artifact_id"])
            handle = handles_by_artifact.setdefault(
                artifact_id, f"K{len(handles_by_artifact) + 1}"
            )
            artifacts.setdefault(
                handle,
                KnowledgeArtifact(
                    document_id=row["document_id"],
                    title=row["title"],
                    page_number=row["page_number"],
                    original_text_chunk=chunk,
                    source_url=row["source_url"],
                    printed_page_label=row["printed_page_label"],
                ),
            )
            supporting_evidence.append({"exact_span": span, "evidence_id": handle})

        related_entities: list[dict[str, str]] = []
        for row in relationships:
            artifact_id = row["artifact_id"]
            if artifact_id is None:
                continue
            handle = handles_by_artifact.get(int(artifact_id))
            if handle is None:
                artifact_rows = self._artifacts_by_ranked_ids([int(artifact_id)])
                if not artifact_rows:
                    continue
                handle = f"K{len(handles_by_artifact) + 1}"
                handles_by_artifact[int(artifact_id)] = handle
                artifacts[handle] = artifact_rows[0]
            span = str(row["exact_span"])
            if span not in artifacts[handle].original_text_chunk:
                raise RuntimeError("Stored relationship evidence no longer validates")
            related_entities.append(
                {"entity_name": row["entity_name"],
                 "relationship_type": row["relationship_type"],
                 "evidence_id": handle, "exact_span": span}
            )

        return {
            "concept": {
                "concept_title": knowledge["concept_title"],
                "summary": knowledge["summary"],
                "important_facts": [row["fact_text"] for row in facts],
                "related_entities": related_entities,
                "supporting_evidence": supporting_evidence,
            },
            "artifacts": artifacts,
            "knowledge_id": knowledge["knowledge_id"],
            "generation_version": knowledge["generation_version"],
            "model_name": knowledge["model_name"],
            "cached": True,
        }

    def invalidate_compiled_knowledge(self, document_id: str) -> int:
        """Remove derived items supported by a document that is being replaced."""

        rows = self.connection.execute(
            """SELECT DISTINCT e.knowledge_id FROM compiled_evidence e
               JOIN knowledge_artifacts a ON a.artifact_id=e.artifact_id
               WHERE a.document_id=?""",
            (document_id,),
        ).fetchall()
        with self._lock, self.connection:
            self.connection.executemany(
                "DELETE FROM compiled_knowledge WHERE knowledge_id=?",
                [(row["knowledge_id"],) for row in rows],
            )
        return len(rows)

    def search_compiled_knowledge(self, query_text: str, top_k: int = 5) -> list[dict[str, object]]:
        """Search reusable knowledge independently of raw evidence retrieval."""

        if not query_text.strip() or top_k <= 0:
            raise ValueError("query_text and a positive top_k are required")
        pattern = f"%{query_text.strip()}%"
        rows = self.connection.execute(
            """SELECT knowledge_id, concept_title, summary, generation_version,
                      model_name, generated_at
               FROM compiled_knowledge
               WHERE concept_title LIKE ? OR summary LIKE ?
               ORDER BY concept_title LIMIT ?""",
            (pattern, pattern, top_k),
        ).fetchall()
        return [dict(row) for row in rows]


def ingest_chunk(store: KnowledgeStore, artifact: KnowledgeArtifact, embedding: Embedding) -> int:
    return store.ingest_chunk(artifact, embedding)


def retrieve(store: KnowledgeStore, query_embedding: Embedding, top_k: int) -> list[KnowledgeArtifact]:
    return store.retrieve(query_embedding, top_k)
