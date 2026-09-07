"""Declarative datasets and disposable canonical/fixture stores."""

from contextlib import closing, contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

from data_models import DocumentSource, KnowledgeArtifact
from database import KnowledgeStore

DATASET_DIR = Path(__file__).parent / "datasets"


def load_dataset(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("dataset_id"), str):
        raise ValueError("Dataset needs a dataset_id")
    if type(payload.get("version")) is not int or payload["version"] < 1:
        raise ValueError("Dataset needs a positive integer version")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset needs nonempty cases")
    ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(ids) != len(cases) or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Case IDs must be nonempty and unique")
    return payload


def fingerprint(path):
    if Path(path).suffix == ".json":
        # Ignore platform line endings/formatting while preserving all case data.
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def corpus_copy(path):
    """Yield a disposable snapshot and its pre-evaluation binary fingerprint.

    SQLite backup includes committed WAL contents. Hash the closed backup before
    KnowledgeStore initialization or evaluation can write to it, rather than
    hashing a source file that another writer may subsequently change.
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError("Corpus database does not exist")
    with tempfile.TemporaryDirectory(prefix="v3-eval-corpus-") as directory:
        target = Path(directory) / "corpus.db"
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as reader:
            with closing(sqlite3.connect(target)) as writer:
                reader.backup(writer)
        snapshot_fingerprint = fingerprint(target)
        with KnowledgeStore(target) as store:
            store.evaluation_snapshot_fingerprint = snapshot_fingerprint
            yield store


@contextmanager
def fixture_store(path=DATASET_DIR / "fixture_corpus.json"):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("synthetic") is not True or not payload.get("documents"):
        raise ValueError("Fixture corpus must explicitly identify synthetic evidence")
    with KnowledgeStore(":memory:") as store:
        for item in payload["documents"]:
            store.upsert_document_sources([DocumentSource(
                item["document_id"], item["title"], item.get("source_url"), None,
                item["document_id"] + ".pdf", "pdf",
            )])
            artifact = KnowledgeArtifact(**{key: value for key, value in item.items()
                                           if key in KnowledgeArtifact.__dataclass_fields__})
            store.ingest_chunk(artifact, item["embedding"])
        yield store
