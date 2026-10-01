"""Run reproducible BM25, dense, and RRF candidate retrieval baselines.

The script reads the frozen page corpus and benchmark questions once, computes
full-corpus rankings for BM25 and dense retrieval, then writes the top 20 results
for each system. RRF uses 1-based ranks over the complete rankings.

Example:
    python scripts/run_conventional_retrievers.py
    python scripts/run_conventional_retrievers.py \
        --corpus benchmark/corpus_pages.jsonl \
        --questions benchmark/questions.csv \
        --output-dir benchmark/candidate_runs
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Protocol

import numpy as np


LOGGER = logging.getLogger("run_conventional_retrievers")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPOSITORY_ROOT / "benchmark" / "corpus_pages.jsonl"
DEFAULT_QUESTIONS = REPOSITORY_ROOT / "benchmark" / "questions.csv"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "benchmark" / "candidate_runs"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Pin the model artifact so a future update to the Hugging Face repository does
# not silently change benchmark scores.
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TOP_K = 20
RRF_CONSTANT = 60
SEED = 202609
TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


class RetrievalInputError(ValueError):
    """Raised when a benchmark input is missing, malformed, or ambiguous."""


@dataclass(frozen=True)
class Page:
    page_id: str
    page_text: str


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class Hit:
    page_id: str
    score: float


class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        """Encode text as a two-dimensional NumPy array."""


def set_reproducible_seed(seed: int = SEED) -> None:
    """Seed libraries used by the dense encoder and request deterministic ops."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_pages(path: Path) -> list[Page]:
    """Load and validate the page corpus without retaining unused metadata."""

    if not path.is_file():
        raise RetrievalInputError(f"Corpus JSONL does not exist: {path}")

    pages: list[Page] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise RetrievalInputError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
            page_id = item.get("page_id")
            page_text = item.get("page_text")
            if not isinstance(page_id, str) or not page_id.strip():
                raise RetrievalInputError(
                    f"Missing string page_id at {path}:{line_number}"
                )
            if not isinstance(page_text, str):
                raise RetrievalInputError(
                    f"Missing string page_text at {path}:{line_number}"
                )
            if page_id in seen:
                raise RetrievalInputError(f"Duplicate page_id in corpus: {page_id}")
            seen.add(page_id)
            pages.append(Page(page_id=page_id, page_text=page_text))

    if not pages:
        raise RetrievalInputError(f"Corpus contains no page records: {path}")
    return pages


def load_queries(path: Path) -> list[Query]:
    """Load query_id and query columns from a UTF-8 CSV file."""

    if not path.is_file():
        raise RetrievalInputError(
            f"Questions CSV does not exist: {path}. "
            "Create it with columns query_id,query."
        )

    queries: list[Query] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = set(reader.fieldnames or ())
        if "query_id" not in fieldnames or "query" not in fieldnames:
            raise RetrievalInputError(
                f"{path} must contain CSV columns query_id and query"
            )
        for line_number, row in enumerate(reader, start=2):
            query_id = (row.get("query_id") or "").strip()
            query_text = row.get("query") or ""
            if not query_id or not query_text.strip():
                raise RetrievalInputError(
                    f"Blank query_id or query at {path}:{line_number}"
                )
            if query_id in seen:
                raise RetrievalInputError(f"Duplicate query_id: {query_id}")
            seen.add(query_id)
            queries.append(Query(query_id=query_id, text=query_text))

    if not queries:
        raise RetrievalInputError(f"Questions CSV contains no queries: {path}")
    return queries


def tokenize(text: str) -> list[str]:
    """Apply one deterministic tokenizer to both documents and queries."""

    return TOKEN_PATTERN.findall(text.casefold())


def rank_scores(page_ids: list[str], scores: np.ndarray) -> list[Hit]:
    """Sort descending by score and then ascending by page_id for stable ties."""

    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(page_ids),):
        raise ValueError(
            f"Expected {len(page_ids)} scores, received shape {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Retriever produced a non-finite score")
    order = sorted(
        range(len(page_ids)),
        key=lambda index: (-float(values[index]), page_ids[index]),
    )
    return [Hit(page_id=page_ids[index], score=float(values[index])) for index in order]


def run_bm25(pages: list[Page], queries: list[Query]) -> dict[str, list[Hit]]:
    """Build one BM25 index and return a complete ranking for every query."""

    try:
        from rank_bm25 import BM25Okapi
    except ImportError as error:
        raise RuntimeError(
            "rank-bm25 is required; install project dependencies first"
        ) from error

    page_ids = [page.page_id for page in pages]
    tokenized_pages = [tokenize(page.page_text) for page in pages]
    bm25 = BM25Okapi(tokenized_pages)
    return {
        query.query_id: rank_scores(
            page_ids,
            np.asarray(bm25.get_scores(tokenize(query.text)), dtype=np.float64),
        )
        for query in queries
    }


def load_dense_encoder(
    model_name: str,
    model_revision: str,
    device: str,
) -> Encoder:
    """Load the pinned SentenceTransformer model only when dense retrieval runs."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is required; install project dependencies first"
        ) from error
    return SentenceTransformer(
        model_name,
        revision=model_revision,
        device=device,
    )


def run_dense(
    pages: list[Page],
    queries: list[Query],
    *,
    model: Encoder,
    batch_size: int,
) -> dict[str, list[Hit]]:
    """Encode pages once and rank them by cosine similarity for every query."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    page_ids = [page.page_id for page in pages]
    page_embeddings = np.asarray(
        model.encode(
            [page.page_text for page in pages],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    query_embeddings = np.asarray(
        model.encode(
            [query.text for query in queries],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if page_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("Dense encoder must return two-dimensional arrays")
    if page_embeddings.shape[0] != len(pages):
        raise ValueError("Dense encoder returned the wrong number of page vectors")
    if query_embeddings.shape[0] != len(queries):
        raise ValueError("Dense encoder returned the wrong number of query vectors")
    if page_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("Page and query embedding dimensions differ")

    similarity = query_embeddings @ page_embeddings.T
    return {
        query.query_id: rank_scores(page_ids, similarity[query_index])
        for query_index, query in enumerate(queries)
    }


def reciprocal_rank_fusion(
    bm25_rankings: dict[str, list[Hit]],
    dense_rankings: dict[str, list[Hit]],
    *,
    constant: int = RRF_CONSTANT,
) -> dict[str, list[Hit]]:
    """Fuse complete BM25 and dense rankings using 1-based reciprocal ranks."""

    if constant < 0:
        raise ValueError("RRF constant must be non-negative")
    if bm25_rankings.keys() != dense_rankings.keys():
        raise ValueError("BM25 and dense query IDs do not match")

    fused: dict[str, list[Hit]] = {}
    for query_id in bm25_rankings:
        bm25_hits = bm25_rankings[query_id]
        dense_hits = dense_rankings[query_id]
        bm25_ids = {hit.page_id for hit in bm25_hits}
        dense_ids = {hit.page_id for hit in dense_hits}
        if bm25_ids != dense_ids:
            raise ValueError(f"Candidate sets differ for query {query_id}")

        scores = {page_id: 0.0 for page_id in bm25_ids}
        for rank, hit in enumerate(bm25_hits, start=1):
            scores[hit.page_id] += 1.0 / (constant + rank)
        for rank, hit in enumerate(dense_hits, start=1):
            scores[hit.page_id] += 1.0 / (constant + rank)

        fused[query_id] = [
            Hit(page_id=page_id, score=score)
            for page_id, score in sorted(
                scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
    return fused


def write_run(
    path: Path,
    queries: list[Query],
    rankings: dict[str, list[Hit]],
    *,
    system: str,
    top_k: int = TOP_K,
) -> int:
    """Atomically write one JSONL run in stable query and rank order."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    records_written = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            for query in queries:
                try:
                    hits = rankings[query.query_id][:top_k]
                except KeyError as error:
                    raise ValueError(
                        f"Missing ranking for query {query.query_id}"
                    ) from error
                for rank, hit in enumerate(hits, start=1):
                    record = {
                        "query_id": query.query_id,
                        "page_id": hit.page_id,
                        "rank": rank,
                        "score": hit.score,
                        "system": system,
                    }
                    destination.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    )
                    destination.write("\n")
                    records_written += 1
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return records_written


def validate_existing_run(
    path: Path,
    *,
    system: str,
    queries: list[Query],
    valid_page_ids: set[str],
    top_k: int = TOP_K,
) -> bool:
    """Return True only for a complete, schema-valid conventional run."""

    if not path.is_file():
        return False
    expected_per_query = min(top_k, len(valid_page_ids))
    grouped: dict[str, list[tuple[int, str]]] = {}
    try:
        with path.open("r", encoding="utf-8-sig") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                query_id = record.get("query_id")
                page_id = record.get("page_id")
                rank = record.get("rank")
                score = record.get("score")
                if (
                    record.get("system") != system
                    or not isinstance(query_id, str)
                    or not isinstance(page_id, str)
                    or type(rank) is not int
                    or not isinstance(score, (int, float))
                    or not np.isfinite(float(score))
                    or page_id not in valid_page_ids
                ):
                    return False
                grouped.setdefault(query_id, []).append((rank, page_id))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False

    expected_queries = {query.query_id for query in queries}
    if set(grouped) != expected_queries:
        return False
    for rows in grouped.values():
        rows.sort()
        if len(rows) != expected_per_query:
            return False
        if [rank for rank, _ in rows] != list(range(1, expected_per_query + 1)):
            return False
        page_ids = [page_id for _, page_id in rows]
        if len(page_ids) != len(set(page_ids)):
            return False
    return True


def conventional_outputs_complete(
    output_dir: Path,
    *,
    pages: list[Page],
    queries: list[Query],
    top_k: int = TOP_K,
) -> bool:
    """Validate all three run files before skipping native dense encoding."""

    valid_page_ids = {page.page_id for page in pages}
    return all(
        validate_existing_run(
            output_dir / f"{system}.jsonl",
            system=system,
            queries=queries,
            valid_page_ids=valid_page_ids,
            top_k=top_k,
        )
        for system in ("bm25", "dense", "hybrid")
    )

def run_all(
    *,
    corpus_path: Path,
    questions_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    model_revision: str = DEFAULT_MODEL_REVISION,
    device: str = "cpu",
    batch_size: int = 64,
    seed: int = SEED,
    top_k: int = TOP_K,
) -> dict[str, int]:
    """Execute all three baseline systems and write their top-k runs."""

    set_reproducible_seed(seed)
    pages = load_pages(corpus_path)
    queries = load_queries(questions_path)
    LOGGER.info("Loaded %d pages and %d queries", len(pages), len(queries))

    bm25_rankings = run_bm25(pages, queries)
    encoder = load_dense_encoder(model_name, model_revision, device)
    dense_rankings = run_dense(
        pages,
        queries,
        model=encoder,
        batch_size=batch_size,
    )
    hybrid_rankings = reciprocal_rank_fusion(bm25_rankings, dense_rankings)

    output_dir = output_dir.resolve()
    counts = {
        "bm25": write_run(
            output_dir / "bm25.jsonl", queries, bm25_rankings, system="bm25", top_k=top_k
        ),
        "dense": write_run(
            output_dir / "dense.jsonl", queries, dense_rankings, system="dense", top_k=top_k
        ),
        "hybrid": write_run(
            output_dir / "hybrid.jsonl", queries, hybrid_rankings, system="hybrid", top_k=top_k
        ),
    }
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BM25, dense, and RRF retrieval over the frozen page corpus."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--device",
        default="cpu",
        help="SentenceTransformer device. CPU is the reproducible default.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip recomputation when all conventional run files validate as complete.",
    )
    parser.add_argument(
        "--log-level",        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
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
        if args.resume:
            pages = load_pages(args.corpus)
            queries = load_queries(args.questions)
            if conventional_outputs_complete(
                args.output_dir.resolve(),
                pages=pages,
                queries=queries,
                top_k=args.top_k,
            ):
                LOGGER.info(
                    "Conventional candidate runs are complete and valid; "
                    "skipping BM25 and dense recomputation."
                )
                return 0
        counts = run_all(
            corpus_path=args.corpus,
            questions_path=args.questions,
            output_dir=args.output_dir,
            model_name=args.model,
            model_revision=args.model_revision,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            top_k=args.top_k,
        )
    except (OSError, RuntimeError, RetrievalInputError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1

    for system, count in counts.items():
        LOGGER.info("Wrote %d %s records", count, system)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
