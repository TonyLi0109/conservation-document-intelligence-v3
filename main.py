"""Executable orchestration layer for Conservation Intelligence V3.

PDF extraction uses the production page-aware parser, and embedding plus
structured synthesis use production API clients. Storage, provenance mapping,
validation, and citation rendering remain application-owned V3 boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
import sys

from api_clients import call_llm, generate_embedding, generate_embeddings
from config import SETTINGS
from data_models import KnowledgeArtifact
from database import KnowledgeStore
from pdf_parser import (
    DocumentParseReport,
    parse_pdf_to_artifacts,
    parse_text_to_artifacts,
    read_pdf_page_labels,
)
from prompts import SYSTEM_PROMPT, answer_length_constraints, build_synthesis_prompt
from source_catalog import load_source_catalog
from validator import validate_render_and_collect_sources
from validator import VALIDATION_FAILED_MESSAGE, format_artifact_location


DEFAULT_TOP_K = SETTINGS.retrieval.top_k
INGESTION_BATCH_SIZE = SETTINGS.models.embedding_batch_size
DOCUMENT_FILENAME_PATTERN = re.compile(
    r"^(?P<document_id>DOC\d{3})[_\-\s]+(?P<title>.+)$",
    re.IGNORECASE,
)
DOCUMENT_DISCOVERY_PATTERN = re.compile(
    r"\b(which|what|list|identify|find)\b.*\b(document|documents|sources|reports|plans)\b",
    re.IGNORECASE,
)
SIMPLE_EXISTENCE_PATTERN = re.compile(
    r"^\s*(?:are|is|do|does|did|has|have)\b", re.IGNORECASE
)
EXISTENCE_STOPWORDS = {
    "a", "an", "any", "are", "did", "do", "does", "has", "have", "in",
    "is", "of", "the", "there", "was", "were",
}


def _sha256_file(path: Path) -> str:
    """Hash a source in bounded blocks for versioned provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ingest_corpus(pdf_directory: str, store: KnowledgeStore) -> int:
    """Parse all PDFs, batch-embed them, then atomically persist the corpus."""

    directory = Path(pdf_directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"PDF directory does not exist: {directory}")
    if not isinstance(store, KnowledgeStore):
        raise TypeError("store must be a KnowledgeStore")

    catalog = load_source_catalog()
    pending: list[KnowledgeArtifact] = []
    ingestion_reports: list[tuple[str, str, DocumentParseReport]] = []
    staged_count = 0

    def flush_batch() -> None:
        nonlocal staged_count
        if not pending:
            return
        embeddings = generate_embeddings(
            [artifact.original_text_chunk for artifact in pending],
            batch_size=INGESTION_BATCH_SIZE,
        )
        staged_count += store.stage_batch(pending, embeddings)
        logging.info("Persisted %s staged chunks", staged_count)
        pending.clear()

    source_files: dict[str, tuple[Path, str]] = {}
    for source_path in sorted(directory.iterdir(), key=lambda path: path.name.casefold()):
        if not source_path.is_file() or source_path.suffix.casefold() not in {".pdf", ".txt"}:
            continue
        match = DOCUMENT_FILENAME_PATTERN.fullmatch(source_path.stem)
        if not match:
            logging.warning(
                "Skipping non-canonical source without a DOCxxx filename prefix: %s",
                source_path.name,
            )
            continue
        document_id = match.group("document_id").upper()
        if document_id in source_files:
            previous = source_files[document_id][0]
            raise ValueError(
                f"Duplicate source files for {document_id}: {previous.name}, {source_path.name}"
            )
        raw_title = match.group("title")
        title = catalog.get(document_id).title if document_id in catalog else raw_title.replace("_", " ").replace("-", " ").strip()
        source_files[document_id] = (source_path, title)

    if not source_files:
        raise RuntimeError("No canonical DOCxxx PDF or TXT sources were found")
    catalog_ids = set(catalog)
    unknown = sorted(set(source_files) - catalog_ids)
    missing = sorted(catalog_ids - set(source_files))
    if unknown or missing:
        details = []
        if unknown:
            details.append("not in catalog: " + ", ".join(unknown))
        if missing:
            details.append("missing source files: " + ", ".join(missing))
        raise RuntimeError("Corpus/catalog mismatch; " + "; ".join(details))

    store.begin_rebuild()
    try:
        completed_documents: set[str] = set()
        for document_id in sorted(source_files):
            source_path, title = source_files[document_id]
            logging.info("Parsing %s as %s", source_path.name, document_id)
            parser = (
                parse_pdf_to_artifacts
                if source_path.suffix.casefold() == ".pdf"
                else parse_text_to_artifacts
            )
            document_chunk_count = 0
            parse_report = DocumentParseReport()
            parser_arguments = {
                "document_id": document_id,
                "title": title,
            }
            if source_path.suffix.casefold() == ".pdf":
                parser_arguments["report"] = parse_report
            for artifact in parser(str(source_path), **parser_arguments):
                pending.append(artifact)
                document_chunk_count += 1
                if len(pending) >= INGESTION_BATCH_SIZE:
                    flush_batch()
            if document_chunk_count == 0:
                raise RuntimeError(
                    f"{document_id} produced no readable chunks: {source_path.name}; "
                    "existing index retained"
                )
            completed_documents.add(document_id)
            if source_path.suffix.casefold() == ".txt":
                parse_report.total_pages = 1
                parse_report.extracted_pages = 1
                parse_report.chunk_count = document_chunk_count
            ingestion_reports.append(
                (document_id, _sha256_file(source_path), parse_report)
            )
            logging.info(
                "Completed %s with %s chunk(s)", document_id, document_chunk_count
            )
        flush_batch()
        if staged_count == 0:
            raise RuntimeError(
                "No readable PDF chunks were produced; existing index retained"
            )
        if completed_documents != set(source_files):
            missing = sorted(set(source_files) - completed_documents)
            raise RuntimeError(f"Ingestion did not complete every source: {', '.join(missing)}")
        committed_count = store.commit_rebuild()
        store.upsert_document_sources(list(catalog.values()))
        for document_id, file_sha256, report in ingestion_reports:
            store.record_ingestion_report(
                document_id,
                file_sha256,
                total_pages=report.total_pages,
                extracted_pages=report.extracted_pages,
                empty_pages=report.empty_pages,
                failed_pages=report.failed_pages,
                fallback_pages=report.fallback_pages,
                chunk_count=report.chunk_count,
            )
        logging.info(
            "Committed %s artifacts across %s canonical document(s)",
            committed_count,
            len(completed_documents),
        )
        return committed_count
    except Exception:
        store.abort_rebuild()
        raise


def backfill_printed_page_labels(source_directory: str, store: KnowledgeStore) -> int:
    """Add standard logical page labels to an existing index without API calls."""

    directory = Path(source_directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {directory}")
    updated = 0
    for path in sorted(directory.glob("DOC*.pdf")):
        match = DOCUMENT_FILENAME_PATTERN.fullmatch(path.stem)
        if match is None:
            continue
        document_id = match.group("document_id").upper()
        try:
            labels = read_pdf_page_labels(path)
            changed = store.set_document_page_labels(document_id, labels)
            updated += changed
            logging.info(
                "%s: applied %s logical labels to %s indexed chunks",
                document_id, len(labels), changed,
            )
        except Exception as error:
            logging.warning("Could not backfill page labels for %s: %s", path, error)
    return updated


def ask_chatbot(
    question: str,
    store: KnowledgeStore,
    *,
    top_k: int = DEFAULT_TOP_K,
    model: str | None = None,
) -> str:
    """Retrieve evidence, synthesize via API, validate, and render Markdown."""

    answer, _, _ = ask_chatbot_with_context(question, store, top_k=top_k, model=model)
    return answer


def ask_chatbot_with_sources(
    question: str,
    store: KnowledgeStore,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[str, list[KnowledgeArtifact]]:
    """Return validated Markdown and only the artifacts cited by valid claims."""

    answer, _, sources = ask_chatbot_with_context(question, store, top_k=top_k)
    return answer, sources


def ask_chatbot_with_context(
    question: str,
    store: KnowledgeStore,
    *,
    top_k: int = DEFAULT_TOP_K,
    model: str | None = None,
) -> tuple[str, str, list[KnowledgeArtifact]]:
    """Return validated Markdown, an ungrounded scope preamble, and cited sources.

    The preamble is removed from the model envelope before the unchanged
    ``SynthesisResponse`` payload enters provenance validation. It may describe
    corpus coverage only; all conservation facts remain validator-owned claims.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(store, KnowledgeStore):
        raise TypeError("store must be a KnowledgeStore")

    if DOCUMENT_DISCOVERY_PATTERN.search(question):
        artifacts = store.retrieve_document_matches(
            question, top_k=max(10, top_k)
        )
        if artifacts:
            lines = [
                f"- [{artifact.document_id} — {artifact.title}, "
                f"{format_artifact_location(artifact)}]"
                for artifact in artifacts
            ]
            return (
                "\n".join(lines),
                "A corpus-wide document scan found the following sources with explicit matching evidence:",
                artifacts,
            )

    try:
        query_embedding = generate_embedding(question)
        candidate_count = max(top_k, top_k * 4)
        candidates = store.retrieve(query_embedding, candidate_count)
    except Exception as error:
        logging.exception("Semantic retrieval failed; attempting keyword fallback")
        candidates = store.retrieve(
            None, max(top_k, top_k * 4), method="keyword", query_text=question
        )
    # Document-oriented questions benefit from source diversity rather than five
    # neighboring chunks from the same report. The order remains retrieval-owned.
    artifacts: list[KnowledgeArtifact] = []
    seen_documents: set[str] = set()
    for artifact in candidates:
        if artifact.document_id in seen_documents:
            continue
        seen_documents.add(artifact.document_id)
        artifacts.append(artifact)
        if len(artifacts) >= top_k:
            break
    artifact_handles = {
        f"K{index}": artifact for index, artifact in enumerate(artifacts, start=1)
    }
    if not artifact_handles:
        answer, sources = validate_render_and_collect_sources(
            json.dumps(
                {
                    "status": "insufficient_evidence",
                    "claims": [],
                    "unsupported_facets": [
                        "No relevant evidence was retrieved for the question."
                    ],
                }
            ),
            artifact_handles,
        )
        return answer, "No relevant evidence was retrieved from the corpus.", sources
    direct_answer = _direct_existence_answer(question, store)
    if direct_answer is not None:
        return direct_answer

    user_prompt = build_synthesis_prompt(question, artifact_handles)
    max_claims, max_output_tokens, _ = answer_length_constraints(question)
    try:
        llm_response = call_llm(
            SYSTEM_PROMPT,
            user_prompt,
            artifact_handles,
            model=model,
            max_claims=max_claims,
            max_output_tokens=max_output_tokens,
        )
        envelope = json.loads(llm_response)
        expected_fields = {"preamble", "status", "claims", "unsupported_facets"}
        if not isinstance(envelope, dict) or set(envelope) != expected_fields:
            raise ValueError("Chatbot response does not match the synthesis envelope")
        preamble = envelope.pop("preamble")
        if not isinstance(preamble, str):
            raise TypeError("Chatbot preamble must be a string")
        preamble = preamble.strip()
        if len(preamble) > 1_000:
            raise ValueError("Chatbot preamble exceeds 1,000 characters")
        answer, sources = validate_render_and_collect_sources(
            json.dumps(envelope, ensure_ascii=False), artifact_handles
        )
        if answer == VALIDATION_FAILED_MESSAGE:
            raise ValueError("all generated claims failed provenance validation")
        return answer, preamble, sources
    except Exception:
        logging.exception(
            "Grounded synthesis failed; returning canonical retrieval fallback"
        )
        lines = [
            f"- [{artifact.document_id} — {artifact.title}, "
            + format_artifact_location(artifact)
            + "]"
            for artifact in artifacts
        ]
        return (
            "\n".join(lines),
            "A fully synthesized answer could not be validated, so the system is showing the most relevant canonical sources instead:",
            artifacts,
        )


def _normalized_existence_terms(question: str) -> list[str]:
    """Extract conservative content terms from a simple existence question."""

    terms: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z'-]*", question.casefold()):
        if raw in EXISTENCE_STOPWORDS:
            continue
        term = raw[:-1] if raw.endswith("s") and len(raw) > 4 else raw
        if term not in terms:
            terms.append(term)
    return terms


def _direct_existence_answer(
    question: str, store: KnowledgeStore
) -> tuple[str, str, list[KnowledgeArtifact]] | None:
    """Answer only when one affirmative canonical sentence covers every term."""

    if not SIMPLE_EXISTENCE_PATTERN.search(question):
        return None
    terms = _normalized_existence_terms(question)
    if len(terms) < 2:
        return None
    query = " ".join(terms)
    candidates = store.retrieve(
        None, max(DEFAULT_TOP_K * 4, 20), method="keyword", query_text=query
    )
    negative = re.compile(r"\b(?:no|not|none|without|unknown|uncertain|may|might)\b")
    for artifact in candidates:
        for raw_sentence in re.split(r"(?<=[.!?])\s+", artifact.original_text_chunk):
            sentence = raw_sentence.strip()
            normalized = sentence.casefold()
            if not sentence or negative.search(normalized):
                continue
            if all(term in normalized for term in terms):
                handles = {"K1": artifact}
                payload = {
                    "status": "answered",
                    "claims": [{
                        "text": (
                            "Yes. The corpus explicitly documents "
                            + " ".join(terms[:-1])
                            + f" in {terms[-1].title()}."
                        ),
                        "evidence_ids": ["K1"],
                        "supporting_spans": [sentence],
                    }],
                    "unsupported_facets": [],
                }
                answer, sources = validate_render_and_collect_sources(
                    json.dumps(payload), handles
                )
                if answer != VALIDATION_FAILED_MESSAGE:
                    return answer, "", sources
    return None


def search_corpus(
    query: str,
    store: KnowledgeStore,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[KnowledgeArtifact]:
    """Embed a query and return canonical semantic matches for UI/API callers."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(store, KnowledgeStore):
        raise TypeError("store must be a KnowledgeStore")
    query_embedding = generate_embedding(query)
    return store.retrieve(query_embedding, top_k)


def _parse_args() -> argparse.Namespace:
    """Parse mutually exclusive persistent ingestion and query commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ingest", metavar="PDF_DIRECTORY", help="Build the persistent corpus index")
    mode.add_argument("--query", metavar="QUESTION", help="Query the existing persistent index")
    mode.add_argument(
        "--backfill-page-labels", metavar="SOURCE_DIRECTORY",
        help="Add PDF logical page labels to the existing index without embedding",
    )
    mode.add_argument(
        "--precompile-wiki", action="store_true",
        help="Create fast deterministic Wiki pages for every curated entity",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--debug", action="store_true", help="Enable provenance validation diagnostics")
    return parser.parse_args()


def _run_cli() -> None:
    """Run production parsing, retrieval, generation, and validation."""

    # Keep Unicode citation punctuation intact when PowerShell output is piped
    # through Tee-Object or captured by another process.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    with KnowledgeStore() as store:
        store.upsert_document_sources(list(load_source_catalog().values()))
        if args.ingest:
            chunk_count = ingest_corpus(args.ingest, store)
            print(f"Persisted {chunk_count} canonical artifact(s).")
            return
        if args.backfill_page_labels:
            updated = backfill_printed_page_labels(args.backfill_page_labels, store)
            print(f"Updated logical page labels on {updated} canonical artifact(s).")
            return
        if args.precompile_wiki:
            from wiki_compiler import precompile_all_wiki_concepts

            generated = precompile_all_wiki_concepts(store)
            print(f"Pre-generated {generated} Wiki page(s).")
            return
        if store.artifact_count == 0:
            raise RuntimeError("The persistent corpus is empty. Run --ingest first.")
        answer = ask_chatbot(args.query, store, top_k=args.top_k)
        print(answer)


if __name__ == "__main__":
    _run_cli()
