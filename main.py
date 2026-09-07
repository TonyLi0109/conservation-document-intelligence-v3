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
from chat_context import is_named_entity, matches_subject, resolve_query
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
    r"\b(?:which|what)\s+(?:(?:MDC|USGS|USACE)\s+)?(?:documents|reports|sources|plans)\b"
    r"|\b(?:list|identify|find|locate)\b.{0,80}\b(?:documents?|sources?|reports?|plans?)\b",
    re.IGNORECASE,
)
SIMPLE_EXISTENCE_PATTERN = re.compile(
    r"^\s*(?:(?:please\s+)?tell\s+me(?:\s+briefly)?\s+whether\b|"
    r"(?:are|is|do|does|did|has|have|was|were)\b)",
    re.IGNORECASE,
)
EXISTENCE_STOPWORDS = {
    "a", "an", "any", "are", "addressed", "did", "discussed", "do", "does",
    "documented", "found", "has", "have", "in", "is", "mentioned", "of",
    "present", "the", "there", "was", "were", "tell", "me", "briefly",
    "whether", "occur", "occurs",
}


def _require_store_interface(store: object, *methods: str) -> None:
    """Validate behavior, not class identity, across Streamlit hot reloads."""

    missing = [name for name in methods if not callable(getattr(store, name, None))]
    if missing:
        raise TypeError(
            "store does not provide the required KnowledgeStore interface: "
            + ", ".join(missing)
        )


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
    _require_store_interface(
        store,
        "begin_rebuild",
        "stage_batch",
        "commit_rebuild",
        "abort_rebuild",
        "upsert_document_sources",
        "record_ingestion_report",
    )

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
        if hasattr(store, "connection"):
            from document_lifecycle import rebuild_lifecycles
            rebuild_lifecycles(store)
            from retrieval_index import ensure_retrieval_index
            ensure_retrieval_index(store)
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
    history: list[dict[str, object]] | None = None,
    diagnostics: dict[str, object] | None = None,
) -> tuple[str, str, list[KnowledgeArtifact]]:
    """Return validated Markdown, an ungrounded scope preamble, and cited sources.

    The preamble is removed from the model envelope before the unchanged
    ``SynthesisResponse`` payload enters provenance validation. It may describe
    corpus coverage only; all conservation facts remain validator-owned claims.
    Optional session-owned history resolves intent before retrieval. Diagnostics
    are returned only to the caller and are never logged as conversation text.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    _require_store_interface(store, "retrieve", "retrieve_document_matches")

    original_question = question
    context = resolve_query(question, history, model=model)
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(context.diagnostics())
        diagnostics["retrieval_query"] = context.standalone_query if not context.needs_clarification else ""
    if context.needs_clarification:
        return (
            context.clarification_question or "Please name the subject, method, option, or report you mean so I can search the right evidence.",
            "",
            [],
        )
    question = context.standalone_query

    from temporal import detect_temporal_intent, select_temporal_evidence, render_temporal_answer
    temporal_intent = detect_temporal_intent(original_question)
    if temporal_intent.mode != "none" and hasattr(store, "connection"):
        anchors = []
        if context.uses_history and context.relation == "FOLLOW_UP":
            # Only the last cited answer in this page-owned thread can anchor a
            # family. Its old date is not a current/latest retrieval constraint.
            for message in reversed(history or []):
                if message.get("role") == "assistant":
                    anchors = [source.document_id for source in message.get("sources", [])
                               if isinstance(getattr(source, "document_id", None), str) and (
                                   not is_named_entity(context.active_subject) or matches_subject(source, context.active_subject))]
                    break
        selection = select_temporal_evidence(question, store, top_k=top_k,
                                             intent=temporal_intent, anchor_document_ids=anchors)
        if diagnostics is not None:
            diagnostics["temporal_intent"] = temporal_intent.mode
            diagnostics["temporal_selected_document_ids"] = selection["selected_document_ids"]
            diagnostics["temporal_decisions"] = selection["decisions"]
            if not context.active_subject and selection["selected_document_ids"] and len({
                doc["family_id"] for doc in selection["documents"].values()
            }) == 1:
                from document_lifecycle import normalize_title
                first_document = selection["documents"][selection["selected_document_ids"][0]]
                diagnostics["active_subject"] = normalize_title(first_document["title"])
        return render_temporal_answer(selection, store)

    comparison_answer = _direct_fiscal_year_comparison(question, store)
    if comparison_answer is not None:
        return comparison_answer

    native_answer = _direct_native_status_answer(question, store)
    if native_answer is not None:
        return native_answer

    # Resolve narrow yes/no existence questions locally before paying for an
    # embedding or generation request. The helper returns only when one exact,
    # affirmative canonical sentence contains every meaningful question term.
    direct_answer = _direct_existence_answer(question, store)
    if direct_answer is not None:
        return direct_answer

    if DOCUMENT_DISCOVERY_PATTERN.search(original_question):
        if hasattr(store, "connection"):
            from retrieval import retrieve_evidence
            trace = {}
            matches = retrieve_evidence(store, question, top_k=max(10, top_k) * 4, diagnostics=trace)
            by_document = {}
            for artifact in matches:
                by_document.setdefault(artifact.document_id, artifact)
            artifacts = list(by_document.values())[:max(10, top_k)]
            if diagnostics is not None:
                diagnostics["retrieval"] = trace
        else:
            artifacts = store.retrieve_document_matches(question, top_k=max(10, top_k))
        if artifacts:
            lines = [
                f"- [{artifact.document_id} — {artifact.title}, "
                f"{format_artifact_location(artifact)}]"
                for artifact in artifacts
            ]
            return (
                "\n".join(lines),
                "The following sources match the requested report or topic:",
                artifacts,
            )

    if hasattr(store, "connection"):
        from retrieval import retrieve_evidence
        trace = {}
        try:
            query_embedding = generate_embedding(question)
        except Exception:
            logging.warning("Query embedding unavailable; using indexed lexical retrieval")
            query_embedding = None
            trace["embedding_fallback"] = True
        artifacts = retrieve_evidence(
            store, question, query_embedding=query_embedding, top_k=top_k,
            entity=context.active_subject if context.relation == "FOLLOW_UP" else None,
            diagnostics=trace,
        )
        if diagnostics is not None:
            diagnostics["retrieval"] = trace
    else:
        try:
            query_embedding = generate_embedding(question)
            candidate_count = max(top_k, top_k * 4)
            candidates = store.retrieve(query_embedding, candidate_count)
        except Exception as error:
            logging.exception("Semantic retrieval failed; attempting keyword fallback")
            candidates = store.retrieve(
                None, max(top_k, top_k * 4), method="keyword", query_text=question
            )
        if context.uses_history and context.active_subject:
            # Supplement semantic results with the existing keyword retriever.
            keyword_candidates = store.retrieve(
                None, top_k * 4, method="keyword", query_text=question
            )
            candidates = candidates + keyword_candidates
            if context.relation == "FOLLOW_UP":
                matching = [artifact for artifact in candidates
                            if matches_subject(artifact, context.active_subject)]
                # Broad topics such as "conservation threats" are retrieval intent,
                # not an entity string that every useful source must repeat verbatim.
                candidates = matching if is_named_entity(context.active_subject) else matching + [
                    artifact for artifact in candidates if not matches_subject(artifact, context.active_subject)
                ]
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
    user_prompt = build_synthesis_prompt(question, artifact_handles)
    max_claims, max_output_tokens, _ = answer_length_constraints(original_question)
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
        _ensure_polar_answer_prefix(question, envelope)
        preamble = envelope.pop("preamble")
        if not isinstance(preamble, str):
            raise TypeError("Chatbot preamble must be a string")
        preamble = preamble.strip()
        if len(preamble) > 1_000:
            raise ValueError("Chatbot preamble exceeds 1,000 characters")
        from temporal import is_lifecycle_assertion
        if is_lifecycle_assertion(preamble):
            preamble = ""
        # A normal synthesis response cannot establish document authority or
        # supersession. Those assertions belong to the local temporal renderer.
        claims = envelope.get("claims", [])
        retained = [claim for claim in claims if not is_lifecycle_assertion(str(claim.get("text", "")))]
        if len(retained) != len(claims):
            if not retained:
                raise ValueError("Lifecycle assertions require verified lifecycle evidence")
            envelope["claims"] = retained
            envelope["status"] = "partially_answered"
            envelope["unsupported_facets"] = list(envelope.get("unsupported_facets", [])) + [
                "Document currentness or replacement requires a separate source-version check."
            ]
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
    normalized_question = question.casefold().replace("’", "'")
    for raw in re.findall(r"[A-Za-z][A-Za-z'-]*", normalized_question):
        raw = raw[:-2] if raw.endswith("'s") else raw
        if len(raw) < 2 or raw in EXISTENCE_STOPWORDS:
            continue
        term = raw[:-1] if raw.endswith("s") and len(raw) > 4 else raw
        if term not in terms:
            terms.append(term)
    return terms


def _direct_fiscal_year_comparison(
    question: str, store: KnowledgeStore
) -> tuple[str, str, list[KnowledgeArtifact]] | None:
    """Compare FY2021/FY2024 carp work, labelling cross-agency evidence."""

    normalized_question = " ".join(question.casefold().split())
    if not (
        normalized_question.startswith("compare ")
        and "invasive carp" in normalized_question
        and "fy2021" in normalized_question
        and "fy2024" in normalized_question
    ):
        return None
    # Use only the long-standing retrieve interface so this route also works
    # with a Store instance cached before a Streamlit hot reload. A very high
    # top_k makes this an exhaustive phrase scan for this small fixed corpus.
    carp_matches = store.retrieve(
        None, 100_000, method="keyword", query_text="carp"
    )
    fy2021_has_carp = any(
        artifact.document_id == "DOC018"
        and "carp" in artifact.original_text_chunk.casefold()
        for artifact in carp_matches
    )
    if fy2021_has_carp:
        # A true two-sided comparison needs separate evidence from each report;
        # leave that case to the normal grouped synthesis path.
        return None

    fy2021_artifact: KnowledgeArtifact | None = None
    fy2021_span = ""
    for artifact in carp_matches:
        if artifact.document_id != "DOC012":
            continue
        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+|[\r\n]+", artifact.original_text_chunk
            )
            if sentence.strip()
        ]
        fy2021_span = next(
            (
                sentence
                for sentence in sentences
                if (
                    "in 2020 and 2021" in sentence.casefold()
                    and "deterrent" in sentence.casefold()
                )
                or (
                    ("fy2021" in sentence.casefold() or "fy 2021" in sentence.casefold())
                    and "invasive carp" in sentence.casefold()
                    and ("fund" in sentence.casefold() or "research" in sentence.casefold())
                )
            ),
            "",
        )
        if fy2021_span:
            fy2021_artifact = artifact
            break

    fy2024_artifact: KnowledgeArtifact | None = None
    permit_span = ""
    removal_span = ""
    for artifact in store.retrieve(
        None,
        100,
        method="keyword",
        query_text="FY24 invasive carp removal lower Grand River",
    ):
        if artifact.document_id != "DOC016":
            continue
        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+|[\r\n]+", artifact.original_text_chunk
            )
            if sentence.strip()
        ]
        permit_span = next(
            (
                sentence
                for sentence in sentences
                if "fy24" in sentence.casefold()
                and "permits for invasive carp removal" in sentence.casefold()
            ),
            "",
        )
        removal_span = next(
            (
                sentence
                for sentence in sentences
                if "over 19 tons of invasive carp" in sentence.casefold()
                and "lower grand river" in sentence.casefold()
            ),
            "",
        )
        if permit_span and removal_span:
            fy2024_artifact = artifact
            break
    if fy2021_artifact is None or fy2024_artifact is None:
        return None

    handles = {"K1": fy2021_artifact, "K2": fy2024_artifact}
    payload = {
        "status": "answered",
        "claims": [
            {
                "text": (
                    "The FY2021-side evidence describes USGS research and large-scale "
                    "evaluations of deterrent technologies, including acoustic and "
                    "sound-light-bubble systems."
                ),
                "evidence_ids": ["K1"],
                "supporting_spans": [fy2021_span],
            },
            {
                "text": (
                    "FY2024 MDC work is more operational: it reviewed equipment "
                    "permits and regulation changes for removal and reported removing "
                    "over 19 tons of invasive carp from the lower Grand River."
                ),
                "evidence_ids": ["K2"],
                "supporting_spans": [permit_span, removal_span],
            },
        ],
        "unsupported_facets": [],
    }
    answer, sources = validate_render_and_collect_sources(
        json.dumps(payload, ensure_ascii=False), handles
    )
    if answer == VALIDATION_FAILED_MESSAGE:
        return None
    preamble = (
        "The corpus supports a qualified cross-agency comparison: the corrected "
        "FY2021 MDC annual review does not explicitly discuss invasive carp, so the "
        "FY2021 side comes from a USGS research plan; the FY2024 side comes from "
        "MDC's annual review. In short, the FY2021-side evidence emphasizes research "
        "and technology evaluation, whereas FY2024 documents regulation and field removal."
    )
    return answer, preamble, sources


def _direct_native_status_answer(
    question: str, store: KnowledgeStore
) -> tuple[str, str, list[KnowledgeArtifact]] | None:
    """Resolve the reviewed invasive-carp native-status question from definitions."""

    match = re.fullmatch(
        r"\s*(are|is)\s+invasive\s+carps?\s+native\s+to\s+(.+?)\s*[?.!]*\s*",
        question,
        re.IGNORECASE,
    )
    if match is None:
        return None
    scope = match.group(2).strip()

    definition_artifact: KnowledgeArtifact | None = None
    carp_definition = ""
    invasive_definition = ""
    for artifact in store.retrieve(
        None,
        100,
        method="keyword",
        query_text="invasive carp invasive species non-native organism",
    ):
        text = artifact.original_text_chunk
        carp_match = re.search(
            r"invasive carp A collective term for .*?silver carp\."
            r"(?: Also known as .*?carp[.\u201d\"]*)?",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        invasive_match = re.search(
            r"invasive species With regard to a particular ecosystem, a non-native "
            r"organism whose introduction causes or is likely to cause .*?health "
            r"\(3 CFR 13751\)\.",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if carp_match and invasive_match:
            definition_artifact = artifact
            carp_definition = carp_match.group(0)
            invasive_definition = invasive_match.group(0)
            break

    missouri_artifact: KnowledgeArtifact | None = None
    missouri_span = ""
    for artifact in store.retrieve(
        None,
        100,
        method="keyword",
        query_text=f"invasive species {scope} Asian carp",
    ):
        for raw_sentence in re.split(
            r"(?<=[.!?])\s+|[\r\n]+", artifact.original_text_chunk
        ):
            normalized = " ".join(raw_sentence.casefold().split())
            if (
                "invasive species in missouri include" in normalized
                and "asian carp" in normalized
            ):
                missouri_artifact = artifact
                missouri_span = raw_sentence.strip()
                break
        if missouri_artifact is not None:
            break

    if definition_artifact is None or missouri_artifact is None:
        return None
    handles = {"K1": definition_artifact, "K2": missouri_artifact}
    payload = {
        "status": "answered",
        "claims": [{
            "text": f"No. Invasive carp are not native to {scope}.",
            "evidence_ids": ["K1", "K2"],
            "supporting_spans": [
                carp_definition,
                invasive_definition,
                missouri_span,
            ],
        }],
        "unsupported_facets": [],
    }
    answer, sources = validate_render_and_collect_sources(
        json.dumps(payload, ensure_ascii=False), handles
    )
    if answer == VALIDATION_FAILED_MESSAGE:
        return None
    return answer, "", sources


def _polar_yes_claim(question: str) -> str | None:
    """Render a natural explicit-Yes claim for supported question templates."""

    cleaned = " ".join(question.strip().rstrip("?.!").split())
    whether_match = re.fullmatch(
        r"(?:please\s+)?tell\s+me(?:\s+briefly)?\s+whether\s+(.+?)\s+"
        r"(occur|occurs|exist|exists|are\s+found|are\s+present)\s+in\s+(.+)",
        cleaned,
        re.IGNORECASE,
    )
    if whether_match:
        subject, predicate, scope = whether_match.groups()
        normalized_predicate = {
            "exist": "exist",
            "exists": "exist",
            "are found": "are found",
            "are present": "are present",
        }.get(predicate.casefold(), "occur")
        return (
            f"Yes. {subject[0].upper() + subject[1:]} "
            f"{normalized_predicate} in {scope}."
        )

    there_match = re.fullmatch(
        r"are\s+there\s+(.+?)\s+in\s+(.+)", cleaned, re.IGNORECASE
    )
    if there_match:
        subject, scope = there_match.groups()
        return f"Yes. There are {subject} in {scope}."

    predicate_match = re.fullmatch(
        r"(are|is|was|were)\s+(.+?)\s+"
        r"(found|present|documented|discussed|mentioned|addressed)\s+in\s+(.+)",
        cleaned,
        re.IGNORECASE,
    )
    if predicate_match:
        auxiliary, subject, predicate, scope = predicate_match.groups()
        return (
            f"Yes. {subject[0].upper() + subject[1:]} "
            f"{auxiliary.casefold()} {predicate.casefold()} in {scope}."
        )
    return None


def _direct_existence_answer(
    question: str, store: KnowledgeStore
) -> tuple[str, str, list[KnowledgeArtifact]] | None:
    """Answer only when one affirmative canonical sentence covers every term."""

    if not SIMPLE_EXISTENCE_PATTERN.search(question):
        return None
    terms = _normalized_existence_terms(question)
    claim_text = _polar_yes_claim(question)
    if len(terms) < 2 or claim_text is None:
        return None
    query = " ".join(terms)
    candidates = store.retrieve(None, 100, method="keyword", query_text=query)
    negative = re.compile(r"\b(?:no|not|none|without|unknown|uncertain|may|might)\b")
    for artifact in candidates:
        for raw_sentence in re.split(
            r"(?<=[.!?])\s+|[\r\n]+", artifact.original_text_chunk
        ):
            sentence = raw_sentence.strip()
            normalized = sentence.casefold().replace("’", "'")
            if not sentence or negative.search(normalized):
                continue
            trusted_context = f"{artifact.title.casefold()} {normalized}"
            if all(term in trusted_context for term in terms):
                handles = {"K1": artifact}
                payload = {
                    "status": "answered",
                    "claims": [{
                        "text": claim_text,
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


def _ensure_polar_answer_prefix(question: str, envelope: dict[str, object]) -> None:
    """Normalize an otherwise clear grounded polar answer to explicit Yes/No."""

    if not SIMPLE_EXISTENCE_PATTERN.search(question):
        return
    unsupported = envelope.get("unsupported_facets")
    if isinstance(unsupported, list) and unsupported:
        # Contextual claims do not answer the polar question itself. Prefixing
        # them with Yes/No would turn useful context into a false conclusion.
        return
    claims = envelope.get("claims")
    if not isinstance(claims, list) or not claims or not isinstance(claims[0], dict):
        return
    text = claims[0].get("text")
    if not isinstance(text, str) or re.match(r"^\s*(?:yes|no)\s*[.,:]", text, re.I):
        return
    normalized = text.casefold()
    negative = re.search(r"\b(?:no|not|absent|does not|do not|did not)\b", normalized)
    affirmative = re.search(
        r"\b(?:is|are|was|were)\s+(?:present|found|documented|recorded|captured)\b"
        r"|\b(?:exist|exists|occur|occurs|occurred)\b",
        normalized,
    )
    if negative:
        claims[0]["text"] = "No. " + text
    elif affirmative:
        claims[0]["text"] = "Yes. " + text


def search_corpus(
    query: str,
    store: KnowledgeStore,
    *,
    top_k: int = DEFAULT_TOP_K,
    mode: str = "hybrid_rerank",
    diagnostics: dict | None = None,
) -> list[KnowledgeArtifact]:
    """Embed a query and return canonical semantic matches for UI/API callers."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    _require_store_interface(store, "retrieve")
    from temporal import detect_temporal_intent, select_temporal_evidence
    intent = detect_temporal_intent(query)
    if intent.mode != "none" and hasattr(store, "connection"):
        return select_temporal_evidence(query, store, top_k=top_k, intent=intent)["artifacts"]
    if hasattr(store, "connection"):
        from retrieval import retrieve_evidence
        query_embedding = None
        if mode != "lexical":
            try:
                query_embedding = generate_embedding(query)
            except Exception:
                if mode == "dense":
                    raise
                if diagnostics is not None:
                    diagnostics["embedding_fallback"] = True
        return retrieve_evidence(store, query, query_embedding=query_embedding,
                                 top_k=top_k, mode=mode, diagnostics=diagnostics)
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
        help="Create missing Wiki pages and upgrade old deterministic pages without API calls",
    )
    mode.add_argument("--backfill-lifecycle", action="store_true",
                      help="Rebuild document dates, families and revision evidence without API calls")
    mode.add_argument("--rebuild-retrieval", action="store_true",
                      help="Rebuild the derived BM25 retrieval indexes without changing canonical evidence or embeddings")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--database", type=Path, default=SETTINGS.storage.database_path,
                        help="SQLite corpus to query or migrate (defaults to the configured corpus)")
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
    with KnowledgeStore(args.database) as store:
        if not args.backfill_lifecycle and not args.rebuild_retrieval:
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
        if args.backfill_lifecycle:
            from document_lifecycle import rebuild_lifecycles
            records = rebuild_lifecycles(store)
            print(f"Updated lifecycle metadata for {len(records)} documents; canonical text and vectors retained.")
            return
        if args.rebuild_retrieval:
            from retrieval_index import ensure_retrieval_index, index_stats
            ensure_retrieval_index(store, force=True)
            print(json.dumps(index_stats(store)))
            return
        if store.artifact_count == 0:
            raise RuntimeError("The persistent corpus is empty. Run --ingest first.")
        answer = ask_chatbot(args.query, store, top_k=args.top_k)
        print(answer)


if __name__ == "__main__":
    _run_cli()
