"""Diagnose Q9 funding-evidence loss from PipelineTracer JSON or an offline replay."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import SETTINGS
from data_models import KnowledgeArtifact
from database import KnowledgeStore
from pipeline_tracer import PipelineTracer, capture

QUERY = ("Compare the administrative advantages and funding structures of the Montreal "
         "Protocol Multilateral Fund and the Cleveland Climate Action Fund based strictly "
         "on the corpus.")
NEEDLES = ("120,000", "50,000")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, help="Existing PipelineTracer JSON")
    parser.add_argument("--database", type=Path, default=ROOT / "data/corpus.db")
    parser.add_argument("--replay", action="store_true",
                        help="Run retrieval with a fresh embedding; does not call the generator")
    parser.add_argument("--output", type=Path, default=ROOT / "traces")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", default=QUERY)
    return parser.parse_args()


@contextmanager
def copied_store(path: Path):
    source_path = path.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="v3-q9-") as directory:
        destination = Path(directory) / "corpus.db"
        source = sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)
        target = sqlite3.connect(destination)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        with KnowledgeStore(destination) as store:
            yield store


def _stage(trace, name):
    record = trace.get("stages", {}).get(name, {})
    return record.get("value") if record.get("status") == "captured" else None


def _event(trace, name):
    values = [event.get("value") for event in trace.get("events", [])
              if event.get("stage") == name]
    return values[-1] if values else None


def _contains_numbers(value):
    packed = json.dumps(value, ensure_ascii=False, default=str)
    return {needle: needle in packed for needle in NEEDLES}


def amount_excerpt(text, radius=260):
    positions = [text.find(needle) for needle in NEEDLES if needle in text]
    if not positions:
        return text[: radius * 2]
    start = max(0, min(positions) - radius)
    end = min(len(text), max(positions) + max(len(item) for item in NEEDLES) + radius)
    return text[start:end].replace("\n", " ")


def raw_score_events(initial, target_ids):
    matches = []
    for event in initial.get("raw_searches", []):
        value = event.get("value") or {}
        for rank, hit in enumerate(value.get("hits", []), 1):
            artifact_id = int(hit["artifact_id"])
            if artifact_id in target_ids:
                matches.append({
                    "stage": event.get("stage"),
                    "artifact_id": artifact_id,
                    "rank": rank,
                    "score": hit.get("score"),
                    "score_kind": value.get("score_kind"),
                    "higher_is_better": value.get("higher_is_better"),
                    "document_scope": value.get("document_ids"),
                })
    return matches

def funding_chunks(connection):
    rows = connection.execute(
        "SELECT artifact_id,document_id,title,page_number,printed_page_label,"
        "original_text_chunk FROM knowledge_artifacts "
        "WHERE document_id='DOC033' AND "
        "(original_text_chunk LIKE '%120,000%' OR original_text_chunk LIKE '%50,000%') "
        "ORDER BY artifact_id"
    ).fetchall()
    return [dict(row) for row in rows]


def replay(args):
    from api_clients import generate_embedding, LLM_MODEL
    from document_targets import comparison_targets
    from prompts import SYSTEM_PROMPT, build_synthesis_prompt
    from retrieval import retrieve_evidence

    args.output.mkdir(parents=True, exist_ok=True)
    with copied_store(args.database) as store, PipelineTracer(args.output) as tracer:
        capture("original_query", args.query)
        capture("rewritten_query", args.query)
        documents = [dict(row) for row in store.connection.execute(
            "SELECT document_id,title,year FROM documents")]
        target_ids = comparison_targets(args.query, documents)
        embedding = generate_embedding(args.query)
        diagnostics = {}
        artifacts = retrieve_evidence(
            store, args.query, query_embedding=embedding, top_k=args.top_k,
            diagnostics=diagnostics, document_ids=target_ids or None,
        )
        handles = {f"K{i}": artifact for i, artifact in enumerate(artifacts, 1)}
        capture("generation_evidence_handles", handles)
        prompt = build_synthesis_prompt(args.query, handles)
        capture("llm_context_payload", {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "diagnostic_only": True,
        })
    return tracer.path


def simulate_standard_selection(store, ranked_ids, top_k):
    from retrieval import _duplicates, _signature
    artifacts = dict(zip(ranked_ids, store._artifacts_by_ranked_ids(ranked_ids)))
    selected, suppressed, signatures = [], [], []
    for artifact_id in ranked_ids:
        signature = _signature(artifacts[artifact_id].original_text_chunk)
        if any(_duplicates(signature, prior, SETTINGS.retrieval.duplicate_threshold)
               for prior in signatures):
            suppressed.append(artifact_id)
            continue
        selected.append(artifact_id)
        signatures.append(signature)
        if len(selected) >= top_k:
            break
    return selected, suppressed

def simulate_decays(store, ranking, target_artifact_ids):
    from retrieval import select_diverse_evidence
    ranked = [int(item) for item in ranking.get("reranked_ids", [])]
    target_docs = ranking.get("target_document_ids", [])
    top_k = len(ranking.get("final_artifact_ids", []))
    if not ranked or not target_docs or not top_k:
        return []
    artifacts_list = store._artifacts_by_ranked_ids(ranked)
    artifacts = dict(zip(ranked, artifacts_list))
    rows = []
    for decay in (1.0, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1, 0.0):
        selected, suppressed, _ = select_diverse_evidence(
            ranked, artifacts, top_k, target_docs,
            duplicate_threshold=SETTINGS.retrieval.duplicate_threshold, decay=decay,
        )
        rows.append({
            "decay": decay,
            "funding_chunk_selected": any(item in selected for item in target_artifact_ids),
            "funding_chunk_suppressed_as_duplicate": any(item in suppressed for item in target_artifact_ids),
            "selected_ids": selected,
        })
    return rows


def diagnose(trace_path: Path, database: Path):
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    original = _stage(trace, "original_query")
    initial = _stage(trace, "initial_retrieval") or {}
    reranked_stage = _stage(trace, "reranked_evidence") or []
    context = _stage(trace, "llm_context_payload")
    ranking = _event(trace, "retrieval_ranking") or {}

    with copied_store(database) as store:
        targets = funding_chunks(store.connection)
        target_ids = {int(row["artifact_id"]) for row in targets}
        initial_ids = {int(item) for item in initial.get("fused_ids", [])}
        pre_mmr_ids = [int(item) for item in ranking.get("reranked_ids", [])]
        final_ids = [int(item) for item in ranking.get("final_artifact_ids", [])]
        simulations = simulate_decays(store, ranking, target_ids)
        selected_plus_one, suppressed_plus_one = simulate_standard_selection(
            store, pre_mmr_ids, len(final_ids) + 1
        )
        raw_scores = raw_score_events(initial, target_ids)
        final_chunks = []
        for artifact in store._artifacts_by_ranked_ids(final_ids):
            final_chunks.append({
                "artifact_id": final_ids[len(final_chunks)],
                "document_id": artifact.document_id,
                "page_number": artifact.page_number,
                "contains_120000": "120,000" in artifact.original_text_chunk,
                "contains_50000": "50,000" in artifact.original_text_chunk,
                "excerpt": amount_excerpt(artifact.original_text_chunk, 140),
            })

    initial_presence = bool(target_ids & initial_ids) or any(_contains_numbers(initial).values())
    pre_mmr_presence = bool(target_ids & set(pre_mmr_ids))
    final_presence = bool(target_ids & set(final_ids)) or any(_contains_numbers(reranked_stage).values())
    context_presence = any(_contains_numbers(context).values()) if context is not None else None

    if not initial_presence:
        loss = "initial_retrieval"
        diagnosis = "The funding chunk was never recalled; MMR could not have removed it."
    elif not pre_mmr_presence:
        loss = "reranking_before_mmr"
        diagnosis = "The funding chunk entered fusion but was removed before diversity selection."
    elif not final_presence and "diversity" in ranking:
        loss = "mmr_or_duplicate_suppression"
        diagnosis = "The funding chunk reached the MMR pool but was absent from final evidence."
    elif not final_presence and target_ids & {
            int(item) for item in ranking.get("duplicate_suppressed_ids", [])}:
        loss = "same_document_duplicate_suppression"
        diagnosis = "MMR was not active; same-document duplicate suppression removed the funding chunk."
    elif not final_presence:
        loss = "top_k_truncation"
        diagnosis = "MMR was not active; the funding chunk ranked below the final top-k cutoff."
    elif context_presence is False:
        loss = "context_assembly"
        diagnosis = "The funding chunk survived retrieval but was omitted while assembling the LLM payload."
    else:
        loss = "not_lost"
        diagnosis = "The funding chunk is present in final evidence and the LLM context."

    viable = [row["decay"] for row in simulations if row["funding_chunk_selected"]]
    suggested = max(viable) if viable else None
    return {
        "trace": str(trace_path),
        "query": original,
        "target_chunks": [{
            "artifact_id": row["artifact_id"],
            "document_id": row["document_id"],
            "title": row["title"],
            "page_number": row["page_number"],
            "printed_page_label": row["printed_page_label"],
            "contains_120000": "120,000" in row["original_text_chunk"],
            "contains_50000": "50,000" in row["original_text_chunk"],
            "excerpt": amount_excerpt(row["original_text_chunk"]),
        } for row in targets],
        "initial_retrieval": {
            "present": initial_presence,
            "target_ranks": {str(item): list(initial.get("fused_ids", [])).index(item) + 1
                             for item in target_ids if item in initial.get("fused_ids", [])},
            "numbers": _contains_numbers(initial),
            "raw_score_events": raw_scores,
        },
        "reranking_and_context": {
            "present_before_mmr": pre_mmr_presence,
            "pre_mmr_ranks": {str(item): pre_mmr_ids.index(item) + 1
                              for item in target_ids if item in pre_mmr_ids},
            "present_final": final_presence,
            "final_ranks": {str(item): final_ids.index(item) + 1
                            for item in target_ids if item in final_ids},
            "present_in_llm_context": context_presence,
            "numbers_in_context": _contains_numbers(context) if context is not None else None,
        },
        "selection_details": {
            "resolved_diversity_targets": ranking.get("target_document_ids", []),
            "final_artifact_ids": final_ids,
            "final_document_ids": ranking.get("final_document_ids", []),
            "duplicate_suppressed_ids": ranking.get("duplicate_suppressed_ids", []),
            "final_chunks": final_chunks,
            "top_k_plus_one_simulation": {
                "top_k": len(final_ids) + 1,
                "selected_ids": selected_plus_one,
                "suppressed_ids": suppressed_plus_one,
                "funding_chunk_selected": bool(target_ids & set(selected_plus_one)),
            },
            "funding_chunk_suppressed_as_duplicate": bool(
                target_ids & {int(item) for item in ranking.get("duplicate_suppressed_ids", [])}
            ),
        },
        "mmr": {
            "was_applied": "diversity" in ranking,
            "configured_decay": (ranking.get("diversity") or {}).get("decay"),
            "selection": (ranking.get("diversity") or {}).get("selection", []),
            "decay_simulation": simulations,
            "largest_tested_decay_preserving_chunk": suggested,
        },
        "loss_point": loss,
        "diagnosis": diagnosis,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = arguments()
    trace_path = replay(args) if args.replay else args.trace
    if trace_path is None:
        raise SystemExit("Pass --trace TRACE.json or --replay")
    report = diagnose(trace_path, args.database)
    report_path = trace_path.with_name(trace_path.stem + "-q9-diagnostic.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()