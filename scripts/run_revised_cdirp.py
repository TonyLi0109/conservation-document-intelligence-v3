"""Run real CDIRP orchestration on a canonical-PDF-only isolated database.

Candidate occurrences include raw dense/lexical hits, fused/reranked candidates,
temporal evidence, and generation context, preserving stage-local rankings.
No substitute retriever or synthetic answer is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read(path):
    return json.loads(path.read_bytes())


def canonical_id(document, page):
    if not str(page).isascii() or not str(page).isdigit():
        return None
    return f"{document}_P{int(page):04d}"


def collect_candidates(trace, artifact_pages, allowed):
    """Extract every returned ranked hit, not query scopes/allowed index entries."""
    rows = []
    list_keys = {"dense_ids", "lexical_ids", "document_chunk_ids", "fused_ids",
                 "dense_candidates", "lexical_candidates", "document_chunk_candidates",
                 "fused_ranking", "reranked_ids", "final_artifact_ids", "duplicate_suppressed_ids"}
    def walk(value, stage, path, rank=1):
        if isinstance(value, dict):
            pid = None
            if "document_id" in value and "page_number" in value:
                pid = canonical_id(value["document_id"], value["page_number"])
            elif isinstance(value.get("artifact_id"), int):
                pid = artifact_pages.get(value["artifact_id"])
            if pid:
                if pid not in allowed:
                    raise ValueError(f"Noncanonical retrieved evidence: {pid}")
                rows.append({"page_id": pid, "method": "cdirp_v3", "stage": stage,
                             "stage_path": path, "stage_rank": rank,
                             "score": value.get("score")})
            for key, child in value.items():
                if key in {"raw_searches", "documents", "metadata_signals"}:
                    # raw searches already have their own events; documents are scope metadata.
                    continue
                if key in list_keys and isinstance(child, list):
                    for n, artifact in enumerate(child, 1):
                        if isinstance(artifact, int) and artifact in artifact_pages:
                            rows.append({"page_id": artifact_pages[artifact], "method": "cdirp_v3",
                                         "stage": stage, "stage_path": path + "." + key, "stage_rank": n})
                elif isinstance(child, (dict, list)):
                    walk(child, stage, path + "." + key, rank)
        elif isinstance(value, list):
            for n, child in enumerate(value, 1):
                walk(child, stage, path + f"[{n-1}]", n)
    for event in trace["events"]:
        if event["stage"] in {
            "dense_search", "lexical_search", "initial_retrieval", "retrieval_ranking",
            "reranked_evidence", "temporal_selection", "generation_evidence_handles", "final_output",
        }:
            walk(event["value"], event["stage"], f"events[{event['sequence']}]")
    return rows


def canonical_chunks(pages, titles, target_words=750, overlap_words=100):
    """Use native chunk lengths, but never allow a chunk to cross a PDF page."""
    from data_models import KnowledgeArtifact
    for page in pages:
        words = list(re.finditer(r"\S+", page["page_text"]))
        for start in range(0, len(words), target_words-overlap_words):
            end = min(start+target_words, len(words))
            text = page["page_text"][words[start].start():words[end-1].end()]
            yield KnowledgeArtifact(page["doc_id"], titles[page["doc_id"]],
                                    str(page["pdf_page"]), text)
            if end == len(words):
                break


def run(root):
    # All potentially writable state is isolated from the production seed.
    target = root / "runs/cdirp_v3"
    target.mkdir(parents=True, exist_ok=True)
    canonical = {p["page_id"] for p in map(json.loads, (root / "inputs/corpus.jsonl").read_bytes().splitlines())}
    runtime = target / "runtime.db"
    if not runtime.exists():
        shutil.copy2(root / "inputs/cdirp_source.db", runtime)
        connection = sqlite3.connect(runtime)
        try:
            connection.execute("PRAGMA foreign_keys=OFF")
            all_rows = connection.execute("SELECT artifact_id,document_id,page_number FROM knowledge_artifacts").fetchall()
            excluded = [a for a, d, p in all_rows if canonical_id(d, p) not in canonical]
            for aid in excluded:
                connection.execute("DELETE FROM vector_embeddings WHERE artifact_id=?", (aid,))
                connection.execute("DELETE FROM knowledge_artifacts WHERE artifact_id=?", (aid,))
            # Generated Wiki caches may cite Web artifacts; canonical source retrieval only.
            for (table,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                if table.startswith("compiled_"):
                    connection.execute('DELETE FROM "' + table + '"')
            connection.execute("DELETE FROM retrieval_index_state")
            connection.commit()
            save(target / "runtime_policy.json", {
                "excluded_noncanonical_chunks": len(excluded), "retained_pdf_chunks": len(all_rows)-len(excluded),
                "compiled_knowledge": "disabled in isolated runtime to exclude precompiled/Web-derived evidence",
                "source_database_unchanged": True,
                "source_database_sha256": hashlib.sha256((root / "inputs/cdirp_source.db").read_bytes()).hexdigest(),
                "generator_configuration": "production default gpt-5.6-sol, reasoning_effort=none",
                "retrieval": "production native hybrid/temporal routing and default top_k=6"})
        finally:
            connection.close()
    from database import KnowledgeStore
    from pipeline_tracer import PipelineTracer
    import main as app
    from config import SETTINGS
    manifest = read(root / "manifest.json")
    # Same experiment-wide instruction; revised question string is passed unchanged.
    app.SYSTEM_PROMPT += "\n\n" + manifest["benchmark_instruction"]
    store = KnowledgeStore(runtime)
    # The historical database has chunks spanning physical pages, with a single
    # legacy start-page label. Reindex source text for strict canonical citations.
    canonical_index = target / "canonical_index.json"
    if not canonical_index.exists():
        from api_clients import generate_embeddings
        build_start = time.perf_counter_ns()
        pages = list(map(json.loads, (root / "inputs/corpus.jsonl").read_bytes().splitlines()))
        titles = dict(store.connection.execute("SELECT document_id,title FROM documents"))
        chunks = list(canonical_chunks(pages, titles))
        store.begin_rebuild()
        for offset in range(0, len(chunks), 100):
            batch = chunks[offset:offset+100]
            embeddings = generate_embeddings([a.original_text_chunk for a in batch])
            store.stage_batch(batch, embeddings)
            print(f"cdirp_v3 canonical index: {min(offset+100,len(chunks))}/{len(chunks)} chunks", flush=True)
        store.commit_rebuild()
        for table in ("retrieval_index_state", "document_lifecycle_state", "document_lifecycles"):
            store.connection.execute('DELETE FROM "' + table + '"')
        store.connection.commit()
        save(canonical_index, {"corpus_pages": len(pages), "chunks": len(chunks),
            "embedding_model": SETTINGS.models.embedding_model,
            "elapsed_ns": time.perf_counter_ns()-build_start,
            "chunk_policy": "native 750-word/100-word-overlap windows, bounded within each canonical physical PDF page; exact source substrings",
            "reason": "Historical index chunks crossed page boundaries; legacy start-page IDs cannot identify every supporting passage."})
    canonical_text = {p["page_id"]: p["page_text"] for p in map(json.loads, (root / "inputs/corpus.jsonl").read_bytes().splitlines())}
    for document, page, text in store.connection.execute("SELECT document_id,page_number,original_text_chunk FROM knowledge_artifacts"):
        if text not in canonical_text.get(canonical_id(document, page), ""):
            raise ValueError("Indexed chunk does not occur on its claimed canonical page")
    artifact_pages = {a: canonical_id(d, p) for a, d, p in store.connection.execute(
        "SELECT artifact_id,document_id,page_number FROM knowledge_artifacts")}
    try:
        failed = False
        for question in read(root / "inputs/questions.json"):
            qid = question["query_id"]
            directory = target / qid
            if (directory / "status.json").exists() and read(directory / "status.json")["status"] == "completed":
                continue
            save(directory / "status.json", {"status": "running"})
            start = time.perf_counter_ns()
            tracer = PipelineTracer(directory / "trace")
            try:
                diagnostics = {}
                with tracer:
                    answer, preamble, sources = app.ask_chatbot_with_context(
                        question["query"], store, diagnostics=diagnostics)
                rows = collect_candidates(tracer.data, artifact_pages, canonical)
                for row in rows:
                    row["query_id"] = qid
                save(directory / "candidate_occurrences.json", rows)
                # Native final order first; remaining candidates keep first-observed rank.
                priority = {"reranked_evidence": 0, "generation_evidence_handles": 1,
                            "final_output": 2, "retrieval_ranking": 3}
                ordered = sorted(enumerate(rows), key=lambda pair: (priority.get(pair[1]["stage"], 4), pair[0]))
                seen, ranked = set(), []
                for _, row in ordered:
                    if row["page_id"] not in seen:
                        seen.add(row["page_id"])
                        ranked.append({**row, "rank": len(ranked)+1,
                                       "rank_policy": "native final evidence first, then remaining retrieved candidates in observed order"})
                save(directory / "retrieval_results.json", ranked)
                citations = []
                for source in sources:
                    pid = canonical_id(source.document_id, source.page_number)
                    if pid not in canonical:
                        raise ValueError("Noncanonical final citation")
                    if pid not in citations:
                        citations.append(pid)
                save(directory / "answer.json", {"question": question["query"],
                    "answer_markdown": answer, "preamble": preamble, "cited_page_ids": citations,
                    "sources": [asdict(source) for source in sources], "diagnostics": diagnostics,
                    "trace_path": str(tracer.path.relative_to(root)), "generator": SETTINGS.models.llm_model})
                (directory / "answer.md").write_text(preamble + "\n\n" + answer + "\n", encoding="utf-8")
                errors = [e["value"] for e in tracer.data["events"] if e["stage"] == "synthesis_fallback"]
                if errors:
                    raise RuntimeError("Production returned source-list fallback: " + json.dumps(errors))
                if tracer.data["errors"]:
                    raise RuntimeError("Candidate trace capture failed")
                save(directory / "status.json", {"status": "completed", "elapsed_ns": time.perf_counter_ns()-start,
                                                 "candidate_pages": len(ranked), "finished_at_utc": datetime.now(timezone.utc).isoformat()})
                print(f"cdirp_v3 {qid}: COMPLETE, {len(ranked)} candidate pages", flush=True)
            except Exception as error:
                failed = True
                # Even a failed answer must not lose retrieval candidates.
                if tracer.data["events"]:
                    save(directory / "candidate_occurrences.json",
                         [{**r, "query_id": qid} for r in collect_candidates(tracer.data, artifact_pages, canonical)])
                save(directory / "status.json", {"status": "failed", "error": str(error),
                                                 "elapsed_ns": time.perf_counter_ns()-start})
                print(f"cdirp_v3 {qid}: FAILED: {error}", flush=True)
        return 1 if failed else 0
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.root.resolve()))
