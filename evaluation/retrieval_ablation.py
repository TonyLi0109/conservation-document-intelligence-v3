"""Compare final-K retrieval variants; paid query embeddings are explicit opt-in."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from config import SETTINGS
from evaluation.dataset import DATASET_DIR, corpus_copy, fingerprint, load_dataset
from evaluation.retrieval_quality import canonical_errors, evidence_metrics, quality_fixture_store, validate_judgments


VARIANTS = ("legacy_keyword", "lexical", "lexical_rerank", "dense", "hybrid", "hybrid_rerank")
VECTOR_VARIANTS = {"dense", "hybrid", "hybrid_rerank"}


def query_cache_key(query, model, dimension):
    value = json.dumps([model, dimension, query], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_vector(vector, dimension):
    return (isinstance(vector, list) and len(vector) == dimension
            and all(type(value) in (float, int) and math.isfinite(value) for value in vector)
            and any(vector))


def load_query_vectors(queries, cache_path, *, allow_live=False):
    """Reuse only matching model/query/dimension vectors; never silently call API."""
    model, dimension = SETTINGS.models.embedding_model, SETTINGS.models.embedding_dimension
    path = Path(cache_path)
    cached = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"version": 1, "queries": {}}
    if not isinstance(cached, dict) or cached.get("version") != 1 or not isinstance(cached.get("queries"), dict):
        raise ValueError("Invalid retrieval query embedding cache")
    result, missing = {}, []
    for query in dict.fromkeys(queries):
        key = query_cache_key(query, model, dimension)
        item = cached["queries"].get(key)
        if item is None:
            missing.append(query)
            continue
        vector = item.get("embedding") if isinstance(item, dict) else None
        if (not isinstance(item, dict) or item.get("model") != model
                or item.get("dimension") != dimension or item.get("query") != query
                or not _valid_vector(vector, dimension)):
            raise ValueError("Cached query embedding has mismatched metadata or invalid values")
        result[query] = vector
    requested_count = 0
    if missing and allow_live:
        from api_clients import generate_embeddings
        vectors = generate_embeddings(missing)
        for query, vector in zip(missing, vectors, strict=True):
            if not _valid_vector(vector, dimension):
                raise ValueError("Provider query embedding has invalid dimensions or values")
            key = query_cache_key(query, model, dimension)
            cached["queries"][key] = {"query": query, "model": model, "dimension": dimension,
                                     "embedding": vector, "source": "provider_query_embedding"}
            result[query] = vector
        requested_count = len(missing)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return result, {"model": model, "dimension": dimension, "requested_query_count": requested_count,
                    "cached_query_count": len(result) - requested_count, "unavailable_query_count": len(set(queries)) - len(result)}


def run_variant(store, case, variant, *, vector=None, repeats=3):
    """One initial call and repeated warm calls, all returning the same final K."""
    if variant not in VARIANTS:
        raise ValueError("Unknown retrieval ablation variant")
    if variant in VECTOR_VARIANTS and vector is None:
        return {"case_id": case["case_id"], "variant": variant, "skipped": True,
                "reason": "No query embedding available; no dense or hybrid quality score was fabricated"}
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    from retrieval import retrieve_evidence
    errors = validate_judgments(store, case)
    samples, timings = [], []
    artifacts = []
    for _ in range(repeats + 1):
        trace = {}
        start = time.perf_counter()
        if variant == "legacy_keyword":
            artifacts = store.retrieve(None, case.get("k", 5), method="keyword", query_text=case["question"])
        else:
            mode = "hybrid_rerank" if variant == "lexical_rerank" else variant
            artifacts = retrieve_evidence(store, case["question"], top_k=case.get("k", 5), mode=mode,
                                          query_embedding=vector if variant in VECTOR_VARIANTS else None,
                                          diagnostics=trace)
        elapsed = (time.perf_counter() - start) * 1000
        samples.append(elapsed)
        timings.append({"retrieval_ms": trace.get("retrieval_ms", elapsed),
                        "rerank_ms": trace.get("rerank_ms", 0.0),
                        "selection_ms": trace.get("selection_ms", 0.0),
                        "index_ms": trace.get("index_ms", 0.0), "total_ms": elapsed})
    errors += canonical_errors(store, artifacts)
    return {"case_id": case["case_id"], "question": case["question"], "variant": variant, "skipped": False,
            "metrics": evidence_metrics(artifacts, case), "execution_errors": errors,
            "initial_call_total_ms": round(samples[0], 3), "warm_repeats": repeats,
            "warm_latency_ms": {key: round(statistics.mean(sample[key] for sample in timings[1:]), 3) for key in timings[0]},
            "dense_available": vector is not None and variant in VECTOR_VARIANTS,
            "retrieved_evidence": [{"document_id": a.document_id, "page_number": a.page_number} for a in artifacts]}


def summarize_variants(rows):
    result = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        for variant in VARIANTS:
            group = [row for row in rows if row["dataset"] == dataset and row["variant"] == variant]
            measured = [row for row in group if not row.get("skipped")]
            metrics = {}
            for key in set().union(*(row.get("metrics", {}).keys() for row in measured)):
                values = [row["metrics"].get(key) for row in measured]
                values = [value for value in values if type(value) in (float, int)]
                if values:
                    metrics[key] = statistics.mean(values)
            result[dataset + "/" + variant] = {
                "measured_cases": len(measured), "skipped_cases": len(group) - len(measured), "metrics": metrics,
                "mean_warm_latency_ms": {key: statistics.mean(row["warm_latency_ms"][key] for row in measured)
                                         for key in ("retrieval_ms", "rerank_ms", "selection_ms", "index_ms", "total_ms")} if measured else {},
            }
    return result


def render_ablation(report):
    lines = ["# Retrieval ablation", "", "All variants are scored on final K canonical chunks, with partial real-corpus labels.",
             "Recall/MRR/nDCG use unique document IDs in final-chunk order; exact-span and duplicate metrics audit the actual selected chunks.",
             "The unchanged legacy evaluation uses a different 20-chunk/5-document scope and is reported separately.",
             "lexical_rerank is lexical/document fusion plus deterministic reranking with no dense vector.",
             "Synthetic vectors measure controlled pipeline behavior; they are not real model-quality results.", "",
             "| Dataset / variant | Measured / skipped | Recall@K | MRR | nDCG@K | Exact span hit | Warm total ms |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def number(value):
        return "—" if value is None else f"{value:.4f}"
    for name, group in report["summary"].items():
        metrics = group["metrics"]
        lines.append(f"| {name} | {group['measured_cases']} / {group['skipped_cases']} | "
                     + " | ".join(number(metrics.get(key)) for key in ("recall_at_k", "mrr", "ndcg_at_k", "exact_span_hit_rate"))
                     + " | " + number(group["mean_warm_latency_ms"].get("total_ms")) + " |")
    lines += ["", "Warm latency excludes query-embedding API time and corpus loading. The initial call may reuse an index built by an earlier variant; it is not an isolated cold-start measurement.",
              "JSON retains initial-call totals and warm retrieval, feature/rerank, selection and index timings from production diagnostics.",
              "No external reranker or answer model is invoked. Missing query embeddings are skipped, not replaced with proxy vectors.",
              "Copyright/metadata and partial relevance limitations from EVALUATION.md still apply.", ""]
    return "\n".join(lines)


def run_ablation(*, corpus=SETTINGS.storage.database_path, cache_path=None, allow_live=False, repeats=3, case_ids=None):
    old = load_dataset(DATASET_DIR / "retrieval.json")
    quality = load_dataset(DATASET_DIR / "retrieval_quality.json")
    corpus_cases = [("legacy_13_final_chunks", {**case, "k": 5}) for case in old["cases"]]
    corpus_cases += [("expanded_corpus_final_chunks", case) for case in quality["cases"] if case["kind"] == "corpus"]
    fixture_cases = [case for case in quality["cases"] if case["kind"] == "fixture"]
    selected = set(case_ids or [])
    if selected:
        known = {case["case_id"] for _, case in corpus_cases} | {case["case_id"] for case in fixture_cases}
        if selected - known:
            raise ValueError("Unknown ablation case IDs: " + ", ".join(sorted(selected - known)))
        corpus_cases = [(group, case) for group, case in corpus_cases if case["case_id"] in selected]
        fixture_cases = [case for case in fixture_cases if case["case_id"] in selected]
    vectors, usage = load_query_vectors([case["question"] for _, case in corpus_cases],
                                       cache_path or DATASET_DIR.parent / "reports/query_embeddings.json", allow_live=allow_live)
    rows = []
    with corpus_copy(corpus) as store:
        corpus_fingerprint = store.evaluation_snapshot_fingerprint
        for group, case in corpus_cases:
            for variant in VARIANTS:
                result = run_variant(store, case, variant, vector=vectors.get(case["question"]), repeats=repeats)
                rows.append({**result, "dataset": group, "label_scope": "partial"})
    for case in fixture_cases:
        with quality_fixture_store(case) as store:
            for variant in VARIANTS:
                result = run_variant(store, case, variant, vector=case.get("query_vector"), repeats=repeats)
                rows.append({**result, "dataset": "synthetic_final_chunks", "label_scope": "controlled_fixture"})
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "ranking_unit": "final canonical evidence chunks",
            "document_metric_unit": "unique document IDs within final chunks; precision denominator K",
            "configuration": {"repeats": repeats, "live_embeddings_enabled": allow_live,
                              "query_embeddings": usage, "case_ids": sorted(selected),
                              "production_retrieval": {key: getattr(SETTINGS.retrieval, key) for key in SETTINGS.retrieval.__dataclass_fields__}},
            "fingerprints": {"corpus": corpus_fingerprint, "legacy_dataset": fingerprint(DATASET_DIR / "retrieval.json"),
                             "quality_dataset": fingerprint(DATASET_DIR / "retrieval_quality.json"),
                             "fixture_corpus": fingerprint(DATASET_DIR / "fixture_corpus.json")},
            "summary": summarize_variants(rows), "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=SETTINGS.storage.database_path)
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR.parent / "reports/retrieval_ablation")
    parser.add_argument("--embedding-cache", type=Path, default=DATASET_DIR.parent / "reports/query_embeddings.json")
    parser.add_argument("--live-embeddings", action="store_true", help="Explicitly opt in to paid embedding of missing benchmark questions only")
    parser.add_argument("--repeat", type=int, default=3, help="Warm retrieval repetitions after one initial call")
    parser.add_argument("--case", action="append", dest="case_ids")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 10:
        parser.error("--repeat must be between 1 and 10")
    report = run_ablation(corpus=args.corpus, cache_path=args.embedding_cache, allow_live=args.live_embeddings,
                          repeats=args.repeat, case_ids=args.case_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    markdown = render_ablation(report)
    (args.output_dir / "latest.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return int(any(row.get("execution_errors") for row in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
