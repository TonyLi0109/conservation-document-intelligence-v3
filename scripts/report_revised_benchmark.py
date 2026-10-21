"""Validate revised benchmark outputs and write the reviewable frozen-pool report."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_pdf_benchmark as run


def make_report(root, runner=run):
    run = runner
    initial_manifest = run.verify(root)
    complete = run.report(root)
    canonical = {r["page_id"]: r for r in map(json.loads, (root / "inputs/corpus.jsonl").read_bytes().splitlines())}
    pool = run.read(root / "candidate_pool.json")
    questions = {r["query_id"]: r["query"] for r in run.read(root / "inputs/questions.json")}
    if questions != {r["query_id"]: r["query"] for r in run.docx_rows(root / "inputs/Revised_9_Questions.docx")}:
        raise ValueError("Question texts differ from revised DOCX column")
    comparisons, errors, all_candidates = [], [], []
    terminal = getattr(run, "terminal_output", lambda folder, status: status.get("status") == "completed")
    for qid in run.QIDS:
        for method in run.ALL_METHODS:
            folder = root / "runs" / method / qid
            status = run.read(folder / "status.json") if (folder / "status.json").exists() else {"status": "not_run"}
            terminal_available = terminal(folder, status)
            if method == "cdirp_v3" and status["status"] == "failed":
                errors.append({"method": method, "query_id": qid, "category": "cdirp_pipeline_failure",
                               "message": status.get("error"), "provider_message": None,
                               "attempt_record": str((folder / "status.json").relative_to(root))})
            occurrences = run.read(folder / "candidate_occurrences.json") if (folder / "candidate_occurrences.json").exists() else []
            ranked = run.read(folder / "retrieval_results.json") if (folder / "retrieval_results.json").exists() else []
            ids = [r["page_id"] for r in ranked]
            if len(ids) != len(set(ids)) or any(pid not in canonical for pid in ids):
                raise ValueError(f"Invalid rankings: {method}/{qid}")
            if [r["rank"] for r in ranked] != list(range(1, len(ranked)+1)):
                raise ValueError("Ranking sequence broken")
            all_candidates.extend({"query_id": qid, "method": method, "page_id": r["page_id"],
                "rank": r["rank"], "score": r.get("score"), "doc_id": canonical[r["page_id"]]["doc_id"],
                "pdf_page": canonical[r["page_id"]]["pdf_page"], "provenance_file": str((folder / "candidate_occurrences.json").relative_to(root))}
                for r in ranked)
            attempts = [run.read(p) for p in folder.glob("calls/*/attempt_*/result.json")]
            for result_path in folder.glob("calls/*/attempt_*/result.json"):
                result = run.read(result_path)
                if result["status"] == "accepted":
                    command = run.read(result_path.parent / "command.json")
                    if command[command.index("--model") + 1] != run.METHODS[method]["model"]:
                        raise ValueError(f"Command model mismatch: {result_path}")
                    if method == "gpt6_luna" and 'model_reasoning_effort="low"' not in command:
                        raise ValueError("Luna command omitted low reasoning")
                    if method == "claude_haiku55" and command[command.index("--effort") + 1] != "low":
                        raise ValueError("Haiku command omitted low effort")
                    continue
                raw_path = result_path.parent / "provider_response.json"
                if not raw_path.exists():
                    raw_path = result_path.parent / "failed_provider_response.json"
                raw = run.read(raw_path) if raw_path.exists() else {}
                tail = (raw.get("stderr") or "")[-1000:]
                category = "rate_limit_429" if "429 Too Many Requests" in tail else result.get("error_type", "provider_error")
                provider_message = None
                for line in (raw.get("raw_text") or "").splitlines():
                    try:
                        event = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(event, dict) and event.get("api_error_status") == 429:
                        provider_message = event.get("result")
                        category = "subscription_session_limit" if "session limit" in str(provider_message) else "rate_limit_429"
                errors.append({"method": method, "query_id": qid, "category": category,
                               "message": result.get("error"), "provider_message": provider_message,
                               "attempt_record": str(result_path.relative_to(root))})
            if terminal_available:
                answer = run.read(folder / "answer.json")
                if not set(answer["cited_page_ids"]) <= set(ids):
                    raise ValueError(f"Answer cites unretained candidates: {method}/{qid}")
                if method != "cdirp_v3":
                    run.validate_output(answer, run.ANSWER,
                        run.read(folder / "calls/answer/request.json")["payload"])
                    accepted = list(folder.glob("calls/retrieve_*/accepted.json"))
                    if len(accepted) != run.read(root / "manifest.json")["baseline_retrieval"]["batches_per_question"]:
                        raise ValueError(f"Incomplete corpus scan: {method}/{qid}")
                    for request_file in folder.glob("calls/*/request.json"):
                        if run.read(request_file)["payload"]["question"] != questions[qid]:
                            raise ValueError("Old question leaked into inference")
                    for result in attempts:
                        if result["status"] == "accepted" and result["reported_model"] != run.METHODS[method]["model"]:
                            raise ValueError("Model mismatch")
                elif answer["question"] != questions[qid]:
                    raise ValueError("Old question in CDIRP")
            observed_llm_responses = None
            if method == "cdirp_v3":
                observed_llm_responses = sum(e["stage"] == "llm_raw_response"
                    for p in folder.glob("trace/*.json") for e in run.read(p)["events"])
            comparisons.append({
                "query_id": qid, "method": method, "status": status["status"],
                "terminal_output_available": terminal_available,
                "candidate_pages": len({r["page_id"] for r in occurrences}),
                "ranked_pages": len(ids),
                "last_invocation_seconds": status.get("elapsed_ns", 0)/1e9,
                "provider_execution_seconds_all_attempts": sum(r["elapsed_ns"] for r in attempts)/1e9 if method != "cdirp_v3" else None,
                "provider_attempts": len(attempts) if method != "cdirp_v3" else None,
                "observed_cdirp_llm_responses": observed_llm_responses,
                "error_attempts": sum(r["status"] != "accepted" for r in attempts),
                "answer_path": str((folder/"answer.md").relative_to(root)) if (folder/"answer.md").exists() else None,
            })
    for qid, records in pool["queries"].items():
        path = root / "pools" / qid / "pages.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({**canonical[r["page_id"]], "retrieved_by": sorted({o["method"] for o in r["provenance"]})},
                                   ensure_ascii=False)+"\n")
    totals = {}
    for method in run.ALL_METHODS:
        rows = [r for r in comparisons if r["method"] == method]
        totals[method] = {"completed": sum(r["status"] == "completed" for r in rows),
            "terminal_outputs": sum(r["terminal_output_available"] for r in rows),
            "failed_answers": sum(r["status"] == "failed" for r in rows),
            "query_page_memberships": sum(r["candidate_pages"] for r in rows),
            "provider_attempts": sum(r["provider_attempts"] or 0 for r in rows) if method != "cdirp_v3" else None,
            "provider_execution_seconds_all_attempts": sum(r["provider_execution_seconds_all_attempts"] or 0 for r in rows) if method != "cdirp_v3" else None,
            "last_invocation_seconds_sum": sum(r["last_invocation_seconds"] for r in rows),
            "error_attempts": sum(r["error_attempts"] for r in rows)}
    result = {"status": "frozen" if complete else "partial_not_frozen",
              "questions": questions, "methods": totals, "query_method_results": comparisons,
              "union_pages_per_question": {q: len(rows) for q, rows in pool["queries"].items()},
              "errors": errors,
              "limitations": [
                  "No historical qrels or proposed CoreGroups used for inference/scoring; revised references still require adjudication.",
                  "All retrieved-stage candidates retained, including candidates later excluded by CDIRP routing or reranking.",
                  "CDIRP native final evidence order is followed by remaining retrieved candidates; detailed native stage ranks are preserved separately.",
                  "Full-pool union may cover the entire corpus when CDIRP retrieves a complete dense ranking before scope filtering.",
                  "Model baselines answer from their top 30 pages; CDIRP uses native top_k=6 and native routing/answer limits. Compute budgets are not matched.",
                  "CLI attempts are not internal HTTP-call counts. CDIRP provider count is unknown; traces expose successful LLM responses, not every embedding/retry.",
                  "Provider times include failed attempts and resumed calls; last-invocation times do not include earlier abandoned invocations.",
                  "Historical experiments and archived attempts are excluded from this run's candidate pool.",
                  "Candidate pool generation is not answer-quality or CoreGroupRecall scoring.",
              ]}
    run.save(root / "review_summary.json", result)
    run.save(root / "api_errors.json", errors)
    for name, rows in (("all_ranked_candidates.csv", all_candidates), ("query_method_summary.csv", comparisons)):
        if rows:
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            run.write_durable(root / name, buffer.getvalue().encode("utf-8"), replace=True)
    lines = [
        "CDIRP REVISED NINE-QUESTION FAST-MODEL EVALUATION",
        f"Pool status: {result['status']}",
        "Source: REVISED QUESTION column of Revised_9_Questions.docx.",
        "Models: native CDIRP V3; GPT-6 Luna low; Claude Haiku 5.5 low; Gemini 3.8 Flash low.",
        "CDIRP internal configuration: gpt-5.6-sol reasoning=none; text-embedding-3-small.",
        "",
        "Question | CDIRP pages | Luna pages | Haiku pages | Flash pages | Deduplicated union",
    ]
    for qid in run.QIDS:
        counts = {r["method"]: (str(r["candidate_pages"]) if r["status"] == "completed"
                  else f"{r['candidate_pages']} [{r['status']}]")
                  for r in comparisons if r["query_id"] == qid}
        lines.append(" | ".join([qid, *(str(counts[m]) for m in run.ALL_METHODS), str(result["union_pages_per_question"][qid])]))
    lines.extend(["", "Execution totals (all provider attempts, including retries):"])
    for method, values in totals.items():
        seconds = values["provider_execution_seconds_all_attempts"]
        lines.append(f"{method}: {values['completed']}/9 successful; {values['terminal_outputs']}/9 terminal outputs; provider attempts={values['provider_attempts']}; "
                     f"provider seconds={None if seconds is None else round(seconds,2)}; failed attempts={values['error_attempts']}; "
                     f"failed answers={values['failed_answers']}; "
                     f"query wall time, final invocations summed={values['last_invocation_seconds_sum']:.2f}s")
    index = root / "runs/cdirp_v3/canonical_index.json"
    if index.exists():
        idx = run.read(index)
        lines.append(f"CDIRP canonical-index rebuild: {idx['chunks']} chunks; {idx['elapsed_ns']/1e9:.2f}s (separate setup cost).")
    lines.extend(["", "Recorded error categories: " + json.dumps(dict(Counter(r["category"] for r in errors))), "",
                  "Interpretation / limitations:", *["- " + s for s in result["limitations"]], "",
                  "Files: candidate_pool.json; pools/Q*/pages.jsonl; all_ranked_candidates.csv;",
                  "runs/<method>/Q*/answer.md; query_method_summary.csv; review_summary.json; api_errors.json."])
    unfinished = [f"{r['method']}/{r['query_id']} ({r['status']})"
                  for r in comparisons if not r["terminal_output_available"]]
    if unfinished:
        pending_methods = " ".join(m for m in run.ALL_METHODS
                                   if any(r["method"] == m and not r["terminal_output_available"] for r in comparisons))
        lines.extend(["", "Unfinished: " + ", ".join(unfinished),
                      "The union is provisional; do not use it as the final frozen four-method pool.",
                      "Resume from workspace root: python scripts/" + Path(run.__file__).name + " run --methods " + pending_methods,
                      "Then validate/export: python scripts/" + ("report_pdf_benchmark.py" if initial_manifest.get("pdf_inventory") else "report_revised_benchmark.py")])
    limits = sorted({e["provider_message"] for e in errors
                     if e["category"] == "subscription_session_limit" and e["provider_message"]})
    if limits:
        lines.extend(["", "Provider quota messages:"] + limits)
    if initial_manifest.get("pdf_inventory"):
        inventory = initial_manifest["pdf_inventory"]
        lines.extend(["", "PDF corpus: frozen copies of V3/PDF; physical PDF page numbers, including converted TXT documents.",
                      "Freeze criterion: every method completed retrieval and saved a terminal output. Native CDIRP fallback answers count as terminal outputs but remain reported as failed answers.",
                      f"Documents: {len(inventory)}; physical pages: {sum(x['physical_pages'] for x in inventory)}.",
                      "Text extraction: PyMuPDF sort=True; no OCR or image interpretation. Empty-text pages retain their canonical IDs.",
                      "Luna requests use a minimum 20-second start-to-start interval to reduce subscription rate limits. Provider execution time excludes this pacing; query wall time includes it.",
                      "Pages without extractable text: " + json.dumps({x['doc_id']: x['no_extractable_text_pages'] for x in inventory if x['no_extractable_text_pages']}),
                      "Page IDs belong to this PDF snapshot and must not be joined to historical text-derived qrels without remapping."])
    text = "\n".join(lines) + "\n"
    run.write_durable(root / "execution_report.txt", text.encode(), replace=True)
    manifest = run.read(root / "manifest.json")
    manifest["review_report_sha256"] = {name: run.sha(root/name) for name in (
        "execution_report.txt", "review_summary.json", "api_errors.json", "query_method_summary.csv",
        "all_ranked_candidates.csv")}
    manifest["pool_text_sha256"] = {str(p.relative_to(root)): run.sha(p) for p in (root/"pools").glob("Q*/pages.jsonl")}
    manifest["available_output_sha256"] = {
        str(p.relative_to(root)): run.sha(p)
        for method in run.ALL_METHODS for qid in run.QIDS
        for p in (root / "runs" / method / qid).glob("*")
        if p.is_file() and p.name in {"answer.json", "answer.md", "retrieval_results.json",
                                      "candidate_occurrences.json", "status.json"}
    }
    manifest["report_generator_sha256"] = run.sha(Path(__file__))
    run.save(root/"manifest.json", manifest)
    print(text)
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=run.DEFAULT)
    args = parser.parse_args()
    raise SystemExit(0 if make_report(args.root.resolve()) else 1)
