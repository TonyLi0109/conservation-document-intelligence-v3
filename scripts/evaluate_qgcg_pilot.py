"""Offline QGCG blind review, verifier diagnostics and frozen-run Boolean recall.

Run: python scripts/evaluate_qgcg_pilot.py
No inference, provider login, retrieval rerun, or gold-label mutation occurs here.
The existing 46-run inventory defines the population before inspecting scores.
Keep blind_key.json and all comparison/diagnostic files away from blind reviewers.
Re-running retains the original per-question blinded assignment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
from importlib.metadata import version as package_version
import json
from pathlib import Path
import random
import re
import secrets
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from scripts import build_qgcg_groups as qgcg
from scripts import run_qgcg_pilot as pilot
from scripts.run_qgcg_pilot import json_bytes, write_durable

VERSION = "qgcg-evaluation-1.1-mixed-provider"
PILOT_IDS = ("Q2", "Q3")
STRESS_IDS = ("Q1", "Q6", "Q7")
ARMS = ("E0", "E1", "E2")
INVENTORY = "benchmark/analysis/gpt_v3_comparison_20261007/all_runs_recomputed.csv"
FROZEN_MANIFEST = "benchmark/analysis/core_group_page_comparison_20261008/manifest.json"
RUBRIC_FIELDS = ["required_group_validity", "required_group_validity_notes",
                 "requirement_completeness", "membership_and_boolean_correctness",
                 "human_edits_needed", "human_edits_description"]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_bytes())


def read_rows(path: Path, data: bytes | None = None):
    text = (path.read_bytes() if data is None else data).decode("utf-8-sig")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return list(csv.DictReader(io.StringIO(text, newline="")))


def csv_bytes(rows, fieldnames=None):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative.replace("\\", "/")).resolve()
    path.relative_to(root.resolve())
    return path


class Provenance:
    def __init__(self):
        self.inputs = {}
        self.relocations = []

    def read(self, path: Path, expected: str | None = None) -> bytes:
        data = path.read_bytes()
        sha = digest(data)
        if expected is not None and sha != expected:
            raise ValueError(f"Frozen input checksum mismatch: {path}")
        prior = self.inputs.get(str(path.resolve()))
        if prior and prior["sha256"] != sha:
            raise ValueError(f"Input changed during evaluation: {path}")
        self.inputs[str(path.resolve())] = {"path": str(path.resolve()), "sha256": sha,
                                           "bytes": len(data)}
        return data

    def json(self, path: Path, expected=None):
        return json.loads(self.read(path, expected))


def resolve_output(directory: Path, name: str, expected: str, audit: Provenance):
    """Accept a renamed groups file only if its bytes match the frozen manifest."""
    original = directory / name
    if original.exists():
        audit.read(original, expected)
        return original
    candidates = [p for p in directory.glob("groups*.json")
                  if name == "groups.json" and digest(p.read_bytes()) == expected]
    if len(candidates) != 1:
        raise ValueError(f"Cannot uniquely resolve frozen output {original}")
    actual = candidates[0]
    audit.read(actual, expected)
    audit.relocations.append({"manifest_path": str(original.resolve()),
                              "actual_path": str(actual.resolve()), "sha256": expected})
    return actual


def boolean_hit(options: list[list[str]], retrieved: set[str]) -> bool:
    # A missing option is never vacuously true. The inner conjunction cannot be empty.
    return any(bool(option) and all(page in retrieved for page in option) for option in options)


def boolean_text(options) -> str:
    return " OR ".join("(" + " AND ".join(option) + ")" for option in options) or "[] — no complete evidence option"


def group_metrics(groups, retrieved):
    required = [g for g in groups if g["status"] == "required"]
    unresolved = [g for g in groups if g["status"] == "unresolved"]
    if any(not g["evidence_options"] for g in required):
        raise ValueError("Required groups must have a nonempty evidence option")
    hits = sum(boolean_hit(g["evidence_options"], retrieved) for g in required)
    total = len(required) + len(unresolved)
    return {"hit_required_groups": hits, "required_groups": len(required),
            "unresolved_groups": len(unresolved),
            "supplementary_groups": sum(g["status"] == "supplementary" for g in groups),
            "denominator_excluding_unresolved": len(required),
            "denominator_including_unresolved": total,
            "CoreGroupRecall_excluding_unresolved": hits / len(required) if required else None,
            "CoreGroupRecall_including_unresolved_as_misses": hits / total if total else None}


def load_arm(directory: Path, audit: Provenance, corpus, questions):
    manifest = audit.json(directory / "run_manifest.json")
    if not manifest.get("finished_at_utc"):
        raise ValueError(f"Cannot evaluate an active run: {directory}")
    for record in manifest["artifacts"]:
        audit.read(safe_path(directory, record["relative_path"]), record["sha256"])
    outputs = {name: resolve_output(directory, name, row["sha256"], audit)
               for name, row in manifest["output_files"].items()}
    wrapper = audit.json(outputs["groups.json"], manifest["output_files"]["groups.json"]["sha256"])
    if (wrapper["arm"], wrapper["run_id"]) != (manifest["arm"], manifest["run_id"]):
        raise ValueError("Pilot groups and manifest identity disagree")
    trials = {}
    for qid, trial in wrapper["queries"].items():
        if trial["status"] != "completed":
            continue
        result = trial["result"]
        if result["question"] != questions[qid]:
            raise ValueError(f"Exact question differs: {directory}/{qid}")
        index = audit.json(directory / f"inputs/{qid}/pages.json")
        pages = {p["page_id"]: audit.read(safe_path(directory, p["relative_path"]), p["sha256"]).decode("utf-8")
                 for p in index}
        if any(corpus.get(pid) != text for pid, text in pages.items()):
            raise ValueError(f"Archived and canonical page text differ: {directory}/{qid}")
        for key in ("groups", "initial_groups"):
            qgcg.Pass2Output.model_validate(result[key], context={"pages": pages, "requirements": None})
        trials[qid] = result
    return {"directory": directory, "manifest": manifest, "trials": trials,
            "outcomes": wrapper["queries"], "groups_path": outputs["groups.json"],
            **{label: read_rows(outputs[name], audit.read(outputs[name], manifest["output_files"][name]["sha256"]))
               for label, name in (("requests", "requests.jsonl"), ("raw", "raw_responses.jsonl"),
                                    ("errors", "error_logs.jsonl"))}}


def normalize_baseline(question, corpus):
    output = []
    for group in question["groups"]:
        pages = group["member_page_ids"]
        if not pages or len(set(pages)) != len(pages):
            raise ValueError("Empty or duplicate E0 members")
        excerpts = [{"page_id": pid, "quote": corpus[pid][:1200],
                     "explanation": "Canonical page context; the excerpt is not a claim-specific citation.",
                     "kind": "canonical_context", "truncated": len(corpus[pid]) > 1200} for pid in pages]
        output.append({"group_id": group["group_id"], "requirement": group["facet_description"],
                       "why_indispensable": group["why_this_facet_is_core"], "status": "required",
                       "evidence_options": [[pid] for pid in pages], "source_excerpts": excerpts,
                       "issues": ["Manual review requested in source annotations."]
                       if group.get("manual_review_required") else []})
    return output


def normalize_result(qid, result):
    return [{**g, "group_id": f"{qid}-G{i:04d}"} for i, g in enumerate(result["groups"]["mappings"], 1)]


def load_frozen_runs(root: Path, audit: Provenance, frozen):
    pins = {p.replace("\\", "/"): sha for p, sha in frozen["source_sha256"].items()}
    inventory_path = root / INVENTORY
    inventory_bytes = audit.read(inventory_path, pins[INVENTORY])
    runs = []
    seen = set()
    for spec in read_rows(inventory_path, inventory_bytes):
        if spec["run_key"] in seen:
            raise ValueError("Duplicate frozen run key")
        seen.add(spec["run_key"])
        path = safe_path(root, spec["source"])
        ranking_bytes = audit.read(path, spec["source_sha256"])
        ranks = {}
        rank_sets = {}
        for row in read_rows(path, ranking_bytes):
            qid, page, rank = row["query_id"], row["page_id"], int(row["rank"])
            if rank < 1 or not re.fullmatch(r"DOC\d{3}_P\d{4}", page):
                raise ValueError(f"Invalid frozen ranking row: {path}")
            rr = ranks.setdefault(qid, {})
            ss = rank_sets.setdefault(qid, set())
            if page in rr or rank in ss:
                raise ValueError(f"Duplicate page/rank; do not silently rerank: {path}/{qid}")
            rr[page] = rank
            ss.add(rank)
        for qid, rr in ranks.items():
            if set(rr.values()) != set(range(1, len(rr) + 1)):
                raise ValueError(f"Frozen ranking has gaps: {path}/{qid}")
        runs.append({**spec, "ranks": ranks, "target_depth": int(spec["target_depth"])})
    return runs


def diagnostic_recall(runs, sets, query_ids=PILOT_IDS):
    rows, group_rows = [], []
    for run in runs:
        for qid in query_ids:
            rr = run["ranks"].get(qid, {})
            for k in (20, 100):
                status = ("N/A_missing_query" if not rr else "N/A_depth_not_available"
                          if run["target_depth"] < k else "available")
                retrieved = {p for p, rank in rr.items() if rank <= k}
                for arm in ARMS:
                    groups = sets[qid][arm]
                    metrics = group_metrics(groups, retrieved)
                    if status != "available":
                        for field in ("hit_required_groups", "CoreGroupRecall_excluding_unresolved",
                                      "CoreGroupRecall_including_unresolved_as_misses"):
                            metrics[field] = None
                    common = {"run_key": run["run_key"], "system": run["System_Name"],
                              "collection": run["collection"], "protocol": run["protocol"],
                              "query_id": qid, "arm": arm, "cutoff": k, "status": status}
                    rows.append({**common, **metrics, "target_depth": run["target_depth"],
                                 "saved_pages": len(rr), "returned_within_cutoff": len(retrieved),
                                 "short_of_cutoff": len(retrieved) < k,
                                 "source": run["source"], "source_sha256": run["source_sha256"]})
                    for group in groups:
                        options = group["evidence_options"]
                        group_rows.append({**common, "group_id": group["group_id"],
                                           "group_status": group["status"],
                                           "included_in_primary_denominator": group["status"] == "required",
                                           "hit": int(boolean_hit(options, retrieved)) if status == "available" else None,
                                           "evidence_options": json.dumps(options),
                                           "option_missing_pages": json.dumps([[p for p in opt if p not in retrieved]
                                                                                for opt in options]) if status == "available" else "",
                                           "member_ranks": json.dumps({p: rr.get(p) for opt in options for p in opt})})
    return rows, group_rows


def seconds(start, end):
    return round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(), 3)


def operational_stats(arm_data):
    """Count invocations once by call ID/attempt, not by stream events or replays."""
    rows = []
    manifest = arm_data["manifest"]
    key = lambda row: (row.get("call_id"), row.get("attempt"))
    for qid in manifest["query_ids"]:
        requests = [r for r in arm_data["requests"] if r.get("query_id") == qid]
        errors = [r for r in arm_data["errors"] if r.get("query_id") == qid]
        attempts = {key(r): r for r in requests if r["event"] == "provider_attempt"}
        raw = {key(r): r for r in arm_data["raw"] if r.get("query_id") == qid and r["event"] == "provider_raw"}
        model_specs = manifest.get("models", {})
        api_raw = {k: r for k, r in raw.items()
                   if model_specs.get(r.get("role"), {}).get("billing_mode") == "developer_api"}
        cli_raw = {k: r for k, r in raw.items() if k not in api_raw}
        api_started = {key(r) for r in requests if r["event"] == "api_request_started"}
        api_usage = Counter()
        for record in api_raw.values():
            try:
                usage = json.loads(record["raw_text"]).get("usageMetadata", {})
            except (ValueError, AttributeError):
                continue
            for name in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount"):
                if isinstance(usage.get(name), int):
                    api_usage[name] += usage[name]
        counts = Counter(r["event"] for r in errors)
        durations = [seconds(attempts[k]["timestamp_utc"], r["timestamp_utc"])
                     for k, r in raw.items() if k in attempts]
        trial = arm_data["trials"].get(qid)
        query_meta = manifest["queries"].get(qid, {})
        rows.append({"arm": manifest["arm"], "query_id": qid,
                     "status": query_meta.get("status", "not_started"),
                     "provider_attempts": len(attempts), "provider_responses_with_raw_capture": len(raw),
                     "cli_invocations_with_raw_capture": len(cli_raw),
                     "attempts_without_raw_capture": len(attempts.keys() - raw.keys()),
                     "cli_calls_by_pass": json.dumps(dict(Counter(r["pass_number"] for r in cli_raw.values())), sort_keys=True),
                     "retry_invocations": sum(r["attempt"] > 1 for r in raw.values()),
                     "retry_events": sum(counts[n] for n in ("provider_retry", "schema_response_retry", "manual_response_retry")),
                     "json_parsing_errors": counts["json_parsing_error"],
                     "pydantic_validation_errors": counts["pydantic_validation_error"],
                     "provider_errors": counts["provider_error"], "provider_decode_errors": counts["provider_decode_error"],
                     "pipeline_errors": counts["pipeline_error"],
                     "validated_response_reuses": sum(r["event"] == "validated_response_reused" for r in requests),
                     "saved_response_redecodes": sum(r["event"] == "saved_provider_response_redecoded" for r in requests),
                     "repair_rounds": len(trial["repair_history"]) if trial else None,
                     "accepted_repairs": sum(bool(x.get("accepted")) for x in trial["repair_history"]) if trial else None,
                     "initial_verifier_issues": len(trial["initial_issues"]) if trial else None,
                     "final_verifier_issues": len(trial["issues"]) if trial else None,
                     "observed_attempt_elapsed_seconds": round(sum(durations), 3),
                     "final_query_segment_seconds": seconds(query_meta["started_at_utc"], query_meta["finished_at_utc"])
                     if query_meta.get("finished_at_utc") else None,
                     "arm_wall_seconds_including_recovery_gaps": seconds(manifest["started_at_utc"], manifest["finished_at_utc"]),
                     "developer_api_calls": len(api_started),
                     "developer_api_responses_captured": len(api_raw),
                     "successful_api_responses": sum(r.get("exit_code") == 0 for r in api_raw.values()),
                     "api_token_usage": json.dumps(dict(api_usage), sort_keys=True),
                     "billing_policy": manifest["billing_policy"],
                     "internal_provider_http_call_count": "Gemini API: explicit starts counted; account CLI internal requests not exposed"})
    return rows


STYLE = """
:root{font-family:system-ui,sans-serif;color:#202b39;background:#f4f6f9;line-height:1.5}
body{max-width:1500px;margin:auto;padding:24px}h1,h2,h3{line-height:1.2;color:#162f50}
section,article{background:white;border:1px solid #d6dfea;border-radius:8px;padding:18px;margin:16px 0}
article{border-left:4px solid #7a98ba}table{border-collapse:collapse;width:100%;background:white;margin:12px 0}
th,td{padding:10px;border:1px solid #d6dfea;text-align:left;vertical-align:top}
th{background:#eaf0f7}blockquote{white-space:pre-wrap;border-left:3px solid #b4c7da;margin:8px 0;padding:10px;background:#f6f8fb}
.boolean{white-space:pre-wrap;overflow-wrap:anywhere;font:14px ui-monospace,monospace;background:#edf3fa;padding:10px}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;background:#e8eff7;font-size:13px}
.unresolved{background:#fff0d3}.pairs{display:grid;grid-template-columns:1fr 1fr;gap:18px}.pairs>div{min-width:0}
.rubric{display:grid;grid-template-columns:1fr 1fr;gap:12px}label{display:block;font-weight:600}
textarea,input,select{display:block;width:100%;box-sizing:border-box;padding:8px;margin-top:4px;font:inherit}
textarea{min-height:70px}button{background:#224c78;color:white;border:0;border-radius:5px;padding:12px;cursor:pointer}
details{margin:10px 0}summary{cursor:pointer;font-weight:600}.muted{color:#526377}.scroll{overflow:auto}
@media(max-width:850px){.pairs,.rubric{grid-template-columns:1fr}body{padding:12px}}
@media print{button{display:none}body{background:white}article{break-inside:avoid}}
"""


def e(value):
    return html.escape(str(value), quote=True)


def document(title, body, script=""):
    return (f'<!doctype html><html lang="en"><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{e(title)}</title><style>{STYLE}</style><body>{body}{script}</body></html>')


def excerpt_html(excerpts):
    blocks = []
    for excerpt in excerpts:
        tail = " [Context excerpt truncated; full page is available below.]" if excerpt.get("truncated") else ""
        blocks.append(f'<p><strong>{e(excerpt["page_id"])}</strong> — {e(excerpt.get("explanation", ""))}{e(tail)}</p>'
                      f'<blockquote>{e(excerpt["quote"])}</blockquote>')
    return "".join(blocks) or "<p>No source excerpts supplied.</p>"


def group_html(group, display_id=None, expanded=False):
    issue_text = "\n".join(group["issues"]) or "None recorded."
    return (f'<article><h4>{e(display_id or group["group_id"])} '
            f'<span class="badge {e(group["status"])}">{e(group["status"])}</span></h4>'
            f'<p><strong>{e(group["requirement"])}</strong></p><p>{e(group["why_indispensable"])}</p>'
            f'<div class="boolean">{e(boolean_text(group["evidence_options"]))}</div>'
            f'<details {"open" if expanded else ""}><summary>Source excerpts and explanations</summary>'
            f'{excerpt_html(group["source_excerpts"])}</details><p><strong>Issues:</strong> {e(issue_text)}</p></article>')


def blind_assignment(sets, path, seed=None):
    fingerprint = digest(json_bytes({q: sets[q] for q in PILOT_IDS}))
    if path.exists():
        key = load_json(path)
        if key["groups_sha256"] != fingerprint:
            raise ValueError("Blind groups changed; choose a new review directory to create a new assignment")
        if seed is not None and key["seed"] != seed:
            raise ValueError("Existing blind key uses a different seed")
        for qid in PILOT_IDS:
            if set(key["assignments"][qid].values()) != set(ARMS):
                raise ValueError("Invalid existing blind key")
        return key
    seed = secrets.randbits(64) if seed is None else seed
    rng = random.Random(seed)
    assignments = {}
    group_keys = {}
    for qid in PILOT_IDS:
        order = list(ARMS)
        rng.shuffle(order)
        assignments[qid] = dict(zip(("Set_A", "Set_B", "Set_C"), order))
        group_keys[qid] = {label: {f"{qid}-{label}-G{i:02d}": g["group_id"]
                                  for i, g in enumerate(sets[qid][arm], 1)}
                           for label, arm in assignments[qid].items()}
    key = {"version": VERSION, "seed": seed, "package_id": secrets.token_hex(12),
           "groups_sha256": fingerprint, "assignments": assignments, "group_ids": group_keys,
           "warning": "CONFIDENTIAL: withhold from blind reviewers; assignment is independent for each question."}
    write_durable(path, json_bytes(key))
    return key


def make_blind_rows(sets, questions, key):
    rows = []
    for qid in PILOT_IDS:
        for label, arm in key["assignments"][qid].items():
            for i, group in enumerate(sets[qid][arm], 1):
                rows.append({"query_id": qid, "question": questions[qid], "set_label": label,
                             "review_group_id": f"{qid}-{label}-G{i:02d}",
                             "requirement": group["requirement"], "why_indispensable": group["why_indispensable"],
                             "status": group["status"], "evidence_options": json.dumps(group["evidence_options"]),
                             "evidence_logic": boolean_text(group["evidence_options"]),
                             "source_excerpts": json.dumps(group["source_excerpts"], ensure_ascii=False),
                             "issues": json.dumps(group["issues"], ensure_ascii=False),
                             **{field: "" for field in RUBRIC_FIELDS}})
    return rows


BLIND_JS = r"""
<script>
const rows=JSON.parse(document.getElementById('review-data').textContent);
const packageId=document.getElementById('review-data').dataset.package;
const storeKey='qgcg-review-'+packageId;
const state={};
try{Object.assign(state,JSON.parse(localStorage.getItem(storeKey)||'{}'));}catch(err){}
document.querySelectorAll('[data-row]').forEach(el=>{
 const key=el.dataset.row+':'+el.dataset.field;
 el.value=state[key]||'';
 el.addEventListener('input',()=>{state[key]=el.value;try{localStorage.setItem(storeKey,JSON.stringify(state));}catch(err){}});
});
document.getElementById('export').addEventListener('click',()=>{
 const fields=Object.keys(rows[0]);
 const cell=v=>'"'+String(v??'').replace(/"/g,'""')+'"';
 const output=[fields.map(cell).join(',')];
 rows.forEach((row,index)=>{
  const result={...row};
  document.querySelectorAll('[data-row="'+index+'"]').forEach(el=>{result[el.dataset.field]=el.value;});
  output.push(fields.map(k=>cell(/^[=+@-]/.test(String(result[k]))?"'"+result[k]:result[k])).join(','));
 });
 const url=URL.createObjectURL(new Blob(['\ufeff'+output.join('\r\n')],{type:'text/csv;charset=utf-8'}));
 const link=document.createElement('a');link.href=url;link.download='blind_review_q2_q3_scored.csv';link.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
});
</script>
"""


def blind_html(rows, corpus, key):
    body = ['<h1>Independent evidence-group review: Q2 and Q3</h1>',
            '<p>Review each set against the exact question and canonical sources. Labels are independently '
            'randomized for each question. Judge semantic support, indispensability and completeness; '
            'a source quotation appearing verbatim does not establish that its interpretation is correct.</p>',
            '<p><strong>Boolean reading:</strong> options separated by OR are independently sufficient alternatives; '
            'pages joined by AND must be used together. An empty option list means no complete evidence option. '
            'Required groups are answer essentials with proposed support; unresolved groups record an essential '
            'evidence gap; supplementary groups are optional.</p>',
            '<p>Fill all four dimensions below. Record missing requirements at the set level in any group’s '
            'completeness field, prefixed “SET:”. Record a cross-group edit only once and reference the other '
            'review group IDs. A zero edit count means no edits after review; blank means unscored. '
            'Scores are stored locally where browser settings permit. Export CSV to preserve your review.</p>',
            '<button id="export" type="button">Download completed review CSV</button>']
    previous = None
    for index, row in enumerate(rows):
        section = (row["query_id"], row["set_label"])
        if section != previous:
            body.append(f'<h2>{e(section[0])} · {e(section[1])}</h2><p>{e(row["question"])}</p>')
            previous = section
        group = {**row, "evidence_options": json.loads(row["evidence_options"]),
                 "source_excerpts": json.loads(row["source_excerpts"]), "issues": json.loads(row["issues"])}
        body.append(group_html(group, row["review_group_id"], expanded=True))
        body.append('<section class="rubric">')
        labels = {"required_group_validity": "1. Required-group validity",
                  "required_group_validity_notes": "Validity notes",
                  "requirement_completeness": "2. Completeness: missing components or constraints",
                  "membership_and_boolean_correctness": "3. Membership / Boolean errors: false ORs, missing ANDs, wrong pages",
                  "human_edits_needed": "4. Number of human edits needed",
                  "human_edits_description": "Edit descriptions (reference affected review group IDs)"}
        for field in RUBRIC_FIELDS:
            attributes = f'id="r{index}-{field}" data-row="{index}" data-field="{field}"'
            if field == "required_group_validity":
                control = (f'<select {attributes}><option value="">Unscored</option><option>pass</option>'
                           '<option>fail</option><option>not_applicable</option></select>')
            elif field == "human_edits_needed":
                control = f'<input {attributes} type="number" min="0" step="1">'
            else:
                control = f'<textarea {attributes}></textarea>'
            body.append(f'<div><label for="r{index}-{field}">{e(labels[field])}</label>{control}</div>')
        body.append('</section>')
    body.append('<h2>Canonical source text reference</h2><p>Full page text for the union of cited pages across all sets; '
                'document IDs identify distinct documents. Use these pages to verify quotations in context.</p>')
    pages = sorted({p for row in rows for opt in json.loads(row["evidence_options"]) for p in opt}
                   | {x["page_id"] for row in rows for x in json.loads(row["source_excerpts"])})
    for pid in pages:
        body.append(f'<details><summary>{e(pid)}</summary><blockquote>{e(corpus[pid])}</blockquote></details>')
    encoded = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    script = f'<script type="application/json" id="review-data" data-package="{key["package_id"]}">{encoded}</script>' + BLIND_JS
    return document("Blind QGCG review — Q2 and Q3", "".join(body), script)


def md_cell(value):
    return str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def md_table(headers, rows):
    return ("| " + " | ".join(map(md_cell, headers)) + " |\n| " + " | ".join("---" for _ in headers) + " |\n"
            + "\n".join("| " + " | ".join(map(md_cell, row)) + " |" for row in rows) + "\n")


def html_table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{e(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{e(c)}</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def group_summary(group):
    return (f'{group["group_id"]} [{group["status"]}]: {group["requirement"]}\n'
            f'{boolean_text(group["evidence_options"])}')


def stress_structure(groups):
    options = [option for group in groups for option in group["evidence_options"]]
    multi = [option for option in options if len(option) > 1]
    cross_document = sum(len({page.split("_P")[0] for page in option}) > 1 for option in multi)
    adjacent = 0
    for option in multi:
        parsed = {(page.split("_P")[0], int(page.split("_P")[1])) for page in option}
        adjacent += any((doc, number + 1) in parsed for doc, number in parsed)
    return [sum(g["status"] == "required" for g in groups),
            sum(g["status"] == "unresolved" for g in groups),
            sum(g["status"] == "supplementary" for g in groups),
            len(options) - len(multi), len(multi), cross_document, adjacent]


def intervention_rows(arm_data):
    rows = []
    for qid, result in arm_data["trials"].items():
        for issue in result["initial_issues"]:
            reference = issue["requirement_reference"]
            match = re.fullmatch(r"G(\d+)", reference)
            index = int(match[1]) - 1 if match else None
            initial = result["initial_groups"]["mappings"]
            final = result["groups"]["mappings"]
            before = initial[index] if index is not None and index < len(initial) else None
            # Requirement references are ordinal per pass. Match text after split/merge;
            # do not silently transfer an old ordinal to a different requirement.
            after = next((g for g in final if before and g["requirement"] == before["requirement"]), None)
            correspondence = "identical requirement text" if after else "not uniquely established"
            if before and after is None and issue["recommended_action"] == "revise":
                tracked_requirement = before["requirement"]
                for history in result["repair_history"]:
                    if not history.get("accepted"):
                        continue
                    previous = history["before"]["mappings"]
                    candidate = history["candidate"]["mappings"]
                    old_names = {g["requirement"] for g in previous}
                    new_names = {g["requirement"] for g in candidate}
                    removed, added = old_names - new_names, new_names - old_names
                    # A unique rewrite is traceable without assuming ordinal identity.
                    if removed == {tracked_requirement} and len(added) == 1 and len(previous) == len(candidate):
                        tracked_requirement = next(iter(added))
                        correspondence = "unique requirement rewrite in accepted repair history"
                after = next((g for g in final if g["requirement"] == tracked_requirement), None)
            rows.append({"query_id": qid, "group": f"{qid}-{reference}",
                         "issue_type": issue["issue_type"], "severity": issue["severity"],
                         "recommended_action": issue["recommended_action"],
                         "explanation": issue["evidence_based_explanation"],
                         "correspondence": correspondence,
                         "before_requirement": before["requirement"] if before else "new requirement",
                         "after_requirement": after["requirement"] if after else "see complete repair history",
                         "before_status": before["status"] if before else "new requirement",
                         "after_status": after["status"] if after else "removed / rewritten; inspect repair history",
                         "before_options": before["evidence_options"] if before else [],
                         "after_options": after["evidence_options"] if after else [],
                         "repair_rounds": len(result["repair_history"]),
                         "accepted_rounds": [x["round"] for x in result["repair_history"] if x.get("accepted")],
                         "remaining_issues": len(result["issues"])})
    return rows


def make_reports(questions, sets, arms, stats, recalls, stress_state, pool_counts, question_differences):
    md = ["# QGCG pilot comparison: Q2 and Q3, with Q1/Q6/Q7 stress checks\n"]
    body = ['<h1>QGCG pilot comparison: Q2 and Q3</h1><p>Unblinded diagnostics and stress-check extension.</p>']

    def heading(text, level=2):
        md.append(f'{"#" * level} {text}\n')
        body.append(f'<h{level}>{e(text)}</h{level}>')

    def paragraph(text):
        md.append(text + "\n")
        body.append(f'<p>{e(text)}</p>')

    def table(headers, rows):
        md.append(md_table(headers, rows))
        body.append(html_table(headers, rows))

    heading("Experimental contrasts and interpretation")
    paragraph("E0 is the frozen, human-reviewed original baseline. Its member-page lists are converted to singleton OR alternatives, preserving the original any-member-hit convention. E1 runs QGCG on the same per-question core pool; E2 uses the expanded per-question relevant pool. The 62 and 127 totals are query–page memberships across all nine questions, not page counts supplied to every question. No original labels or rankings are changed.")
    paragraph("E0→E1 is the intended algorithm contrast on a matched page pool. E1→E2 is the intended expanded-pool contrast, but Pass 1 was generated independently in each arm: these are not identical fixed-requirement, seed-controlled replications. Requirement wording/order, stochastic outputs and verifier sensitivity can also affect differences. Report them as diagnostic contrasts, not a clean causal estimate of the pool effect.")
    paragraph("Passing JSON/Pydantic and exact-quote checks does not establish semantic correctness. Zero final verifier issues is not human acceptance. The primary evaluation remains the four-dimensional blinded human review; all rubric fields are intentionally unscored.")
    paragraph("The baseline retained slightly different question wording for Q1/Q3 and a rephrased Q6. Blind review uses the clean benchmark/questions.csv question uniformly; E0 groups and historical retrieval rankings are preserved exactly. This wording change is an additional limitation of the baseline contrast.")
    table(["Question", "Frozen baseline wording", "Clean pilot wording"],
          [[r["query_id"], r["baseline_question"], r["clean_question"]] for r in question_differences])
    requirement_rows = []
    for q in (*PILOT_IDS, *STRESS_IDS):
        scope = "pilot" if q in PILOT_IDS else "stress"
        trials = [arms.get((scope, a), {}).get("trials", {}).get(q) for a in ("E1", "E2")]
        identical = str(trials[0]["initial_requirements"] == trials[1]["initial_requirements"]) if all(trials) else "pending stress run"
        requirement_rows.append([q, pool_counts["E1"][q], pool_counts["E2"][q], identical])
    table(["Question", "E1 core pages", "E2 relevant pages", "Identical Pass 1 requirements?"], requirement_rows)
    if any(scope == "stress" for scope, _ in arms):
        paragraph("Stress-check protocol amendment: the user explicitly authorized Gemini developer API while GPT and Claude continue through their subscription CLIs. Pass 3 uses gemini-3.1-pro-preview with Low thinking, temperature 1.0 and no tools. The earlier Q2/Q3 pilot used gemini-3.1-pro-low through an account CLI. The model family/Low label does not establish backend or prompt-wrapper equivalence; the transport/billing change is recorded in each stress manifest and is a comparability limitation.")
    heading("Pass 3 interventions: initial issues → targeted repair → verification")
    paragraph("E2/Q2-G0001 changed from required to unresolved with [] options. The national habitat-loss and invasive-species statistics do not establish a Missouri-specific top-three ranking. Excluding this gap from recall reduces the Q2 denominator from three to two; the alternate denominator preserves the gap as a miss.")
    paragraph("E2/Q3-G0002 removed incomplete singleton OR alternatives and retained two complete singleton alternatives plus a three-page AND option. E2/Q3-G0003 removed the local case-study and WRE-only singleton options, retaining DOC036_P0288 as its complete evidence option. The exact transitions and verifier reasoning appear below.")
    interventions = []
    for (scope, arm), data in arms.items():
        current = intervention_rows(data)
        interventions.extend({"scope": scope, "arm": arm, **row} for row in current)
        if not current:
            paragraph(f"{scope}/{arm}: no initial verifier issues in completed trials. No targeted repair was triggered.")
        for row in current:
            heading(f'{scope}/{arm} · {row["group"]} · {row["issue_type"]} ({row["severity"]})', 3)
            table(["Before", "After"], [[row["before_requirement"], row["after_requirement"]],
                  [row["before_status"], row["after_status"]],
                  [boolean_text(row["before_options"]), boolean_text(row["after_options"])]] )
            paragraph(row["explanation"])
            paragraph("Before/after correspondence: " + row["correspondence"] + ".")
            paragraph(f'Action: {row["recommended_action"]}; accepted repair rounds: {row["accepted_rounds"]}; final verifier issues: {row["remaining_issues"]}.')
    heading("Repair history audit")
    for (scope, arm), data in arms.items():
        for qid, result in data["trials"].items():
            for record in result["repair_history"]:
                changed = []
                before = {g["requirement"]: g for g in record["before"]["mappings"]}
                for i, candidate in enumerate(record.get("candidate", {}).get("mappings", []), 1):
                    prior = before.get(candidate["requirement"])
                    fields = [key for key in candidate if prior is None or candidate[key] != prior.get(key)]
                    if fields:
                        changed.append(f'{qid}-G{i:04d}: {", ".join(fields)}')
                table(["Trial", "Round", "Accepted", "Changed fields", "Post-repair issues / validation error"],
                      [[f"{scope}/{arm}/{qid}", record["round"], record.get("accepted"), "; ".join(changed),
                        str(record.get("issues_after", [])) + " " + str(record.get("validation_error", ""))]])
    paragraph("The machine-readable verifier_interventions.json preserves the complete initial_issues and repair_history (including before/candidate mappings), plus final issues. Original audit artifacts remain unchanged.")
    heading("Outstanding human review priorities")
    paragraph("E1/Q2-G0001 still accepts DOC001_P0028 alone for a Missouri top-three ranking; E1's verifier raised no issue. Compare this against E2's geographic-scope correction. E1/Q3 also retains singleton impact/management alternatives that warrant review against its own requirement wording. Do not transfer E2's verdict mechanically across differently worded requirements.")
    paragraph("Both Pass 1 outputs interpret Q2 as an explicit first-to-third ordering, although the exact question asks for the top three. The reviewer should judge whether that extra ordering demand is indispensable. For Q3, also check that the explicit three-document constraint, findings/recommendations distinction and any overlapping groups are properly represented.")
    for (scope, arm), data in arms.items():
        if scope != "stress":
            continue
        for qid, result in data["trials"].items():
            initial_requirements = {g["requirement"] for g in result["initial_groups"]["mappings"]}
            final_requirements = {g["requirement"] for g in result["groups"]["mappings"]}
            if initial_requirements - final_requirements:
                paragraph(f"{scope}/{arm}/{qid} rewrote or removed an initial requirement during repair. Human review must check that the revision remains faithful to the question, rather than narrowing the question solely to fit the available sources. The before/after wording is retained in the intervention audit.")

    for qid in (*PILOT_IDS, *STRESS_IDS):
        heading(f"{qid}: side-by-side group sets")
        paragraph(questions[qid])
        if qid in STRESS_IDS:
            focus = {"Q1": "Schedule-page equivalence, especially baseline Q1_G03: newer publication dates and planning/revision periods must not be conflated.",
                     "Q6": "Cross-document complementarity, especially baseline Q6_G02: distinguish direct causal evidence from association or general risk; partial pages may require AND.",
                     "Q7": "Continuous passages over adjacent pages: preserve co-dependencies needed to complete life-cycle strategy, countries and named partners."}[qid]
            paragraph("Stress focus: " + focus)
            paragraph("Stress execution: " + stress_state["questions"][qid])
            table(["Arm", "Required", "Unresolved", "Supplementary", "Singleton options", "Multi-page AND options",
                   "Cross-document AND options", "AND options containing adjacent pages"],
                  [[arm, *stress_structure(groups)] for arm, groups in sets[qid].items()])
            paragraph("These counts describe the saved Boolean structure, not semantic pass/fail judgments. Adjacent-page AND means one option contains consecutive physical pages of the same document. Different group granularity can legitimately change these counts; assess the full requirements and excerpts below.")
        for left, right, label in (("E0", "E1", "Matched core-pool / algorithm contrast"),
                                   ("E1", "E2", "Expanded-pool contrast")):
            heading(f"{left} vs {right} — {label}", 3)
            left_groups, right_groups = sets[qid].get(left), sets[qid].get(right)
            def cell(groups):
                return "\n\n".join(group_summary(g) for g in groups) if groups is not None else "Not run / not completed; no groups fabricated."
            md.append(md_table([left, right], [[cell(left_groups), cell(right_groups)]]))
            body.append('<div class="pairs">')
            for arm, groups in ((left, left_groups), (right, right_groups)):
                body.append(f'<div><h4>{arm}</h4>')
                body.append("".join(group_html(g) for g in groups) if groups is not None else '<p>Not run / not completed; no groups fabricated.</p>')
                body.append('</div>')
            body.append('</div>')
        paragraph("Group order is local to each set. Side-by-side columns do not imply one-to-one equivalence of group ordinals.")

    heading("Frozen retrieval diagnostic")
    paragraph("CoreGroupRecall@k hits a required group iff at least one nonempty option has every member page at original frozen rank ≤ k. OR applies across options; AND applies within an option. Supplementary groups are excluded. Unresolved groups are reported separately: primary recall excludes them; alternate recall counts them as misses. No required groups yields N/A for the primary score, not 0. Original ranks are neither deduplicated nor compressed.")
    paragraph("All 46 runs from the existing hash-pinned inventory are included. Comparisons retain collection and protocol labels; earlier paired versions, shared-pool controls and historical top-20 runs are not interchangeable replications. A run with declared depth 20 has N/A at 100. A depth-100 run returning fewer than 100 pages is evaluated on its actual saved pages and marked short_of_cutoff; missing pages are not invented. The CSV includes group counts, both denominators, source hashes and availability. diagnostic_group_hits_q2_q3.csv exposes each group's retrieved ranks and missing AND dependencies.")
    paragraph("Recall changes reflect changes in the target groups, evidence logic and denominator, not improvements to frozen retrieval itself. In particular, a higher score after excluding an unresolved requirement does not demonstrate a more complete answer.")
    compact = {}
    for row in recalls:
        key = (row["collection"], row["run_key"], row["system"], row["query_id"], row["cutoff"])
        compact.setdefault(key, {})[row["arm"]] = row
    def score(value):
        return "N/A" if value is None else f"{value:.4f}"
    collections = list(dict.fromkeys(key[0] for key in compact))
    for collection in collections:
        heading(collection, 3)
        table(["Run", "System", "Question", "k", "E0", "E1", "E2 excl.", "E2 incl.", "Δ E1−E0", "Δ E2−E1"],
              [[key[1], key[2], key[3], key[4],
                *[score(values[a]["CoreGroupRecall_excluding_unresolved"]) for a in ARMS],
                score(values["E2"]["CoreGroupRecall_including_unresolved_as_misses"]),
                *[score(values[b]["CoreGroupRecall_excluding_unresolved"] - values[a]["CoreGroupRecall_excluding_unresolved"])
                  if values[a]["CoreGroupRecall_excluding_unresolved"] is not None and values[b]["CoreGroupRecall_excluding_unresolved"] is not None else "N/A"
                  for a, b in (("E0", "E1"), ("E1", "E2"))]]
               for key, values in compact.items() if key[0] == collection])

    heading("Operational statistics and model provenance")
    paragraph("Counts include failed historical segments and recoveries, not just the successful final path. CLI invocations are unique subscription call_id/attempt pairs with provider_raw capture; stdout, stderr, stream events, lexical redecoding and replay are not extra calls. Gemini developer API requests are counted separately from explicit api_request_started events; connection failures are still attempted requests, not confirmed billable generations. Captured/successful API responses and usageMetadata are preserved in operational_stats.csv. Preflight GET model-metadata requests are non-inference and excluded. An attempt without raw capture is shown separately; the original pilot includes local failures before inference. Internal HTTP calls/retries inside provider CLIs are not fully exposed and must not be inferred from CLI counts.")
    paragraph("Observed attempt elapsed time sums provider_attempt→provider_raw intervals, including local wrapper/capture overhead. Arm wall time includes waiting and recovery gaps and is repeated per question for context; do not sum repeated arm times. Final query-segment time can be short because it reuses earlier saved responses. Models expose account-CLI model identifiers; opaque backend revisions remain unknown.")
    table(["Scope", "Arm", "Question", "Attempts / captured CLI", "Gemini API requests", "Retries (calls / events)", "JSON / schema / decode errors", "Replays", "Repair rounds", "Attempt seconds", "Arm wall seconds"],
          [[s["scope"], s["arm"], s["query_id"], f'{s["provider_attempts"]} / {s["cli_invocations_with_raw_capture"]}',
            s["developer_api_calls"],
            f'{s["retry_invocations"]} / {s["retry_events"]}',
            f'{s["json_parsing_errors"]} / {s["pydantic_validation_errors"]} / {s["provider_decode_errors"]}',
            s["validated_response_reuses"], s["repair_rounds"], s["observed_attempt_elapsed_seconds"],
            s["arm_wall_seconds_including_recovery_gaps"]] for s in stats])
    model_rows = []
    for (scope, arm), data in arms.items():
        for role, model in data["manifest"]["models"].items():
            model_rows.append([scope, arm, role, model["requested_model"], model["transport"], model["billing_mode"],
                               model["transport_version"], ", ".join(sorted({c["reported_model"] for c in model["observed_calls"]})),
                               "unknown" if all(c.get("backend_revision") is None for c in model["observed_calls"]) else "see manifest"])
    table(["Scope", "Arm", "Role", "Requested model", "Transport", "Billing", "Transport version", "Observed identifiers", "Backend version metadata"], model_rows)
    heading("Stress-check execution status")
    paragraph(stress_state["summary"])
    stress_outcomes = []
    for qid in STRESS_IDS:
        for arm in ("E1", "E2"):
            result = arms.get(("stress", arm), {}).get("trials", {}).get(qid)
            if result:
                stress_outcomes.append([qid, arm, result["review_status"], len(result["initial_issues"]),
                                        len(result["repair_history"]), len(result["issues"]),
                                        result.get("repair_error") or "none"])
    if stress_outcomes:
        table(["Question", "Arm", "Review status", "Initial verifier issues", "Repair rounds",
               "Remaining verifier issues", "Repair error"], stress_outcomes)
    if stress_state.get("preflight"):
        preflight = stress_state["preflight"]
        paragraph(f'Historical preflight failure (retained for audit, not the status of completed trials): {preflight["timestamp_utc"]}. {preflight["message"]}')
        paragraph("Available subscription models returned by the CLI: " + ", ".join(preflight.get("details", {}).get("available_models", [])))
    paragraph("The evaluator automatically incorporates finalized stress_checks/E1 and stress_checks/E2 outputs on rerun. Q2/Q3 blind assignments remain frozen. Pending or failed stress trials are shown explicitly, never as successful empty groups.")
    heading("Review package and reproducibility")
    paragraph("Distribute only blind_review_q2_q3.html and blind_review_q2_q3.csv to blind reviewers. Keep blind_key.json, this report, diagnostic CSVs and manifests with the experiment coordinator. HTML supports local rubric entry and CSV export; canonical full-page context is embedded without external network dependencies. The key stores the random seed and per-question label mapping separately.")
    paragraph("evaluation_manifest.json hashes the exact evaluator, frozen inventory/rankings, original baseline, questions/corpus, pilot inputs, logs and generated outputs. Renamed groups files are accepted only by exact agreement with their original manifest checksum; the relocation is recorded, without rewriting the original manifests. Human ratings and adjudicated labels are not generated by this script.")
    return "\n".join(md), document("Unblinded QGCG comparison", "".join(body)), interventions


def evaluate(data_root: Path, output_root: Path | None = None, blind_seed=None):
    root = data_root.resolve()
    pilot_root = root / "benchmark/qgcg_pilot"
    output_root = (output_root or pilot_root / "review").resolve()
    audit = Provenance()
    audit.read(Path(__file__).resolve())
    audit.read(Path(qgcg.__file__).resolve())
    audit.read(Path(pilot.__file__).resolve())
    frozen = audit.json(root / FROZEN_MANIFEST)
    pins = {p.replace("\\", "/"): sha for p, sha in frozen["source_sha256"].items()}
    baseline = audit.json(root / "benchmark/qrels/core_groups.json", pins["benchmark/qrels/core_groups.json"])
    questions_path = root / "benchmark/questions.csv"
    questions = {r["query_id"]: r["query"] for r in read_rows(questions_path, audit.read(questions_path))}
    corpus_path = root / "benchmark/fair_v2/corpus.jsonl"
    corpus = {r["page_id"]: r["page_text"] for r in read_rows(corpus_path, audit.read(corpus_path))}
    pool_path = root / "benchmark/qrels/consensus_qrels_v2.csv"
    reviewed = audit.json(root / "benchmark/analysis/v2_reviewed_evaluation_manifest.json")
    pool = read_rows(pool_path, audit.read(pool_path, reviewed["files"]["benchmark/qrels/consensus_qrels_v2.csv"]["sha256"]))
    pool_sets = {arm: {qid: {r["page_id"] for r in pool if r["query_id"] == qid and r[flag].lower() in ("true", "1")}
                       for qid in (*PILOT_IDS, *STRESS_IDS)} for arm, flag in (("E1", "core"), ("E2", "relevant"))}
    sets = {}
    question_differences = []
    for qid in (*PILOT_IDS, *STRESS_IDS):
        source = next(q for q in baseline["questions"] if q["question_id"] == qid)
        if source["question"] != questions[qid]:
            question_differences.append({"query_id": qid, "baseline_question": source["question"],
                                         "clean_question": questions[qid]})
        sets[qid] = {"E0": normalize_baseline(source, corpus)}
        if {p for g in sets[qid]["E0"] for opt in g["evidence_options"] for p in opt} != pool_sets["E1"][qid]:
            raise ValueError(f"E0 membership differs from E1 pool: {qid}")
    arms = {}
    for scope, directory, ids in (("pilot", pilot_root, PILOT_IDS),
                                   ("stress", pilot_root / "stress_checks", STRESS_IDS)):
        for arm in ("E1", "E2"):
            location = directory / arm
            if not (location / "run_manifest.json").exists():
                if scope == "pilot":
                    raise FileNotFoundError(location / "run_manifest.json")
                continue
            data = load_arm(location, audit, corpus, questions)
            arms[(scope, arm)] = data
            for qid in ids:
                if qid not in data["trials"]:
                    if scope == "pilot":
                        raise ValueError(f"Pilot incomplete: {arm}/{qid}")
                    continue
                selected = set(audit.json(location / f"inputs/{qid}/page_ids.json"))
                if selected != pool_sets[arm][qid]:
                    raise ValueError(f"Archived pool differs from frozen consensus: {arm}/{qid}")
                sets[qid][arm] = normalize_result(qid, data["trials"][qid])
    runs = load_frozen_runs(root, audit, frozen)
    recalls, group_rows = diagnostic_recall(runs, sets)
    stats = [{"scope": scope, **row} for (scope, arm), data in arms.items() for row in operational_stats(data)]
    stress_state = {"questions": {qid: "; ".join(
        f'{arm}: ' + (arms[("stress", arm)]["outcomes"].get(qid, {}).get("status", "not_started")
                     if ("stress", arm) in arms else "not_started") for arm in ("E1", "E2")) for qid in STRESS_IDS}}
    preflights = [audit.json(p) for p in sorted((pilot_root / "stress_checks/preflight").glob("*.json"))]
    if preflights:
        stress_state["preflight"] = max(preflights, key=lambda x: x["timestamp_utc"])
    stress_complete = all({"E0", "E1", "E2"} <= sets[q].keys() for q in STRESS_IDS)
    stress_state["summary"] = ("All six stress trials completed; inspect verifier and human-review status separately."
                               if stress_complete else "Stress checks remain incomplete. No missing results have been simulated or replaced with a different model. See per-question status and the recorded account preflight.")
    output_root.mkdir(parents=True, exist_ok=True)
    key = blind_assignment(sets, output_root / "blind_key.json", blind_seed)
    blind_rows = make_blind_rows(sets, questions, key)
    md, report, interventions = make_reports(questions, sets, arms, stats, recalls, stress_state,
                                             {arm: {q: len(p) for q, p in queries.items()} for arm, queries in pool_sets.items()},
                                             question_differences)
    histories = {f"{scope}/{arm}": {qid: {field: result[field] for field in
                 ("initial_issues", "repair_history", "issues", "review_status", "repair_error")}
                 for qid, result in data["trials"].items()} for (scope, arm), data in arms.items()}
    outputs = {"blind_review_q2_q3.html": blind_html(blind_rows, corpus, key).encode("utf-8"),
               "blind_review_q2_q3.csv": csv_bytes(blind_rows),
               "comparison_report_q2_q3.html": report.encode("utf-8"),
               "comparison_report_q2_q3.md": md.encode("utf-8"),
               "diagnostic_recall_q2_q3.csv": csv_bytes(recalls),
               "diagnostic_group_hits_q2_q3.csv": csv_bytes(group_rows),
               "operational_stats.csv": csv_bytes(stats),
               "verifier_interventions.json": json_bytes({"interventions": interventions, "histories": histories}),
               "stress_check_status.json": json_bytes(stress_state)}
    for name, data in outputs.items():
        write_durable(output_root / name, data, replace=True)
    outputs["blind_key.json"] = (output_root / "blind_key.json").read_bytes()
    manifest = {"version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "environment": {"python": sys.version, "pydantic": package_version("pydantic")},
                "status": "complete" if stress_complete else "pilot_evaluation_complete_stress_incomplete",
                "frozen_runs": len(runs), "recall_rows": len(recalls), "blind_group_rows": len(blind_rows),
                "inference_calls_by_evaluator": 0, "retrieval_reruns": 0,
                "input_files": list(audit.inputs.values()), "verified_relocations": audit.relocations,
                "baseline_question_differences": question_differences,
                "outputs": {name: {"sha256": digest(data), "bytes": len(data)} for name, data in outputs.items()},
                "primary_metric": "required groups only; OR of nonempty AND options; unresolved excluded",
                "alternate_metric": "required + unresolved denominator; unresolved counted as misses",
                "blind_seed_location": "confidential blind_key.json only", "human_review_completed": False,
                "stress_execution": stress_state}
    write_durable(output_root / "evaluation_manifest.json", json_bytes(manifest), replace=True)
    return {k: manifest[k] for k in ("status", "frozen_runs", "recall_rows", "blind_group_rows")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=WORKSPACE / "new-V3")
    parser.add_argument("--output-root", type=Path, help="Review directory; defaults to benchmark/qgcg_pilot/review")
    parser.add_argument("--blind-seed", type=int, help="Optional reproducible shuffle seed; stored only in the secret key")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data_root, args.output_root, args.blind_seed), indent=2))


if __name__ == "__main__":
    main()
