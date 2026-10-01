"""Consolidate provisional core pages into evidence-facet groups."""
from __future__ import annotations
import argparse, csv, json, logging, os, re, tempfile, time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env.local", override=False)
LOG = logging.getLogger("consolidate_core_groups")
JUDGES = ("judge_a", "judge_b", "judge_c")
PROMPT_VERSION = "core-group-consolidation-v1-2026-09-30"
CONSOLIDATION_PROMPT = """You are consolidating provisional core-evidence pages for ONE benchmark question.

Inputs:
- The full question
- A list of provisional core pages
- For each page: PAGE_ID, document title, judge-assigned core_facet labels, and short rationales

Goal:
Create a set of core evidence groups representing the distinct indispensable information components needed for a complete, high-quality answer.

Rules:
1. Merge pages into the same group when they provide substantially the same critical fact, conclusion, relationship, or answer component. These are redundant/alternative evidence instances.
2. Keep pages in separate groups when they support different indispensable components of the question.
3. Do not create groups for merely helpful context. Only create groups for evidence that is truly core.
4. A group may contain one page or multiple equivalent pages.
5. Do not require every page in a group to be retrieved. Any one valid member should count as covering that group.
6. Preserve uncertainty: if two pages may or may not be redundant, flag the group for manual review rather than forcing a merge.

Return:
- group_id
- facet_description
- member_page_ids
- why_this_facet_is_core
- redundancy_confidence (high/medium/low)"""

class ConsolidationError(RuntimeError):
    pass

def natural_key(value):
    return tuple((0,int(x)) if x.isdigit() else (1,x.casefold())
                 for x in re.split(r"(\d+)",value) if x)

def questions(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    result={(r["query_id"]).strip():(r["query"]).strip() for r in rows}
    if any(not k or not v for k,v in result.items()):
        raise ConsolidationError("Blank question field")
    return result

def titles(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        key="document_id" if "document_id" in (reader.fieldnames or []) else "doc_id"
        return {(r.get(key) or "").strip():(r.get("title") or "").strip() for r in reader}

def provisional(path):
    result={}
    with path.open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            grades=[int(row[f"r{i}"]) for i in (1,2,3)]
            if float(row["mean_grade"])>=3.5 and sum(x==4 for x in grades)>=2:
                result.setdefault(row["query_id"],[]).append(row["page_id"])
    return {q:sorted(ids) for q,ids in result.items()}

def contexts(directory):
    result={}
    for judge in JUDGES:
        path=directory/f"{judge}.jsonl"
        if not path.is_file():
            raise ConsolidationError(f"Missing {path}")
        for number,line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(),1):
            if not line.strip(): continue
            row=json.loads(line)
            qid=row.get("question_id",row.get("query_id")); pid=row.get("page_id")
            if not isinstance(qid,str) or not isinstance(pid,str):
                raise ConsolidationError(f"{path}:{number}: bad IDs")
            item=result.setdefault((qid,pid),{"core_facet_labels":[],"rationales":[]})
            facet=row.get("core_facet"); rationale=row.get("rationale")
            if isinstance(facet,str) and facet.strip(): item["core_facet_labels"].append(facet.strip())
            if isinstance(rationale,str) and rationale.strip(): item["rationales"].append(rationale.strip())
    for item in result.values():
        item["core_facet_labels"]=list(dict.fromkeys(item["core_facet_labels"]))
        item["rationales"]=list(dict.fromkeys(item["rationales"]))
    return result

def extract(raw):
    cleaned=raw.strip(); values=[]
    try: values.append(json.loads(cleaned))
    except json.JSONDecodeError:
        decoder=json.JSONDecoder()
        for index,char in enumerate(cleaned):
            if char not in "[{": continue
            try:
                value,_=decoder.raw_decode(cleaned[index:]); values.append(value); break
            except json.JSONDecodeError: pass
    for value in values:
        if isinstance(value,dict) and isinstance(value.get("groups"),list): return value["groups"]
        if isinstance(value,list): return value
    raise ConsolidationError("No JSON groups array")

def normalize(qid,raw_groups,allowed):
    groups=[]; used=set()
    for index,raw in enumerate(raw_groups,1):
        if not isinstance(raw,dict): raise ConsolidationError("Group is not an object")
        facet=raw.get("facet_description"); why=raw.get("why_this_facet_is_core")
        members=raw.get("member_page_ids"); confidence=raw.get("redundancy_confidence")
        if not isinstance(facet,str) or not facet.strip(): raise ConsolidationError("Bad facet")
        if not isinstance(why,str) or not why.strip(): raise ConsolidationError("Bad core reason")
        if confidence not in {"high","medium","low"}: raise ConsolidationError("Bad confidence")
        if not isinstance(members,list) or not members: raise ConsolidationError("Empty members")
        accepted=[]
        for pid in members:
            if not isinstance(pid,str) or pid not in allowed: raise ConsolidationError(f"Unknown page {pid!r}")
            if pid in used: raise ConsolidationError(f"Duplicate assignment {pid}")
            used.add(pid); accepted.append(pid)
        groups.append({
            "group_id":f"{qid}_G{index:02d}",
            "facet_description":facet.strip(),
            "member_page_ids":sorted(accepted),
            "why_this_facet_is_core":why.strip(),
            "redundancy_confidence":confidence,
            "manual_review_required":confidence=="low",
        })
    return groups,sorted(allowed-used)

def call(model,prompt,timeout):
    import litellm
    options={"reasoning_effort":"none"} if model.endswith("gpt-5.6-sol") else {}
    response=litellm.completion(
        model=model,
        messages=[{"role":"system","content":CONSOLIDATION_PROMPT},
                  {"role":"user","content":prompt}],
        max_tokens=8192,timeout=timeout,num_retries=0,**options)
    content=response.choices[0].message.content
    if not isinstance(content,str): raise ConsolidationError("Non-text response")
    return content

def consolidate(qid,question,page_ids,title_map,context_map,model,timeout,retries):
    pages=[]
    for pid in page_ids:
        context=context_map.get((qid,pid))
        if context is None: raise ConsolidationError(f"Missing context {qid}/{pid}")
        pages.append({"page_id":pid,"document_title":title_map.get(pid.split("_",1)[0],""),
                      **context})
    payload={"question_id":qid,"question":question,"provisional_core_pages":pages}
    prompt="INPUT JSON:\n"+json.dumps(payload,ensure_ascii=False,indent=2)+(
        "\n\nReturn one JSON object with a single key named groups. "
        "The groups array must use exactly the requested fields.")
    last=None
    for attempt in range(1,retries+1):
        retry_prompt = prompt
        if last is not None:
            retry_prompt += (
                "\n\nVALIDATION CORRECTION REQUIRED:\n"
                f"The previous output failed validation: {last}\n"
                "Return a completely corrected JSON object. Each page_id may appear in "
                "AT MOST ONE group. Do not repeat a page across groups. Use only page_ids "
                "listed in INPUT JSON."
            )
        try:
            groups,unassigned=normalize(
                qid,extract(call(model,retry_prompt,timeout)),set(page_ids)
            )
            flags=[]
            if unassigned: flags.append("unassigned_provisional_core_pages")
            if any(x["manual_review_required"] for x in groups):
                flags.append("low_redundancy_confidence")
            return {"question_id":qid,"question":question,
                    "provisional_core_page_count":len(page_ids),"group_count":len(groups),
                    "groups":groups,"unassigned_page_ids":unassigned,
                    "automatic_review_flags":flags,
                    # Section 5.1 requires human inspection for all nine pilot questions.
                    "manual_review_required":True}
        except Exception as error:
            last=error
            if attempt<retries:
                LOG.warning(
                    "%s consolidation attempt %d/%d failed validation: %s",
                    qid,attempt,retries,error,
                )
                time.sleep(2**(attempt-1))
    raise ConsolidationError(f"{qid} failed after {retries} attempts: {last}") from last

def write_atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    handle,name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent,text=True)
    try:
        with os.fdopen(handle,"w",encoding="utf-8",newline="\n") as f:
            json.dump(payload,f,ensure_ascii=False,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name,path)
    except BaseException:
        Path(name).unlink(missing_ok=True); raise

def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qrels",type=Path,default=ROOT/"benchmark/qrels/consensus_qrels_v2.csv")
    p.add_argument("--judgments",type=Path,default=ROOT/"benchmark/judgments_v2")
    p.add_argument("--questions",type=Path,default=ROOT/"benchmark/questions.csv")
    p.add_argument("--catalog",type=Path,default=ROOT/"data/source_catalog.csv")
    p.add_argument("--output",type=Path,default=ROOT/"benchmark/qrels/core_groups.json")
    p.add_argument("--model",default=os.getenv("CORE_GROUP_MODEL","openai/gpt-5.6-sol"))
    p.add_argument("--timeout",type=float,default=180); p.add_argument("--retries",type=int,default=5)
    p.add_argument("--log-level",default="INFO",choices=("DEBUG","INFO","WARNING","ERROR"))
    return p.parse_args()

def main():
    a=args(); logging.basicConfig(level=getattr(logging,a.log_level),format="%(levelname)s: %(message)s")
    try:
        qs=questions(a.questions); ts=titles(a.catalog); core=provisional(a.qrels); ctx=contexts(a.judgments)
        records=[]
        for qid in sorted(qs,key=natural_key):
            ids=core.get(qid,[]); LOG.info("%s: %d provisional core pages",qid,len(ids))
            records.append(consolidate(qid,qs[qid],ids,ts,ctx,a.model,a.timeout,a.retries)
                           if ids else {"question_id":qid,"question":qs[qid],
                           "provisional_core_page_count":0,"group_count":0,"groups":[],
                           "unassigned_page_ids":[],
                           "automatic_review_flags":["no_provisional_core_pages"],
                           "manual_review_required":True})
        payload={"schema_version":1,"prompt_version":PROMPT_VERSION,"model":a.model,
                 "qrels_source":str(a.qrels),
                 "manual_review_status":"pending_peizhao_songxi_review_for_all_nine_questions",
                 "questions":records}
        write_atomic(a.output,payload)
        LOG.info("Wrote %d questions and %d groups",len(records),sum(x["group_count"] for x in records))
        return 0
    except (OSError,ValueError,json.JSONDecodeError,ConsolidationError) as error:
        LOG.error("%s",error); return 1

if __name__=="__main__":
    raise SystemExit(main())