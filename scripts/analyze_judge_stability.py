"""Compare a repeated three-judge V2 run with the frozen reviewed baseline."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
JUDGES = ("judge_a", "judge_b", "judge_c")


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    import os, tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as out:
            w = csv.DictWriter(out, fieldnames=fields); w.writeheader(); w.writerows(rows)
            out.flush(); os.fsync(out.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True); raise


def load_judgments(directory: Path, judge: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    with (directory/f"{judge}.jsonl").open(encoding="utf-8-sig") as source:
        for line in source:
            if line.strip():
                row=json.loads(line); key=(row.get("question_id",row.get("query_id")),row["page_id"])
                rows[key]=row
    return rows


def load_qrels(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with path.open(encoding="utf-8-sig",newline="") as source:
        return {(r["query_id"],r["page_id"]):r for r in csv.DictReader(source)}


def load_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload=json.loads(path.read_text(encoding="utf-8-sig")); result={}
    for q in payload["questions"]:
        result[q["question_id"]]=q["groups"]
    return result


def jaccard(a:set[Any],b:set[Any])->float:
    return len(a&b)/len(a|b) if a|b else 1.0


def same_group_pairs(groups:list[dict[str,Any]])->set[tuple[str,str]]:
    result=set()
    for group in groups:
        members=sorted(set(group["member_page_ids"]))
        result.update(itertools.combinations(members,2))
    return result


def quadratic_weighted_kappa(a:list[int],b:list[int],n:int=5)->float:
    if len(a)!=len(b) or not a: return 0.0
    observed=[[0]*n for _ in range(n)]
    ca=[0]*n; cb=[0]*n
    for x,y in zip(a,b): observed[x][y]+=1; ca[x]+=1; cb[y]+=1
    total=len(a); num=den=0.0
    for i in range(n):
        for j in range(n):
            weight=((i-j)/(n-1))**2
            num += weight*observed[i][j]/total
            den += weight*(ca[i]*cb[j])/(total*total)
    return 1.0-num/den if den else 1.0


def analyze(base_judgments:Path,repeat_judgments:Path,base_qrels:Path,repeat_qrels:Path,
            base_groups:Path,repeat_groups:Path,output_dir:Path)->None:
    judge_rows=[]
    for judge in JUDGES:
        old=load_judgments(base_judgments,judge); new=load_judgments(repeat_judgments,judge)
        if set(old)!=set(new): raise ValueError(f"{judge} key sets differ")
        keys=sorted(old)
        a=[int(old[k]["grade"]) for k in keys]; b=[int(new[k]["grade"]) for k in keys]
        facet_keys=[k for k in keys if old[k].get("core_facet") or new[k].get("core_facet")]
        facet_exact=sum(
            (old[k].get("core_facet") or "").strip().casefold()==
            (new[k].get("core_facet") or "").strip().casefold() for k in facet_keys
        )
        redundancy_keys=[k for k in keys if old[k].get("redundancy_group_hint") or new[k].get("redundancy_group_hint")]
        redundancy_exact=sum(
            (old[k].get("redundancy_group_hint") or "").strip().casefold()==
            (new[k].get("redundancy_group_hint") or "").strip().casefold() for k in redundancy_keys
        )
        judge_rows.append({
            "judge":judge,"pair_count":len(keys),
            "grade_exact_agreement":f"{sum(x==y for x,y in zip(a,b))/len(a):.6f}",
            "grade_mean_absolute_error":f"{sum(abs(x-y) for x,y in zip(a,b))/len(a):.6f}",
            "quadratic_weighted_kappa":f"{quadratic_weighted_kappa(a,b):.6f}",
            "grade4_membership_jaccard":f"{jaccard({k for k in keys if old[k]['grade']==4},{k for k in keys if new[k]['grade']==4}):.6f}",
            "core_facet_exact_label_agreement":"" if not facet_keys else f"{facet_exact/len(facet_keys):.6f}",
            "redundancy_hint_exact_label_agreement":"" if not redundancy_keys else f"{redundancy_exact/len(redundancy_keys):.6f}",
        })
    atomic_csv(output_dir/"judge_grade_stability.csv",list(judge_rows[0]),judge_rows)

    oldq=load_qrels(base_qrels); newq=load_qrels(repeat_qrels)
    qids=sorted({k[0] for k in oldq}|{k[0] for k in newq},key=lambda x:int(x[1:]))
    consensus=[]
    for qid in qids+["ALL"]:
        filt=lambda data,field:{k for k,v in data.items() if (qid=="ALL" or k[0]==qid) and v[field].lower()=="true"}
        oc,nc=filt(oldq,"core"),filt(newq,"core")
        or_,nr=filt(oldq,"relevant"),filt(newq,"relevant")
        consensus.append({
            "query_id":qid,"baseline_core_pages":len(oc),"repeat_core_pages":len(nc),
            "core_page_jaccard":f"{jaccard(oc,nc):.6f}",
            "baseline_relevant_pages":len(or_),"repeat_relevant_pages":len(nr),
            "relevant_page_jaccard":f"{jaccard(or_,nr):.6f}",
        })
    atomic_csv(output_dir/"consensus_membership_stability.csv",list(consensus[0]),consensus)

    oldg,newg=load_groups(base_groups),load_groups(repeat_groups)
    group_rows=[]; redundancy_rows=[]
    for qid in sorted(set(oldg)|set(newg),key=lambda x:int(x[1:])):
        baseline=oldg.get(qid,[]); repeat=newg.get(qid,[])
        for group in baseline:
            members=set(group["member_page_ids"])
            candidates=[(jaccard(members,set(other["member_page_ids"])),other) for other in repeat]
            score,best=max(candidates,key=lambda x:x[0]) if candidates else (0.0,{})
            group_rows.append({
                "query_id":qid,"baseline_group_id":group["group_id"],
                "best_repeat_group_id":best.get("group_id",""),
                "member_set_jaccard":f"{score:.6f}",
                "baseline_facet":group["facet_description"],
                "repeat_facet":best.get("facet_description",""),
            })
        a,b=same_group_pairs(baseline),same_group_pairs(repeat)
        common_pages={p for g in baseline for p in g["member_page_ids"]}&{p for g in repeat for p in g["member_page_ids"]}
        a={pair for pair in a if set(pair)<=common_pages}; b={pair for pair in b if set(pair)<=common_pages}
        precision=len(a&b)/len(b) if b else (1.0 if not a else 0.0)
        recall=len(a&b)/len(a) if a else (1.0 if not b else 0.0)
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        redundancy_rows.append({
            "query_id":qid,"common_core_pages":len(common_pages),
            "baseline_copairs":len(a),"repeat_copairs":len(b),
            "copair_precision":f"{precision:.6f}","copair_recall":f"{recall:.6f}",
            "copair_f1":f"{f1:.6f}",
        })
    atomic_csv(output_dir/"core_group_matching_stability.csv",list(group_rows[0]),group_rows)
    atomic_csv(output_dir/"redundancy_group_stability.csv",list(redundancy_rows[0]),redundancy_rows)


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-judgments",type=Path,default=ROOT/"benchmark/judgments_v2")
    p.add_argument("--repeat-judgments",type=Path,default=ROOT/"benchmark/judgments_v2_repeat")
    p.add_argument("--baseline-qrels",type=Path,default=ROOT/"benchmark/qrels/consensus_qrels_v2.csv")
    p.add_argument("--repeat-qrels",type=Path,default=ROOT/"benchmark/qrels/consensus_qrels_v2_repeat.csv")
    p.add_argument("--baseline-groups",type=Path,default=ROOT/"benchmark/qrels/core_groups.json")
    p.add_argument("--repeat-groups",type=Path,default=ROOT/"benchmark/qrels/core_groups_v2_repeat.json")
    p.add_argument("--output-dir",type=Path,default=ROOT/"benchmark/analysis/judge_stability")
    a=p.parse_args()
    try: analyze(a.baseline_judgments,a.repeat_judgments,a.baseline_qrels,a.repeat_qrels,a.baseline_groups,a.repeat_groups,a.output_dir)
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}"); return 1
    print(f"Judge stability: {a.output_dir}"); return 0

if __name__=="__main__": raise SystemExit(main())
