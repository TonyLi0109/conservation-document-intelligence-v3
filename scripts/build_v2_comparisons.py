"""Build Section 10 side-by-side V1/V2 qrels, metric, and group-correction reports."""
from __future__ import annotations
import argparse,csv,json,os,tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]

def atomic(path,fields,rows):
 path.parent.mkdir(parents=True,exist_ok=True); fd,name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
 try:
  with os.fdopen(fd,"w",encoding="utf-8-sig",newline="") as f:
   w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
  os.replace(name,path)
 except BaseException: Path(name).unlink(missing_ok=True); raise

def qrels(path):
 with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def compare_qrels(old_path,new_path,out):
 old,new=qrels(old_path),qrels(new_path); rows=[]
 for qid in sorted({r["query_id"] for r in old+new},key=lambda x:int(x[1:])):
  def selected(data,key):return {r["page_id"] for r in data if r["query_id"]==qid and r[key].lower()=="true"}
  oc,nc=selected(old,"core"),selected(new,"core"); or_,nr=selected(old,"relevant"),selected(new,"relevant")
  jac=lambda a,b:len(a&b)/len(a|b) if a|b else 1.0
  rows.append({"query_id":qid,"v1_relevant":len(or_),"v2_relevant":len(nr),"relevant_jaccard":f"{jac(or_,nr):.6f}",
   "v1_core":len(oc),"v2_core":len(nc),"core_jaccard":f"{jac(oc,nc):.6f}",
   "core_added_v2":";".join(sorted(nc-oc)),"core_removed_v2":";".join(sorted(oc-nc))})
 atomic(out,list(rows[0]),rows)

def macro_eval(path):
 with path.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
 metrics=["recall_at_20_relevant","recall_at_20_core","ndcg_at_20","reciprocal_rank"]
 by=defaultdict(list)
 for r in rows:
  for m in metrics:
   if r.get(m) not in ("",None): by[(r["system"],m)].append(float(r[m]))
 return {(s,m):sum(v)/len(v) for (s,m),v in by.items()}

def compare_metrics(old_path,new_path,out):
 old,new=macro_eval(old_path),macro_eval(new_path); rows=[]
 for system in sorted({s for s,_ in old}|{s for s,_ in new}):
  row={"system":system}
  for metric,label in [("recall_at_20_relevant","relevant_recall_at_20"),("recall_at_20_core","page_core_recall_at_20"),("ndcg_at_20","ndcg_at_20"),("reciprocal_rank","mrr")]:
   a,b=old.get((system,metric)),new.get((system,metric)); row[f"v1_{label}"]="" if a is None else f"{a:.6f}"; row[f"v2_{label}"]="" if b is None else f"{b:.6f}"; row[f"delta_{label}"]="" if a is None or b is None else f"{b-a:.6f}"
  rows.append(row)
 atomic(out,list(rows[0]),rows)

def load_old_groups(path):
 result=defaultdict(list)
 with path.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f): result[r["question_id"]].append({"group_id":r["group_id"],"facet_description":r["facet_description"],"members":set(filter(None,r["member_page_ids"].split(";")))})
 return result

def load_new_groups(path):
 d=json.loads(path.read_text(encoding="utf-8-sig")); result=defaultdict(list)
 for q in d["questions"]:
  for g in q["groups"]:result[q["question_id"]].append({"group_id":g["group_id"],"facet_description":g["facet_description"],"members":set(g["member_page_ids"])})
 return result

def compare_groups(old_path,new_path,out):
 old,new=load_old_groups(old_path),load_new_groups(new_path); rows=[]; matched=set()
 for qid in sorted(set(old)|set(new),key=lambda x:int(x[1:])):
  for a in old[qid]:
   candidates=[(len(a["members"]&b["members"])/len(a["members"]|b["members"]),b) for b in new[qid]]
   score,b=max(candidates,key=lambda x:x[0]) if candidates else (0,{})
   if b:matched.add((qid,b["group_id"]))
   status="UNCHANGED" if score==1 and a["facet_description"]==b.get("facet_description") else ("REMOVED" if score==0 else "MODIFIED")
   rows.append({"query_id":qid,"v1_group_id":a["group_id"],"v2_group_id":b.get("group_id",""),"member_jaccard":f"{score:.6f}","correction_status":status,"v1_facet":a["facet_description"],"v2_facet":b.get("facet_description","")})
  for b in new[qid]:
   if (qid,b["group_id"]) not in matched:rows.append({"query_id":qid,"v1_group_id":"","v2_group_id":b["group_id"],"member_jaccard":"0.000000","correction_status":"ADDED","v1_facet":"","v2_facet":b["facet_description"]})
 atomic(out,list(rows[0]),rows)

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir",type=Path,default=ROOT/"benchmark/analysis/v2_comparisons"); a=p.parse_args()
 compare_qrels(ROOT/"benchmark/baselines/core_evidence_v1_2026-09-30/consensus_qrels_full.csv",ROOT/"benchmark/qrels/consensus_qrels_v2.csv",a.output_dir/"qrels_membership_comparison.csv")
 compare_metrics(ROOT/"benchmark/baselines/core_evidence_v1_2026-09-30/system_evaluation.csv",ROOT/"benchmark/analysis/system_evaluation_v2_reviewed.csv",a.output_dir/"system_metric_comparison.csv")
 compare_groups(ROOT/"benchmark/analysis/core_group_manual_review_v2.csv",ROOT/"benchmark/qrels/core_groups.json",a.output_dir/"core_group_corrections.csv")
 print(a.output_dir); return 0
if __name__=="__main__":raise SystemExit(main())
