# CDIRP Code Walkthrough Checklist

Target duration: 15-20 minutes
Presenter: Peizhao or Songxi
Audience: Junshan

## Before the meeting

- [ ] Confirm Junshan can clone/read the repository and create a branch.
- [ ] Confirm `.env.local` and credentials are excluded.
- [ ] Open `docs/experiment_guide.md`.
- [ ] Open the frozen pool, judge prompt, baseline qrels, reviewed groups, and evaluation CSV.
- [ ] Do not run paid API calls during the walkthrough unless explicitly planned.

## Walkthrough sequence

### 0-3 minutes: repository and frozen inputs

- [ ] Show `benchmark/questions.csv`.
- [ ] Show `benchmark/corpus_pages.jsonl` and canonical page IDs.
- [ ] Show `benchmark/pools/candidate_pool_v2_761.jsonl`.
- [ ] Explain that the replication unit is 761 fixed query-page pairs.

### 3-8 minutes: judging

- [ ] Open `scripts/run_llm_judges.py`.
- [ ] Show `JUDGING_PROMPT_VERSION` and `JUDGE_SYSTEM_PROMPT`.
- [ ] Show the three model IDs.
- [ ] Explain blind inputs: question, page ID, title, and page text.
- [ ] Explain parsing, retries, throttling, checkpoints, and `--resume`.
- [ ] Show raw baseline files in `benchmark/judgments_v2/`.

### 8-12 minutes: consensus and groups

- [ ] Open `scripts/build_consensus_qrels.py`.
- [ ] Explain MeanRel, range, variance, relevant threshold, and core threshold.
- [ ] Show `benchmark/qrels/consensus_qrels_v2.csv`.
- [ ] Open `scripts/consolidate_core_groups.py`.
- [ ] Explain page-level core versus evidence-facet core groups.
- [ ] Show reviewed `benchmark/qrels/core_groups.json`.

### 12-15 minutes: evaluation and stability

- [ ] Open `scripts/evaluate_systems.py`.
- [ ] Explain Relevant Recall@20, page CoreRecall@20, CoreGroupRecall@20, nDCG@20, and MRR.
- [ ] Show `benchmark/analysis/system_evaluation_v2_reviewed.csv`.
- [ ] Open `scripts/analyze_judge_stability.py`.
- [ ] Explain exact agreement, MAE, MeanRel correlation, and Jaccard overlap.

### 15-18 minutes: independent run

- [ ] Create or identify `junshan/stability-facet-analysis`.
- [ ] Show the run-specific convention under `benchmark/runs/<run_id>/`.
- [ ] Review exact commands in `docs/experiment_guide.md`.
- [ ] Confirm the baseline must not be overwritten.

### 18-20 minutes: assignment and reporting

- [ ] Review Q3, Q7, and Q9 facet requirements.
- [ ] Review redundancy-group rules.
- [ ] Show `docs/WEEKLY_PROGRESS_REPORT_TEMPLATE.md`.
- [ ] Confirm report deadline and next review owner.

## Completion record

- Date:
- Presenter:
- Attendees:
- Branch created:
- Run ID reserved:
- Questions or decisions:
