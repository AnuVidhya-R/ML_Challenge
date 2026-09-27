# FINAL STATUS — Business Entity Resolution Challenge
# Amazon ML Challenge 2026

---

## STATUS OVERVIEW

| Component | Status |
|---|---|
| Dataset ready | ✅ Yes |
| Training pipeline | ✅ Ready (submission_03_pipeline.py, submission_04_pipeline.py) |
| Test predictions generated | ✅ Yes — 10 submission files across Day 1 and Day 2 |
| Validation | ✅ All 5 distinct submissions PASS fast validation |
| Documentation | ✅ Completed (README, methodology, experiment summary, audit) |
| Packaging | ✅ final_package/ and final_submission/ created |

---

## DATASET METRICS

| File | Rows | Size |
|---|---|---|
| Train S1 (train_source1.tsv) | 2,206,821 | 200.34 MB |
| Train S2 (train_source2.tsv) | 5,034,616 | 466.63 MB |
| Train S3 (train_source3.tsv) | 5,285,603 | 480.37 MB |
| Train GT (train_ground_truth.tsv) | 2,206,821 | 121.13 MB |
| Test S1 (test_source1.tsv) | 1,732,544 | 166.91 MB |
| Test S2 (test_source2.tsv) | 4,887,273 | 485.86 MB |
| Test S3 (test_source3.tsv) | 5,082,316 | 482.56 MB |

---

## SUBMISSIONS DETAIL

| Submission | File Exists | Row Count | Dup S1 | Non-empty | Empty | Validation | Local F0.5 | Notes |
|---|---|---|---|---|---|---|---|---|
| Day1 #1 | ✅ | 1,732,544 | 0 | 1,358,831 | 373,713 | ✅ PASS | 0.3863 (LB: 0.406) | Baseline |
| Day1 #2 | ✅ | 1,732,544 | 0 | 1,166,163 | 566,381 | ✅ PASS | 0.5394 | Threshold 0.75 |
| Day1 #3 | ✅ | 1,732,544 | 0 | 1,315,070 | 417,474 | ✅ PASS | **0.5551** | **Best validated** |
| Day1 #4 | ✅ | 1,732,544 | 0 | 1,204,501 | 528,043 | ✅ PASS | not locally scored | Margin variant |
| Day1 #5 | ✅ | 1,732,544 | 0 | 1,204,501 | 528,043 | ✅ PASS | not locally scored | Identical to Day1 #4 |
| Day2 #1 | ✅ | 1,732,544 | 0 | 955,636 | 776,908 | ✅ PASS | not locally scored | Most conservative |
| Day2 #2 | ✅ | 1,732,544 | 0 | 1,315,070 | 417,474 | ✅ PASS | not locally scored | Identical to Day1 #3 |
| Day2 #3 | ✅ | 1,732,544 | 0 | 955,636 | 776,908 | ✅ PASS | not locally scored | Identical to Day2 #1 |
| Day2 #4 | ✅ | 1,732,544 | 0 | 955,636 | 776,908 | ✅ PASS | not locally scored | Identical to Day2 #1 |
| Day2 #5 | ✅ | 1,732,544 | 0 | 955,636 | 776,908 | ✅ PASS | not locally scored | Identical to Day2 #1 |

---

## DISTINCT SUBMISSION GROUPS (by MD5 hash)

| Group | Files | Non-empty | Unique? |
|---|---|---|---|
| A | Day1 #1 | 1,358,831 | ✅ Unique |
| B | Day1 #2 | 1,166,163 | ✅ Unique |
| C | Day1 #3, Day2 #2 | 1,315,070 | ✅ Unique prediction set |
| D | Day1 #4, Day1 #5 | 1,204,501 | ✅ Unique prediction set |
| E | Day2 #1, Day2 #3, Day2 #4, Day2 #5 | 955,636 | ✅ Unique prediction set |

**Only 5 genuinely distinct prediction sets exist across all 10 submission files.**

---

## BEST SUBMISSION

**Day1 #3** (`submissions/day1/submission_03.tsv`)  
- Local F0.5 = **0.5551**  
- Local Precision = 0.8598  
- Local Recall = 0.4526  
- Threshold: 0.85 (unconditional) / 0.75 (margin-conditional)
- Non-empty predictions: 1,315,070  
- Empty predictions: 417,474  
- Validation: ✅ PASS

---

## CANDIDATE PAIRS FILE

`output/candidate_pairs.tsv`  
- Rows: 1,732,544 ✅  
- Non-empty: 1,445,059  
- Empty: 287,485  
- Validation: ✅ PASS

---

## FINAL PACKAGE

| Path | Contents |
|---|---|
| `final_package/` | Reproducible source code package |
| `final_package/source_code/` | Core pipeline scripts |
| `final_package/requirements.txt` | Python dependencies |
| `final_package/README.md` | Run instructions |
| `final_package/methodology.md` | Technical methodology |

---

## FINAL SUBMISSION PACKAGE

| Path | Contents |
|---|---|
| `final_submission/` | Submission-ready package |
| `final_submission/matching_results.tsv` | Final predictions (Day1 #3) |
| `final_submission/candidate_pairs.tsv` | Candidate pairs from output/ |
| `final_submission/source_code/` | Core pipeline scripts |
| `final_submission/requirements.txt` | Dependencies |
| `final_submission/README.md` | Methodology overview |
| `final_submission/methodology.md` | Detailed methodology |

---

## IMPORTANT NOTES

1. Day1 #5 is byte-identical to Day1 #4 — treat as the same submission.
2. Day2 #2 is byte-identical to Day1 #3 — same predictions.
3. Day2 #3, #4, #5 are all byte-identical to Day2 #1.
4. Only Day1 #1 has a confirmed leaderboard score (0.406).
5. Do NOT use graph transitive closure for matches.
6. No external business databases were used.

---

## VALIDATION TOOL

```bash
# Fast validation (recommended — handles large files with chunking)
python validate_fast.py submissions/day1/submission_03.tsv

# Original validation
python validate_local.py
```
