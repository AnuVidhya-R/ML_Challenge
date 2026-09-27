# Experiment Summary — Business Entity Resolution

## Evaluation Metric: Macro F0.5

> F0.5 weights precision 2.5× more than recall.  
> A false positive (wrong match) is more expensive than a false negative (missed match).

---

## Experiment Table

| Submission | Method | Local Precision | Local Recall | Local F0.5 | Threshold | Leaderboard | Notes |
|---|---|---|---|---|---|---|---|
| Day1 #1 | Fast Alphanumeric (name+country, addr+country exact match) | 0.4778 | 0.3144 | 0.3863 | N/A (exact match) | **0.406** | Baseline; high recall bias; too many false positives |
| Day1 #2 | Vectorized blocking + RapidFuzz scoring | 0.9267 | 0.3908 | 0.5394 | 0.75 | not submitted | Best precision among single-threshold runs |
| Day1 #3 | Multi-signal pipeline (broader blocking, margin decision) | 0.8598 | 0.4526 | **0.5551** | 0.85 | not submitted | **Best locally validated F0.5** |
| Day1 #4 | Precision-first margin threshold pipeline | not locally scored | — | — | 0.85 (margin) | not submitted | Same architecture as #3, margin-filtered |
| Day1 #5 | Identical to Day1 #4 | not locally scored | — | — | — | not submitted | File hash confirms identical to #4 |
| Day2 #1 | Tighter precision-first variant | not locally scored | — | — | — | not submitted | 955,636 non-empty predictions (most conservative) |
| Day2 #2 | Same as Day1 #3 | not locally scored | — | — | — | not submitted | File hash confirms identical to Day1 #3 |
| Day2 #3 | Same as Day2 #1 | not locally scored | — | — | — | not submitted | File hash confirms identical to Day2 #1 |
| Day2 #4 | Same as Day2 #1 | not locally scored | — | — | — | not submitted | File hash confirms identical to Day2 #1 |
| Day2 #5 | Same as Day2 #1 | not locally scored | — | — | — | not submitted | File hash confirms identical to Day2 #1 |

---

## Distinct Prediction Files (by MD5 Hash)

| Group | Files | Non-empty Predictions | Notes |
|---|---|---|---|
| A (unique) | Day1 #1 | 1,358,831 | Aggressively broad baseline |
| B (unique) | Day1 #2 | 1,166,163 | RapidFuzz threshold=0.75 |
| C | Day1 #3 = Day2 #2 | 1,315,070 | Best local F0.5 = 0.5551 |
| D | Day1 #4 = Day1 #5 | 1,204,501 | Margin-threshold variant |
| E | Day2 #1 = Day2 #3 = Day2 #4 = Day2 #5 | 955,636 | Most conservative (fewest predictions) |

---

## Singleton Statistics (Training Set — from experiments/experiment_log.md)

| Metric | Day1 #1 |
|---|---|
| Total Singletons | 123,247 |
| Correctly Predicted Singletons | 73,542 |
| Incorrectly Matched Singletons | 49,705 |

---

## Key Design Observations

- **Higher threshold → fewer predictions → better precision → higher F0.5** in this task
- Day2 #1 group (955,636 non-empty) is most conservative but unvalidated locally
- Day1 #3 (threshold 0.85, 1,315,070 non-empty) achieves best verified local score
- Leaderboard score for Day1 #1 (0.406) confirms the local scoring mechanism is meaningful

---

## Recommendation for Final Submission

**Best validated**: Day1 #3 (= Day2 #2) with local F0.5 = **0.5551**  
**Most conservative**: Day2 #1 group with 955,636 non-empty predictions (unvalidated locally)
