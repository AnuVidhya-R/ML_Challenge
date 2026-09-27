# ML Challenge — Business Entity Resolution

**Amazon ML Challenge 2026 | Business Entity Resolution**

## Task

For every Source 1 business entity, predict its matching entity ID(s) from Source 2 and Source 3 across three noisy TSV data sources.

Evaluation: **Macro F0.5** (precision-weighted).

## Best Result

| Submission | Local F0.5 | Precision | Recall | Threshold |
|---|---|---|---|---|
| Day1 #3 | **0.5551** | 0.8598 | 0.4526 | 0.85 |
| Day1 #2 | 0.5394 | 0.9267 | 0.3908 | 0.75 |
| Day1 #1 (baseline) | 0.3863 | 0.4778 | 0.3144 | exact match |

Confirmed leaderboard score for Day1 #1: **0.406**

## Approach

1. **Text Normalization** — legal suffix removal, address abbreviation expansion, phone/domain extraction
2. **Candidate Blocking** — name+country and address+country block keys with pair explosion guards
3. **RapidFuzz Scoring** — composite name (60%) + address (40%) score with penalties for street-number mismatch and phone/domain conflicts
4. **Precision-First Threshold** — dual threshold (0.85 unconditional / 0.75 margin-based) to maximize F0.5

## Project Structure

```
business_entity_resolution/
├── submission_03_pipeline.py   # Best validated pipeline (F0.5=0.5551)
├── submission_04_pipeline.py   # Margin-threshold pipeline
├── validate_fast.py            # Fast chunked validator
├── validate_local.py           # Original submission validator
├── submissions/day1/           # Day 1 submission TSVs
├── submissions/day2/           # Day 2 submission TSVs
├── reports/                    # Full documentation
│   ├── README_FINAL.md
│   ├── methodology.md
│   ├── experiment_summary.md
│   ├── final_submission_audit.txt
│   └── FINAL_STATUS.md
├── final_package/              # Reproducible code package
└── final_submission/           # Final submission package (code + docs)
```

> **Note:** Dataset files are excluded from this repository (too large).  
> Download from the official Amazon ML Challenge portal.

## Requirements

```bash
pip install pandas numpy rapidfuzz
```

## Run

```bash
python business_entity_resolution/submission_03_pipeline.py
python business_entity_resolution/validate_fast.py submissions/day1/submission_03.tsv
```

See [`reports/methodology.md`](business_entity_resolution/reports/methodology.md) for the full technical description.
