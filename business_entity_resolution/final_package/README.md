# Business Entity Resolution — Reproducible Code Package

## Overview

This package contains the exact code used to generate the final submissions for the Amazon ML Challenge 2026 Business Entity Resolution task.

**Best validated submission: F0.5 = 0.5551** (Day 1, Submission #3)

---

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies:
- `pandas >= 1.5.0`
- `numpy >= 1.23.0`
- `rapidfuzz >= 3.0.0`

---

## Dataset Setup

Place the dataset files at the following paths relative to the project root:

```
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

---

## Running the Pipeline

### Best Validated Method (Submission #3 — F0.5 = 0.5551)

```bash
python source_code/submission_03_pipeline.py
```

Output files:
- `output/matching_results.tsv`
- `submissions/day1/submission_03.tsv`

### Precision-First Margin Method (Submission #4)

```bash
python source_code/submission_04_pipeline.py
```

### Validate Output

```bash
# Fast chunked validator
python source_code/validate_fast.py output/matching_results.tsv
```

---

## Key Configuration Parameters

| Parameter | Value | Description |
|---|---|---|
| `CHUNK_SIZE` | 500,000 | Rows per chunk for target sources |
| `MAX_BLOCK_PAIRS` | 50 | Max cross-product pairs per block key |
| `MAX_TARGET_IN_KEY` | 10 | Max target entities per block key |
| `tau_high` | 0.85 | Unconditional accept threshold |
| `tau_low` | 0.75 | Conditional accept threshold |
| `delta_margin` | 0.05 | Min score margin for conditional accept |
| Name weight | 0.6 | In composite score formula |
| Address weight | 0.4 | In composite score formula |

---

## Methodology

See `methodology.md` for the full technical description.
