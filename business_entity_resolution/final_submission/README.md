# Final Submission Package — Business Entity Resolution

## Final Selected Submission

**`matching_results.tsv`** — Day 1, Submission #3  
- Best locally validated submission  
- Local F0.5 = **0.5551**, Precision = 0.8598, Recall = 0.4526  
- Threshold: 0.85 (unconditional) / 0.75 with 0.05 margin  
- 1,315,070 non-empty predictions, 417,474 empty predictions  
- All 1,732,544 Source 1 test entities included, 0 duplicates  
- Validation: ✅ PASS

## Candidate Pairs

**`candidate_pairs.tsv`** — Generated from output/ by the pipeline  
- 1,445,059 non-empty predictions  
- Validation: ✅ PASS

## Source Code

See `source_code/` for the pipeline used to generate these results.

## Reproducing Results

```bash
pip install -r requirements.txt
python source_code/submission_03_pipeline.py
python source_code/validate_fast.py matching_results.tsv
```

## Methodology

See `methodology.md` for the detailed technical description of the approach.
