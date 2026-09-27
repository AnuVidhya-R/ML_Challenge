# Technical Methodology: Business Entity Resolution Pipeline

## Overview

This methodology describes the exact implementation used in the final matched submissions for the Amazon ML Challenge 2026 — Business Entity Resolution task. The goal is to link each Source 1 business entity record to its corresponding record(s) in Source 2 and Source 3.

The evaluation metric is **macro F0.5** (precision-weighted F-score), making **precision more valuable than recall**.

---

## 1. Data Loading

All three source files (source1, source2, source3) are TSV files read with `pandas.read_csv` using:
- `sep='\t'`
- `quoting=3` (QUOTE_NONE — prevents misparse on business names with quotes)
- `dtype=str` (all fields treated as strings for safety)

Large target sources (source2, source3) are processed in **chunks of 500,000 rows** to avoid memory exhaustion. Source 1 is loaded in full since it is the smaller query set.

---

## 2. Text Normalization

### 2.1 Business Name Normalization

**Step 1 — Lowercase and non-alphanumeric removal:**
```
"Smith & Jones, LLC." → "smith  jones  llc "
```

**Step 2 — Legal suffix stripping:**  
A regex pattern removes ~30 common business suffix tokens:
```
inc, incorporated, corp, corporation, ltd, limited, llc, co, company,
pvt, private, plc, sarl, sa, gmbh, llp, partnership, enterprises,
enterprise, solutions, services, service, group, associates, holdings,
international, intl, industries, global, technologies, technology, tech,
systems, consulting, consultants, consultant
```

**Step 3 — Compact form** (`nn`): Whitespace collapsed to empty → `"smithjones"`  
**Step 4 — Token form** (`nn_tok`): Normalized tokens space-joined → `"smith jones"`

### 2.2 Address Normalization

**Step 1 — Lowercase and punctuation removal**

**Step 2 — Abbreviation expansion** (~20 patterns):
```
st → street, rd → road, ave → avenue, blvd → boulevard,
dr → drive, ln → lane, ste → suite, apt → apartment,
fl → floor, po box → pobox, hwy → highway, pkwy → parkway, etc.
```

**Step 3 — Compact form** (`na`): Whitespace removed → `"123mainstreet"`  
**Step 4 — Token form** (`na_tok`): Space-joined → `"123 main street"`

### 2.3 Country Normalization

Country code uppercased and stripped: `"us"` → `"US"`.  
Country codes serve as a **mandatory matching constraint** — records with different or missing country codes are never matched.

### 2.4 Domain Extraction

Regex pattern searches the business name and address fields for domain-like patterns:
```
([a-z0-9\-]+\.(com|net|org|co|us|io|biz|info|in|uk|de))
```

### 2.5 Phone Number Extraction

Regex extracts 10-digit phone numbers from name and address fields, strips non-digit characters:
```python
re.sub(r'\D', '', m.group(0))
```

---

## 3. Candidate Blocking

Blocking reduces the O(N × M) cross-product (Source1 × Source2/3) to a tractable set of candidate pairs.

Two blocking keys are used (each further conditioned on country):

**Block Key 1 — Normalized compact name + country:**  
Key = `nn + '||' + nc`  
Minimum key length: 4 characters (prevents trivial blocks)

**Block Key 2 — Normalized compact address + country:**  
Key = `na + '||' + nc`  
Minimum key length: 6 characters

**Pair explosion guard:** Each block key is only used if:
- Target entity count for that key ≤ `MAX_TARGET_IN_KEY` (default: 10)
- Cross-product pair count ≤ `MAX_BLOCK_PAIRS` (default: 50)

Pairs from both blocking strategies are unioned and deduplicated.

---

## 4. Similarity Scoring

For each candidate pair (s1_id, cand_id), a composite score is computed.

### 4.1 Country Hard Filter

If the two records have different country codes (or either is blank), **score = 0.0** immediately.

### 4.2 Name Similarity (`n_score`)

- If compact names are exactly equal (and length ≥ 4): **n_score = 1.0**
- Otherwise: `max(ratio, token_sort_ratio)` from RapidFuzz applied to both compact and token forms

### 4.3 Address Similarity (`a_score`)

- If compact addresses are exactly equal (and length ≥ 6): **a_score = 1.0**
- Otherwise: `max(ratio, token_sort_ratio)` from RapidFuzz applied to token form

### 4.4 Address Number Agreement

Street numbers are extracted from address tokens using `re.findall(r'\d+', ...)`.  
If both addresses have numbers but their intersection / union ratio < 0.3, a **penalty of −0.35** is applied.

### 4.5 Domain/Phone Conflict Penalty

If domain or phone fields are **both non-empty but different** across the two records:  
**score −= 0.50**

### 4.6 Final Composite Score

```
score = 0.6 × n_score + 0.4 × a_score
```
Penalties applied post-combination. Score clipped to minimum 0.0.

---

## 5. Threshold-Based Decision (Submission #3 — Best Local F0.5)

### Two-threshold system with margin:

**Parameters used (Submission #3):**
- `tau_high = 0.85` — unconditional accept threshold
- `tau_low = 0.75` — lower conditional threshold
- `delta_margin = 0.05` — minimum margin between top and second candidate

**Decision Logic (per Source 1 entity):**
1. Sort all candidates by score descending.
2. Accept the top candidate if:
   - `top_score ≥ tau_high`, **OR**
   - `top_score ≥ tau_low AND (top_score − second_score) ≥ delta_margin`
3. Additionally accept any further candidates with `score ≥ tau_high`.
4. If no candidate passes: **leave prediction empty** (no false positive).

---

## 6. Singleton / Empty Prediction Handling

Entities where no candidate pair clears the threshold produce an **empty prediction string**.  
This is intentional — F0.5 rewards precision, and a correct empty prediction for a singleton entity (one with no true match) counts as precision = 1.0.

---

## 7. Output Format

Two output files:

**`matching_results.tsv`**
```
source1_entity_id\tmatched_entity_ids
```
- One row per Source 1 entity
- `matched_entity_ids`: comma-separated IDs from Source 2 or Source 3 (empty string if no match)

**`candidate_pairs.tsv`**
```
source1_entity_id\tcandidate_entity_ids
```
- Broader set: all pairs passing a pre-threshold score of ≥ 0.50

---

## 8. Evaluation Function

Macro F0.5 computed per entity then averaged:

```
precision = |pred ∩ true| / |pred|       (1.0 if pred is empty)
recall    = |pred ∩ true| / |true|       (depends on true label)
F0.5      = (1.25 × P × R) / (0.25 × P + R)
```

---

## 9. Design Decisions

| Decision | Rationale |
|---|---|
| F0.5 metric used | Precision-weighted → false positives are costlier than false negatives |
| Country as hard filter | Country mismatch is near-certain misalignment |
| Dual blocking keys | Name blocking misses name-variant matches; address blocking supplements |
| No graph closure | A→B and B→C does NOT imply A→C; direct evidence required |
| No external data | Only provided datasets used |
| Chunked processing | Large TSV files (~500 MB each) processed in 500k-row chunks to avoid OOM |
| Legal suffix removal | "LLC", "Inc", etc. add noise to name matching |
| Street number check | Ensures address similarity isn't just street name similarity |
| Phone/domain conflict | Explicit identifier conflicts → strong evidence of non-match |

---

## 10. Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, merging, chunking, output |
| `numpy` | Array operations, score arrays |
| `rapidfuzz` | Efficient string similarity (ratio, token_sort_ratio) |
| `re` | Regex-based normalization, phone/domain extraction |
| `gc` | Explicit garbage collection during large processing loops |
