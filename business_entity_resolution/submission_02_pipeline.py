# -*- coding: utf-8 -*-
"""Submission 2 – Optimized Pipeline

Improvements over Submission 1:
- Vectorized blocking using pandas merge (no iterrows / row-by-row loops).
- Block-size safety: oversized blocks filtered before merge to avoid OOM.
- rapidfuzz for fast fuzzy name+address scoring (replaces SequenceMatcher).
- Threshold sweep on training split to maximize macro F0.5.
- Generates submissions/day1/submission_02.tsv

DO NOT modify submissions/day1/submission_01.tsv.
"""

import os
import gc
import sys
import time
import psutil
import numpy as np
import pandas as pd
import rapidfuzz.fuzz as fuzz

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution"
DATA_DIR  = os.path.join(BASE_DIR, "dataset")
OUT_DIR   = os.path.join(BASE_DIR, "output")
SUB_DIR   = os.path.join(BASE_DIR, "submissions", "day1")
EXP_LOG   = os.path.join(SUB_DIR, "experiment_log.tsv")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Blocking safety limits
# ---------------------------------------------------------------------------
MAX_BLOCK_PAIRS   = 50   # max (s1_count * target_count) per key
MAX_TARGET_IN_KEY = 10   # max target-side records sharing a key (lower = less RAM)

# ---------------------------------------------------------------------------
# Normalisation helpers (vectorized, operate on Series)
# ---------------------------------------------------------------------------
def norm_name(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    suffixes = r'\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|plc|sarl|sa|gmbh)\b'
    s = s.str.replace(suffixes, ' ', regex=True)
    return s.str.replace(r'\s+', '', regex=True)   # compact, no spaces

def norm_addr(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    abbrs = [
        (r'\bst\b', 'street'), (r'\brd\b', 'road'), (r'\bave\b', 'avenue'),
        (r'\bblvd\b', 'boulevard'), (r'\bdr\b', 'drive'), (r'\bln\b', 'lane'),
        (r'\bste\b', 'suite'), (r'\bapt\b', 'apartment'), (r'\bfl\b', 'floor'),
        (r'\bpo box\b', 'pobox'),
    ]
    for pat, repl in abbrs:
        s = s.str.replace(pat, repl, regex=True)
    return s.str.replace(r'\s+', '', regex=True)   # compact, no spaces

def norm_country(s: pd.Series) -> pd.Series:
    return s.fillna('').astype(str).str.upper().str.strip()

# ---------------------------------------------------------------------------
# Load + normalise a single source file (loads one file at a time to save RAM)
# ---------------------------------------------------------------------------
def load_source(split: str, source: str) -> pd.DataFrame:
    """Load and normalise one source TSV.  source = 'source1'/'source2'/'source3'."""
    path = os.path.join(DATA_DIR, split, f"{split}_{source}.tsv")
    print(f"  Loading {split}_{source}.tsv ...", flush=True)
    df = pd.read_csv(path, sep='\t', quoting=3, dtype=str)
    df['nn'] = norm_name(df['business_name'])
    df['na'] = norm_addr(df['business_address'])
    df['nc'] = norm_country(df['country'])
    return df

def load_gt(split: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, split, f"{split}_ground_truth.tsv")
    return pd.read_csv(path, sep='\t', quoting=3, dtype=str)

# ---------------------------------------------------------------------------
# Vectorized blocking for one target source against S1
# Returns DataFrame with columns [s1_id, cand_id]
# ---------------------------------------------------------------------------
def block_one_target(s1: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
    """
    Two blocking keys:
      1) name_compact || country
      2) addr_compact || country
    Both use pandas merge with pre-computed per-key frequency checks
    to skip oversized blocks before they blow up memory.
    """

    # --- Name block ---
    s1_nc  = s1[s1['nn'].str.len() >= 4].copy()
    tgt_nc = tgt[tgt['nn'].str.len() >= 4].copy()

    s1_nc['key'] = s1_nc['nn'] + '||' + s1_nc['nc']
    tgt_nc['key'] = tgt_nc['nn'] + '||' + tgt_nc['nc']

    s1_nc_cnts  = s1_nc['key'].value_counts()
    tgt_nc_cnts = tgt_nc['key'].value_counts()

    common_nc = set(s1_nc_cnts.index).intersection(tgt_nc_cnts.index)
    bad_nc = {k for k in common_nc
              if s1_nc_cnts[k] * tgt_nc_cnts[k] > MAX_BLOCK_PAIRS
              or tgt_nc_cnts[k] > MAX_TARGET_IN_KEY}

    s1_nc_v  = s1_nc[~s1_nc['key'].isin(bad_nc)][['entity_id', 'key']].rename(columns={'entity_id': 's1_id'})
    tgt_nc_v = tgt_nc[~tgt_nc['key'].isin(bad_nc)][['entity_id', 'key']].rename(columns={'entity_id': 'cand_id'})
    m_nc = pd.merge(s1_nc_v, tgt_nc_v, on='key', how='inner')[['s1_id', 'cand_id']]
    del s1_nc, tgt_nc, s1_nc_v, tgt_nc_v, s1_nc_cnts, tgt_nc_cnts, common_nc, bad_nc
    gc.collect()

    # --- Address block ---
    s1_ac  = s1[s1['na'].str.len() >= 6].copy()
    tgt_ac = tgt[tgt['na'].str.len() >= 6].copy()

    s1_ac['key'] = s1_ac['na'] + '||' + s1_ac['nc']
    tgt_ac['key'] = tgt_ac['na'] + '||' + tgt_ac['nc']

    s1_ac_cnts  = s1_ac['key'].value_counts()
    tgt_ac_cnts = tgt_ac['key'].value_counts()

    common_ac = set(s1_ac_cnts.index).intersection(tgt_ac_cnts.index)
    bad_ac = {k for k in common_ac
              if s1_ac_cnts[k] * tgt_ac_cnts[k] > MAX_BLOCK_PAIRS
              or tgt_ac_cnts[k] > MAX_TARGET_IN_KEY}

    s1_ac_v  = s1_ac[~s1_ac['key'].isin(bad_ac)][['entity_id', 'key']].rename(columns={'entity_id': 's1_id'})
    tgt_ac_v = tgt_ac[~tgt_ac['key'].isin(bad_ac)][['entity_id', 'key']].rename(columns={'entity_id': 'cand_id'})
    m_ac = pd.merge(s1_ac_v, tgt_ac_v, on='key', how='inner')[['s1_id', 'cand_id']]
    del s1_ac, tgt_ac, s1_ac_v, tgt_ac_v, s1_ac_cnts, tgt_ac_cnts, common_ac, bad_ac
    gc.collect()

    pairs = pd.concat([m_nc, m_ac], ignore_index=True).drop_duplicates()
    del m_nc, m_ac
    gc.collect()
    return pairs

# ---------------------------------------------------------------------------
# Build compact lookup dicts  {entity_id -> (nn, na, nc)}
# ---------------------------------------------------------------------------
def build_lookup(df: pd.DataFrame) -> dict:
    return dict(zip(df['entity_id'], zip(df['nn'], df['na'], df['nc'])))

# ---------------------------------------------------------------------------
# Score a batch of pairs using rapidfuzz
# Returns numpy float32 array of combined scores
# ---------------------------------------------------------------------------
def score_pairs(s1_ids: np.ndarray, cand_ids: np.ndarray,
                s1_lk: dict, tgt_lk: dict) -> np.ndarray:
    n = len(s1_ids)
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        n1, a1, c1 = s1_lk[s1_ids[i]]
        rec = tgt_lk.get(cand_ids[i])
        if rec is None:
            continue
        n2, a2, c2 = rec

        if c1 != c2 or not c1:
            continue

        n_exact = (n1 == n2)
        a_exact = (a1 == a2)

        n_rat = 1.0 if n_exact else fuzz.ratio(n1, n2) / 100.0
        n_tok = 1.0 if n_exact else fuzz.token_sort_ratio(n1, n2) / 100.0
        n_sc  = max(n_rat, n_tok)

        a_rat = 1.0 if a_exact else fuzz.ratio(a1, a2) / 100.0
        a_tok = 1.0 if a_exact else fuzz.token_sort_ratio(a1, a2) / 100.0
        a_sc  = max(a_rat, a_tok)

        comb = 0.55 * n_sc + 0.45 * a_sc
        if n_sc >= 0.90 and a_sc >= 0.50:
            comb = max(comb, 0.70 * n_sc + 0.30 * a_sc)
        if a_sc >= 0.90 and n_sc >= 0.50:
            comb = max(comb, 0.30 * n_sc + 0.70 * a_sc)

        scores[i] = comb
    return scores

# ---------------------------------------------------------------------------
# Evaluation helper (macro F0.5, same formula as Submission 1 / validator)
# ---------------------------------------------------------------------------
def evaluate(preds: dict, gt: pd.DataFrame):
    total_prec = total_rec = total_f05 = 0.0
    sing_corr = sing_tot = 0
    n = len(gt)
    for _, row in gt.iterrows():
        sid      = row['source1_entity_id']
        m_str    = row['matched_entity_ids']
        true_set = set(m_str.split(',')) if pd.notna(m_str) and m_str else set()
        pred_set = preds.get(sid, set())
        p_len, t_len = len(pred_set), len(true_set)
        inter  = len(pred_set & true_set) if p_len and t_len else 0
        prec   = 1.0 if p_len == 0 else inter / p_len
        rec    = (1.0 if p_len == 0 else 0.0) if t_len == 0 else inter / t_len
        f05    = 0.0 if prec + rec == 0 else (1.25 * prec * rec) / (0.25 * prec + rec)
        total_prec += prec
        total_rec  += rec
        total_f05  += f05
        if t_len == 0:
            sing_tot += 1
            if p_len == 0:
                sing_corr += 1
    return (total_prec/n, total_rec/n, total_f05/n,
            sing_corr/sing_tot if sing_tot else 0.0)

# ---------------------------------------------------------------------------
# Threshold sweep on training split
# ---------------------------------------------------------------------------
def find_best_threshold(s1_ids_all: np.ndarray, cand_ids_all: np.ndarray,
                         scores_all: np.ndarray, gt: pd.DataFrame,
                         thresholds=None):
    if thresholds is None:
        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    best_thr = 0.80
    best_f05 = -1.0
    best_metrics = None

    for thr in thresholds:
        mask  = scores_all >= thr
        preds = {}
        for sid, cid in zip(s1_ids_all[mask], cand_ids_all[mask]):
            preds.setdefault(sid, set()).add(cid)
        m_prec, m_rec, m_f05, sing_acc = evaluate(preds, gt)
        print(f"  thr={thr:.2f} -> P={m_prec:.4f} R={m_rec:.4f} F0.5={m_f05:.4f} Sing={sing_acc:.4f}", flush=True)
        if m_f05 > best_f05:
            best_f05     = m_f05
            best_thr     = thr
            best_metrics = (m_prec, m_rec, m_f05, sing_acc)

    return best_thr, best_metrics

# ---------------------------------------------------------------------------
# Save output files
# ---------------------------------------------------------------------------
def save_outputs(preds: dict, candidates: dict, s1_all_ids):
    # matching_results.tsv
    rows_m = [(sid, ','.join(sorted(preds.get(sid, set())))) for sid in s1_all_ids]
    pd.DataFrame(rows_m, columns=['source1_entity_id', 'matched_entity_ids']).to_csv(
        os.path.join(OUT_DIR, 'matching_results.tsv'), sep='\t', index=False)

    # candidate_pairs.tsv
    rows_c = [(sid, ','.join(sorted(candidates.get(sid, set())))) for sid in s1_all_ids]
    pd.DataFrame(rows_c, columns=['source1_entity_id', 'candidate_entity_ids']).to_csv(
        os.path.join(OUT_DIR, 'candidate_pairs.tsv'), sep='\t', index=False)

    # submission file
    sub_path = os.path.join(SUB_DIR, 'submission_02.tsv')
    pd.DataFrame(rows_m, columns=['source1_entity_id', 'matched_entity_ids']).to_csv(
        sub_path, sep='\t', index=False)
    print(f"\nSubmission saved to: {sub_path}", flush=True)

# ---------------------------------------------------------------------------
# Log experiment result
# ---------------------------------------------------------------------------
def log_experiment(metrics, threshold, total_preds, runtime):
    log_exists = os.path.exists(EXP_LOG)
    with open(EXP_LOG, 'a', encoding='utf-8') as f:
        if not log_exists:
            f.write("experiment_id\tnormalization_rules\tmatching_rules\t"
                    "val_precision\tval_recall\tval_f0.5\tpredicted_matches\t"
                    "singleton_acc\tthreshold\truntime\n")
        m = metrics
        f.write(f"submission_02\tcompact_norm\tvec_block+rapidfuzz\t"
                f"{m[0]:.6f}\t{m[1]:.6f}\t{m[2]:.6f}\t{total_preds}\t"
                f"{m[3]:.6f}\t{threshold:.2f}\t{runtime:.1f}\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_split(split: str, best_thr: float = None, gt=None):
    """
    Load S1 once, then process S2 and S3 one-at-a-time.
    Returns (s1_ids, cand_ids, scores, all_s1_ids, s1_lk_snapshot)
    If best_thr is None we are in train mode (return raw arrays for sweep).
    If best_thr is set we are in test mode (return preds + cands dicts).
    """
    ram = lambda: psutil.virtual_memory().available / 1024**3

    # --- Load S1 (kept in RAM throughout) ---
    s1 = load_source(split, 'source1')
    all_s1_ids = s1['entity_id'].values.tolist()
    s1_lk = build_lookup(s1)
    print(f"  S1 rows: {len(s1):,} | RAM: {ram():.2f} GB", flush=True)

    collected_s1_ids = []
    collected_cand_ids = []
    tgt_lk: dict = {}

    for src_name in ('source2', 'source3'):
        tgt = load_source(split, src_name)
        print(f"  Blocking vs {src_name} ...", flush=True)
        pairs = block_one_target(s1, tgt)
        print(f"  {src_name}: {len(pairs):,} pairs | RAM: {ram():.2f} GB", flush=True)

        # Build lookup only for matched candidates
        matched = set(pairs['cand_id'].unique())
        lk = build_lookup(tgt[tgt['entity_id'].isin(matched)])
        tgt_lk.update(lk)
        del tgt, lk, matched; gc.collect()

        # Convert pairs to arrays immediately and free DataFrame
        collected_s1_ids.append(pairs['s1_id'].values)
        collected_cand_ids.append(pairs['cand_id'].values)
        del pairs; gc.collect()

    del s1; gc.collect()

    # Combine + deduplicate using a set
    s1_arr   = np.concatenate(collected_s1_ids)
    cand_arr = np.concatenate(collected_cand_ids)
    del collected_s1_ids, collected_cand_ids; gc.collect()

    seen = set()
    keep_mask = np.zeros(len(s1_arr), dtype=bool)
    for i, (s, c) in enumerate(zip(s1_arr, cand_arr)):
        key = (s, c)
        if key not in seen:
            seen.add(key)
            keep_mask[i] = True
    del seen; gc.collect()
    s1_arr   = s1_arr[keep_mask]
    cand_arr = cand_arr[keep_mask]
    del keep_mask; gc.collect()

    total_pairs = len(s1_arr)
    print(f"  Total candidate pairs: {total_pairs:,} | RAM: {ram():.2f} GB", flush=True)

    if total_pairs > 15_000_000:
        print("ERROR: Too many candidate pairs – aborting.", flush=True)
        sys.exit(1)

    print(f"  Scoring {total_pairs:,} pairs ...", flush=True)
    scores = score_pairs(s1_arr, cand_arr, s1_lk, tgt_lk)
    del s1_lk, tgt_lk; gc.collect()

    return s1_arr, cand_arr, scores, all_s1_ids


def main():
    t_global = time.time()
    print("=== SUBMISSION 2 PIPELINE (Optimized) ===\n", flush=True)
    ram = lambda: psutil.virtual_memory().available / 1024**3

    # ---- PHASE 1: Train split – threshold sweep ----
    print("--- PHASE 1: Training split (threshold search) ---", flush=True)
    t0 = time.time()

    s1_ids_tr, cand_ids_tr, scores_tr, all_s1_ids_tr = process_split('train')
    gt = load_gt('train')

    print("\n  Threshold sweep:", flush=True)
    best_thr, best_metrics = find_best_threshold(s1_ids_tr, cand_ids_tr, scores_tr, gt)

    total_train_preds = int((scores_tr >= best_thr).sum())
    del s1_ids_tr, cand_ids_tr, scores_tr, gt; gc.collect()

    print(f"\nBest threshold: {best_thr:.2f}  |  F0.5={best_metrics[2]:.4f}", flush=True)
    print(f"Training phase done in {time.time()-t0:.1f}s\n", flush=True)

    # ---- PHASE 2: Test split – generate submission ----
    print("--- PHASE 2: Test split (generate submission) ---", flush=True)

    s1_ids_te, cand_ids_te, scores_te, all_s1_ids_te = process_split('test')

    mask_te  = scores_te >= best_thr
    preds_te: dict = {}
    cands_te: dict = {}
    for sid, cid, keep in zip(s1_ids_te, cand_ids_te, mask_te):
        cands_te.setdefault(sid, set()).add(cid)
        if keep:
            preds_te.setdefault(sid, set()).add(cid)
    del s1_ids_te, cand_ids_te, scores_te, mask_te; gc.collect()

    save_outputs(preds_te, cands_te, all_s1_ids_te)

    runtime = time.time() - t_global
    log_experiment(best_metrics, best_thr, total_train_preds, runtime)

    print(f"\n=== DONE in {runtime:.1f}s ===", flush=True)
    print(f"  Best threshold : {best_thr:.2f}", flush=True)
    print(f"  Val Precision  : {best_metrics[0]:.4f}", flush=True)
    print(f"  Val Recall     : {best_metrics[1]:.4f}", flush=True)
    print(f"  Val Macro F0.5 : {best_metrics[2]:.4f}", flush=True)
    print(f"  Singleton Acc  : {best_metrics[3]:.4f}", flush=True)

if __name__ == "__main__":
    main()
