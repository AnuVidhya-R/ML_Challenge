# -*- coding: utf-8 -*-
"""Submission 3 – Multi-Signal Entity Matching Pipeline (Memory Safe)

Improvements over Submission 2:
1. BROADER BLOCKING: name-prefix + country blocking
...
"""

import os
import gc
import sys
import time
import re
import numpy as np
import pandas as pd
import rapidfuzz.fuzz as fuzz

# Paths
BASE_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution"
DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUT_DIR  = os.path.join(BASE_DIR, "output")
SUB_DIR  = os.path.join(BASE_DIR, "submissions", "day1")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

MAX_BLOCK_PAIRS   = 50
MAX_TARGET_IN_KEY = 10
CHUNK_SIZE = 500_000

BUSINESS_SUFFIXES = r'\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|plc|sarl|sa|gmbh|llp|partnership|enterprises|enterprise|solutions|services|service|group|associates|holdings|international|intl|industries|global|technologies|technology|tech|systems|consulting|consultants|consultant)\b'

def norm_name(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s = s.str.replace(BUSINESS_SUFFIXES, ' ', regex=True)
    return s.str.replace(r'\s+', '', regex=True)

def norm_name_tokens(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s = s.str.replace(BUSINESS_SUFFIXES, ' ', regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

ADDR_ABBREVS = [
    (r'\bst\b', 'street'), (r'\brd\b', 'road'), (r'\bave\b', 'avenue'),
    (r'\bblvd\b', 'boulevard'), (r'\bdr\b', 'drive'), (r'\bln\b', 'lane'),
    (r'\bste\b', 'suite'), (r'\bapt\b', 'apartment'), (r'\bfl\b', 'floor'),
    (r'\bpo box\b', 'pobox'), (r'\bhwy\b', 'highway'), (r'\bpky\b', 'parkway'),
    (r'\bpkwy\b', 'parkway'), (r'\bcir\b', 'circle'), (r'\bct\b', 'court'),
    (r'\bpl\b', 'place'), (r'\bno\b', 'number'), (r'\bw\b', 'west'),
    (r'\be\b', 'east'), (r'\bn\b', 'north'), (r'\bs\b', 'south'),
]

def norm_addr(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    for pat, repl in ADDR_ABBREVS:
        s = s.str.replace(pat, repl, regex=True)
    return s.str.replace(r'\s+', '', regex=True)

def norm_addr_tokens(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    for pat, repl in ADDR_ABBREVS:
        s = s.str.replace(pat, repl, regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

def norm_country(s: pd.Series) -> pd.Series:
    return s.fillna('').astype(str).str.upper().str.strip()

def name_first_token(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s = s.str.replace(r'^\s*the\s+', '', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s.str.split(' ').str[0].fillna('')

def name_prefix(s: pd.Series, n: int = 5) -> pd.Series:
    return s.str[:n]

def apply_normalizations(df: pd.DataFrame) -> pd.DataFrame:
    df['nn'] = norm_name(df['business_name'])
    df['na'] = norm_addr(df['business_address'])
    df['nc'] = norm_country(df['country'])
    df['nn_tok'] = norm_name_tokens(df['business_name'])
    df['na_tok'] = norm_addr_tokens(df['business_address'])
    df['n_ft'] = name_first_token(df['business_name'])
    df['n_pfx'] = name_prefix(df['nn'], 5)
    return df

def load_source(split: str, source: str, nrows=None) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, split, f"{split}_{source}.tsv")
    print(f"  Loading {split}_{source}.tsv ...", flush=True)
    df = pd.read_csv(path, sep='\t', quoting=3, dtype=str, nrows=nrows)
    return apply_normalizations(df)

def load_source_chunks(split: str, source: str, nrows=None):
    path = os.path.join(DATA_DIR, split, f"{split}_{source}.tsv")
    print(f"  Loading {split}_{source}.tsv in chunks ...", flush=True)
    reader = pd.read_csv(path, sep='\t', quoting=3, dtype=str, chunksize=CHUNK_SIZE, nrows=nrows)
    for chunk in reader:
        yield apply_normalizations(chunk)

def load_gt(split: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, split, f"{split}_ground_truth.tsv")
    return pd.read_csv(path, sep='\t', quoting=3, dtype=str)

# Blocking for one target chunk
def block_chunk(s1: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
    all_pairs = []

    # Safe filtering helper to avoid pyarrow memory error
    def merge_safe(s1_sub, tgt_sub, key_col, max_pairs, max_tgt):
        s1_c = s1_sub[s1_sub[key_col].str.len() >= 4].copy()
        tgt_c = tgt_sub[tgt_sub[key_col].str.len() >= 4].copy()
        
        s1_c['key'] = s1_c[key_col] + '||' + s1_c['nc']
        tgt_c['key'] = tgt_c[key_col] + '||' + tgt_c['nc']
        
        # Simple Python dict for counts to avoid pandas overhead
        s1_counts = s1_c['key'].value_counts().to_dict()
        tgt_counts = tgt_c['key'].value_counts().to_dict()
        
        # Determine good keys
        good_keys = set()
        for k, t_cnt in tgt_counts.items():
            s_cnt = s1_counts.get(k, 0)
            if s_cnt > 0 and (s_cnt * t_cnt <= max_pairs) and (t_cnt <= max_tgt):
                good_keys.add(k)
        
        # Filter with simple mask
        s1_v = s1_c[s1_c['key'].apply(lambda x: x in good_keys)][['entity_id', 'key']].rename(columns={'entity_id': 's1_id'})
        tgt_v = tgt_c[tgt_c['key'].apply(lambda x: x in good_keys)][['entity_id', 'key']].rename(columns={'entity_id': 'cand_id'})
        
        res = pd.merge(s1_v, tgt_v, on='key', how='inner')[['s1_id', 'cand_id']]
        return res

    # 1: Exact name
    all_pairs.append(merge_safe(s1, tgt, 'nn', MAX_BLOCK_PAIRS, MAX_TARGET_IN_KEY))
    
    # 2: Exact address (len >= 6)
    def merge_addr(s1_sub, tgt_sub):
        s1_c = s1_sub[s1_sub['na'].str.len() >= 6].copy()
        tgt_c = tgt_sub[tgt_sub['na'].str.len() >= 6].copy()
        s1_c['key'] = s1_c['na'] + '||' + s1_c['nc']
        tgt_c['key'] = tgt_c['na'] + '||' + tgt_c['nc']
        
        s1_counts = s1_c['key'].value_counts().to_dict()
        tgt_counts = tgt_c['key'].value_counts().to_dict()
        good_keys = {k for k, t_cnt in tgt_counts.items() if s1_counts.get(k,0)*t_cnt <= MAX_BLOCK_PAIRS and t_cnt <= MAX_TARGET_IN_KEY}
        
        s1_v = s1_c[s1_c['key'].apply(lambda x: x in good_keys)][['entity_id', 'key']].rename(columns={'entity_id': 's1_id'})
        tgt_v = tgt_c[tgt_c['key'].apply(lambda x: x in good_keys)][['entity_id', 'key']].rename(columns={'entity_id': 'cand_id'})
        return pd.merge(s1_v, tgt_v, on='key', how='inner')[['s1_id', 'cand_id']]
    all_pairs.append(merge_addr(s1, tgt))
    
    # 3: Name prefix
    all_pairs.append(merge_safe(s1, tgt, 'n_pfx', 30, 6))
    
    # 4: First name token
    all_pairs.append(merge_safe(s1, tgt, 'n_ft', 25, 5))

    pairs = pd.concat(all_pairs, ignore_index=True).drop_duplicates()
    del all_pairs
    gc.collect()
    return pairs

def build_lookup(df: pd.DataFrame) -> dict:
    return dict(zip(df['entity_id'],
                    zip(df['nn'], df['na'], df['nc'], df['nn_tok'], df['na_tok'])))

_NUM_RE = re.compile(r'\d+')
def extract_numbers(addr: str) -> set:
    if not addr:
        return set()
    return set(_NUM_RE.findall(addr))

def score_pairs_multi(s1_ids: np.ndarray, cand_ids: np.ndarray,
                      s1_lk: dict, tgt_lk: dict) -> np.ndarray:
    n = len(s1_ids)
    scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        rec1 = s1_lk.get(s1_ids[i])
        rec2 = tgt_lk.get(cand_ids[i])
        if rec1 is None or rec2 is None:
            continue

        n1, a1, c1, n1_tok, a1_tok = rec1
        n2, a2, c2, n2_tok, a2_tok = rec2

        if c1 != c2 or not c1:
            scores[i] = 0.0
            continue

        n_exact = (n1 == n2 and len(n1) >= 3)
        if n_exact:
            n_score = 1.0
        else:
            nr = fuzz.ratio(n1_tok, n2_tok) / 100.0
            ntsr = fuzz.token_sort_ratio(n1_tok, n2_tok) / 100.0
            ntsetr = fuzz.token_set_ratio(n1_tok, n2_tok) / 100.0
            nr_compact = fuzz.ratio(n1, n2) / 100.0
            n_score = max(nr, ntsr, ntsetr, nr_compact)

        a_exact = (a1 == a2 and len(a1) >= 5)
        if a_exact:
            a_score = 1.0
            num_agree = True
        else:
            ar = fuzz.ratio(a1_tok, a2_tok) / 100.0
            atsr = fuzz.token_sort_ratio(a1_tok, a2_tok) / 100.0
            atsetr = fuzz.token_set_ratio(a1_tok, a2_tok) / 100.0
            ar_compact = fuzz.ratio(a1, a2) / 100.0
            a_score = max(ar, atsr, atsetr, ar_compact)

            nums1 = extract_numbers(a1_tok)
            nums2 = extract_numbers(a2_tok)
            if nums1 and nums2:
                common_nums = nums1 & nums2
                all_nums = nums1 | nums2
                num_agree = len(common_nums) / len(all_nums) >= 0.3 if all_nums else True
            else:
                num_agree = True

        if n_exact and a_exact:
            scores[i] = 1.0
            continue
        if n_exact and a_score >= 0.6:
            scores[i] = max(0.90, 0.50 * n_score + 0.50 * a_score)
            continue
        if a_exact and n_score >= 0.6:
            scores[i] = max(0.90, 0.50 * n_score + 0.50 * a_score)
            continue
        if n_score >= 0.85 and a_score >= 0.70:
            comb = 0.55 * n_score + 0.45 * a_score
            scores[i] = max(comb, 0.80)
            continue
        if not num_agree and a_score < 0.85:
            a_score *= 0.7

        if n_score >= 0.85 and a_score >= 0.50:
            comb = 0.60 * n_score + 0.40 * a_score
        elif a_score >= 0.85 and n_score >= 0.50:
            comb = 0.40 * n_score + 0.60 * a_score
        else:
            comb = 0.55 * n_score + 0.45 * a_score

        scores[i] = comb

    return scores

def evaluate(preds: dict, gt: pd.DataFrame):
    total_prec = total_rec = total_f05 = 0.0
    sing_corr = sing_tot = 0
    n = len(gt)
    for _, row in gt.iterrows():
        sid = row['source1_entity_id']
        m_str = row['matched_entity_ids']
        true_set = set(m_str.split(',')) if pd.notna(m_str) and m_str else set()
        pred_set = preds.get(sid, set())
        p_len, t_len = len(pred_set), len(true_set)
        inter = len(pred_set & true_set) if p_len and t_len else 0
        prec = 1.0 if p_len == 0 else inter / p_len
        rec = (1.0 if p_len == 0 else 0.0) if t_len == 0 else inter / t_len
        f05 = 0.0 if prec + rec == 0 else (1.25 * prec * rec) / (0.25 * prec + rec)
        total_prec += prec
        total_rec += rec
        total_f05 += f05
        if t_len == 0:
            sing_tot += 1
            if p_len == 0:
                sing_corr += 1
    return (total_prec/n, total_rec/n, total_f05/n, sing_corr/sing_tot if sing_tot else 0.0)

def find_best_threshold(s1_ids_all: np.ndarray, cand_ids_all: np.ndarray,
                        scores_all: np.ndarray, gt: pd.DataFrame):
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    best_thr = 0.80
    best_f05 = -1.0
    best_metrics = None

    for thr in thresholds:
        mask = scores_all >= thr
        preds = {}
        for sid, cid in zip(s1_ids_all[mask], cand_ids_all[mask]):
            preds.setdefault(sid, set()).add(cid)
        m_prec, m_rec, m_f05, sing_acc = evaluate(preds, gt)
        print(f"  thr={thr:.2f} -> P={m_prec:.4f} R={m_rec:.4f} F0.5={m_f05:.4f}", flush=True)
        if m_f05 > best_f05:
            best_f05 = m_f05
            best_thr = thr
            best_metrics = (m_prec, m_rec, m_f05, sing_acc)

    return best_thr, best_metrics

def apply_threshold_with_margin(s1_ids, cand_ids, scores, threshold):
    preds = {}
    cands = {}
    pair_data = {}
    for i in range(len(s1_ids)):
        sid = s1_ids[i]
        cid = cand_ids[i]
        sc = scores[i]
        if sid not in pair_data:
            pair_data[sid] = []
        pair_data[sid].append((cid, sc))
        cands.setdefault(sid, set()).add(cid)

    for sid, candidates in pair_data.items():
        candidates.sort(key=lambda x: x[1], reverse=True)
        accepted = {cid for cid, sc in candidates if sc >= threshold}
        if accepted:
            preds[sid] = accepted
    return preds, cands

def save_outputs(preds: dict, candidates: dict, s1_all_ids):
    rows_m = [(sid, ','.join(sorted(preds.get(sid, set())))) for sid in s1_all_ids]
    pd.DataFrame(rows_m, columns=['source1_entity_id', 'matched_entity_ids']).to_csv(
        os.path.join(OUT_DIR, 'matching_results.tsv'), sep='\t', index=False)
    
    rows_c = [(sid, ','.join(sorted(candidates.get(sid, set())))) for sid in s1_all_ids]
    pd.DataFrame(rows_c, columns=['source1_entity_id', 'candidate_entity_ids']).to_csv(
        os.path.join(OUT_DIR, 'candidate_pairs.tsv'), sep='\t', index=False)
    
    sub_path = os.path.join(SUB_DIR, 'submission_03.tsv')
    pd.DataFrame(rows_m, columns=['source1_entity_id', 'matched_entity_ids']).to_csv(
        sub_path, sep='\t', index=False)
    print(f"\nSubmission saved to: {sub_path}", flush=True)

def process_split(split: str, nrows=None):
    try:
        import psutil
        ram = lambda: psutil.virtual_memory().available / 1024**3
    except:
        ram = lambda: -1

    s1 = load_source(split, 'source1', nrows=nrows)
    all_s1_ids = s1['entity_id'].values.tolist()
    s1_lk = build_lookup(s1)
    
    print(f"  S1 loaded. RAM: {ram():.2f} GB", flush=True)

    collected_s1_ids = []
    collected_cand_ids = []
    collected_scores = []

    for src_name in ('source2', 'source3'):
        print(f"  Blocking and scoring vs {src_name} ...", flush=True)
        t_block = time.time()
        
        chunk_idx = 0
        for chunk in load_source_chunks(split, src_name, nrows=nrows):
            chunk_idx += 1
            pairs = block_chunk(s1, chunk)
            
            matched = set(pairs['cand_id'].unique())
            tgt_lk = build_lookup(chunk[chunk['entity_id'].isin(matched)])
            
            s1_arr = pairs['s1_id'].values
            cand_arr = pairs['cand_id'].values
            
            scores = score_pairs_multi(s1_arr, cand_arr, s1_lk, tgt_lk)
            
            # Keep only scores >= 0.65 to allow for threshold sweep >= 0.70
            keep = scores >= 0.65
            
            collected_s1_ids.append(s1_arr[keep])
            collected_cand_ids.append(cand_arr[keep])
            collected_scores.append(scores[keep])
            
            print(f"    Chunk {chunk_idx}: Pairs={len(s1_arr):,}, Kept={keep.sum():,}, RAM={ram():.2f}GB", flush=True)
            
            del pairs, chunk, matched, tgt_lk, s1_arr, cand_arr, scores, keep
            gc.collect()

        print(f"  Finished {src_name} in {time.time()-t_block:.1f}s", flush=True)

    del s1; del s1_lk; gc.collect()

    s1_arr = np.concatenate(collected_s1_ids) if collected_s1_ids else np.array([])
    cand_arr = np.concatenate(collected_cand_ids) if collected_cand_ids else np.array([])
    scores = np.concatenate(collected_scores) if collected_scores else np.array([], dtype=np.float32)
    
    del collected_s1_ids, collected_cand_ids, collected_scores; gc.collect()

    # Deduplicate in case a candidate appeared in multiple chunks
    seen = set()
    keep_mask = np.zeros(len(s1_arr), dtype=bool)
    for i in range(len(s1_arr)):
        key = (s1_arr[i], cand_arr[i])
        if key not in seen:
            seen.add(key)
            keep_mask[i] = True
            
    del seen; gc.collect()
    s1_arr = s1_arr[keep_mask]
    cand_arr = cand_arr[keep_mask]
    scores = scores[keep_mask]
    del keep_mask; gc.collect()

    print(f"  Total candidate pairs kept: {len(s1_arr):,} | RAM: {ram():.2f} GB", flush=True)
    
    return s1_arr, cand_arr, scores, all_s1_ids

def main():
    t_global = time.time()
    print("=== SUBMISSION 3 MEMORY-SAFE RUN ===", flush=True)
    
    try:
        import psutil
        ram = lambda: psutil.virtual_memory().available / 1024**3
    except:
        ram = lambda: -1

    # SMOKE TEST
    print("\n--- SMOKE TEST (First 10,000 rows) ---", flush=True)
    try:
        process_split('train', nrows=10000)
        print("Smoke test passed successfully.", flush=True)
    except Exception as e:
        print(f"Smoke test failed: {e}")
        sys.exit(1)

    # FULL TRAIN
    print("\n--- PHASE 1: Training split ---", flush=True)
    s1_ids_tr, cand_ids_tr, scores_tr, all_s1_ids_tr = process_split('train')
    gt = load_gt('train')
    
    best_thr, best_metrics = find_best_threshold(s1_ids_tr, cand_ids_tr, scores_tr, gt)
    train_runtime = time.time() - t_global
    
    del s1_ids_tr, cand_ids_tr, scores_tr, gt; gc.collect()
    
    print(f"\nBest Threshold: {best_thr:.2f}, F0.5: {best_metrics[2]:.4f}")
    
    # FULL TEST
    print("\n--- PHASE 2: Test split ---", flush=True)
    s1_ids_te, cand_ids_te, scores_te, all_s1_ids_te = process_split('test')
    
    preds_te, cands_te = apply_threshold_with_margin(s1_ids_te, cand_ids_te, scores_te, best_thr)
    save_outputs(preds_te, cands_te, all_s1_ids_te)
    
    test_runtime = time.time() - (t_global + train_runtime)
    total_runtime = time.time() - t_global
    
    print("\n=== SUBMISSION 3 MEMORY-SAFE RUN ===")
    print("Training status: COMPLETED")
    print("Peak/approx RAM: {:.2f} GB".format(ram()))
    print(f"Validation Precision: {best_metrics[0]:.4f}")
    print(f"Validation Recall: {best_metrics[1]:.4f}")
    print(f"Validation Macro F0.5: {best_metrics[2]:.4f}")
    print(f"Singleton Accuracy: {best_metrics[3]:.4f}")
    print(f"Best Threshold: {best_thr:.2f}")
    print(f"Runtime: {total_runtime:.1f}s")

if __name__ == '__main__':
    main()
