import os
import sys
import time
import gc
import psutil
import pandas as pd
import numpy as np
import rapidfuzz.fuzz as fuzz

print("Testing Optimized Submission 2 Training Pipeline...", flush=True)

DATA_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\dataset"

t0 = time.time()
print("STEP 1: Loading data & STEP 2: Normalization (Train S1)...", flush=True)

def quick_norm_name(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    suffixes = r"\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|plc|sarl|sa|gmbh)\b"
    s = s.str.replace(suffixes, ' ', regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

def quick_norm_addr(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    abbrs = [
        (r'\bst\b', 'street'), (r'\brd\b', 'road'), (r'\bave\b', 'avenue'),
        (r'\bblvd\b', 'boulevard'), (r'\bdr\b', 'drive'), (r'\bln\b', 'lane'),
        (r'\bste\b', 'suite'), (r'\bapt\b', 'apartment'), (r'\bfl\b', 'floor'),
        (r'\bpo box\b', 'pobox')
    ]
    for pat, repl in abbrs:
        s = s.str.replace(pat, repl, regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

df_s1 = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source1.tsv'), sep='\t', quoting=3, dtype=str)
n_name1 = quick_norm_name(df_s1['business_name'])
n_addr1 = quick_norm_addr(df_s1['business_address'])
n_ctry1 = df_s1['country'].fillna('').astype(str).str.lower().str.strip()

key_nc1 = n_name1 + "||" + n_ctry1
key_ac1 = n_addr1 + "||" + n_ctry1

# Keep S1 blocking tables
s1_nc = pd.DataFrame({'s1_id': df_s1['entity_id'][n_name1.str.len() >= 3], 'key_nc': key_nc1[n_name1.str.len() >= 3]})
s1_ac = pd.DataFrame({'s1_id': df_s1['entity_id'][n_addr1.str.len() >= 5], 'key_ac': key_ac1[n_addr1.str.len() >= 5]})

# Store full S1 normalized columns in compact df
s1_store = pd.DataFrame({
    'entity_id': df_s1['entity_id'],
    'norm_name': n_name1,
    'norm_address': n_addr1,
    'norm_country': n_ctry1
})

del df_s1, n_name1, n_addr1, n_ctry1, key_nc1, key_ac1
gc.collect()

print(f"STEP 2 completed in {time.time()-t0:.2f}s, RAM avail: {psutil.virtual_memory().available / 1024**3:.2f} GB", flush=True)

print("\nSTEP 3: Candidate blocking...", flush=True)
t_block = time.time()

def process_and_merge_target(split, source_name, s1_nc_df, s1_ac_df, max_block_size=100):
    t_sub = time.time()
    fpath = os.path.join(DATA_DIR, split, f"{split}_{source_name}.tsv")
    print(f"  Loading & normalizing {source_name}...", flush=True)
    df = pd.read_csv(fpath, sep='\t', quoting=3, dtype=str)
    
    n_name = quick_norm_name(df['business_name'])
    n_addr = quick_norm_addr(df['business_address'])
    n_ctry = df['country'].fillna('').astype(str).str.lower().str.strip()
    
    key_nc = n_name + "||" + n_ctry
    key_ac = n_addr + "||" + n_ctry
    
    # Name blocking
    cand_nc = pd.DataFrame({'cand_id': df['entity_id'][n_name.str.len() >= 3], 'key_nc': key_nc[n_name.str.len() >= 3]})
    nc_counts = cand_nc['key_nc'].value_counts()
    oversized_nc_keys = set(nc_counts[nc_counts > max_block_size].index)
    
    cand_nc_valid = cand_nc[~cand_nc['key_nc'].isin(oversized_nc_keys)]
    s1_nc_valid = s1_nc_df[~s1_nc_df['key_nc'].isin(oversized_nc_keys)]
    m_nc = pd.merge(s1_nc_valid, cand_nc_valid, on='key_nc', how='inner')[['s1_id', 'cand_id']]
    del cand_nc, cand_nc_valid, s1_nc_valid
    gc.collect()
    
    # Address blocking
    cand_ac = pd.DataFrame({'cand_id': df['entity_id'][n_addr.str.len() >= 5], 'key_ac': key_ac[n_addr.str.len() >= 5]})
    ac_counts = cand_ac['key_ac'].value_counts()
    oversized_ac_keys = set(ac_counts[ac_counts > max_block_size].index)
    
    cand_ac_valid = cand_ac[~cand_ac['key_ac'].isin(oversized_ac_keys)]
    s1_ac_valid = s1_ac_df[~s1_ac_df['key_ac'].isin(oversized_ac_keys)]
    m_ac = pd.merge(s1_ac_valid, cand_ac_valid, on='key_ac', how='inner')[['s1_id', 'cand_id']]
    del cand_ac, cand_ac_valid, s1_ac_valid
    gc.collect()
    
    pairs = pd.concat([m_nc, m_ac], ignore_index=True).drop_duplicates()
    del m_nc, m_ac
    gc.collect()
    
    # Extract only matching candidate strings for memory safety
    matched_cands = set(pairs['cand_id'].unique())
    mask_matched = df['entity_id'].isin(matched_cands)
    target_lookup = dict(zip(df['entity_id'][mask_matched], zip(n_name[mask_matched], n_addr[mask_matched], n_ctry[mask_matched])))
    
    del df, n_name, n_addr, n_ctry, key_nc, key_ac, matched_cands, mask_matched
    gc.collect()
    
    stats = {
        'nc_oversized': len(oversized_nc_keys),
        'ac_oversized': len(oversized_ac_keys),
        'nc_counts': nc_counts[~nc_counts.index.isin(oversized_nc_keys)],
        'ac_counts': ac_counts[~ac_counts.index.isin(oversized_ac_keys)],
    }
    print(f"  {source_name}: {len(pairs)} pairs in {time.time()-t_sub:.2f}s, RAM avail: {psutil.virtual_memory().available / 1024**3:.2f} GB", flush=True)
    return pairs, target_lookup, stats

pairs_s2, lookup_s2, stats_s2 = process_and_merge_target('train', 'source2', s1_nc, s1_ac, max_block_size=100)
pairs_s3, lookup_s3, stats_s3 = process_and_merge_target('train', 'source3', s1_nc, s1_ac, max_block_size=100)
del s1_nc, s1_ac
gc.collect()

all_pairs = pd.concat([pairs_s2, pairs_s3], ignore_index=True).drop_duplicates()
del pairs_s2, pairs_s3
gc.collect()

total_pairs = len(all_pairs)
total_oversized = stats_s2['nc_oversized'] + stats_s2['ac_oversized'] + stats_s3['nc_oversized'] + stats_s3['ac_oversized']
all_valid_counts = pd.concat([stats_s2['nc_counts'], stats_s2['ac_counts'], stats_s3['nc_counts'], stats_s3['ac_counts']])
largest_block = all_valid_counts.max() if len(all_valid_counts) > 0 else 0
avg_block_size = all_valid_counts.mean() if len(all_valid_counts) > 0 else 0
del stats_s2, stats_s3, all_valid_counts
gc.collect()

print(f"\n--- STEP 4: CANDIDATE STATISTICS ---", flush=True)
print(f"Total candidate pairs: {total_pairs}")
print(f"Largest block: {largest_block}")
print(f"Average block size: {avg_block_size:.2f}")
print(f"Number of oversized blocks: {total_oversized}")
print(f"Blocking completed in {time.time()-t_block:.2f}s", flush=True)

# Merge target lookups
target_lookup = {**lookup_s2, **lookup_s3}
del lookup_s2, lookup_s3
gc.collect()

# Candidate recall on Ground Truth
print("\nEvaluating Candidate Recall against Ground Truth...", flush=True)
t_gt = time.time()
gt = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv'), sep='\t', quoting=3, dtype=str)
true_pairs_set = set()
for s1_id, m_str in zip(gt['source1_entity_id'], gt['matched_entity_ids'].fillna('')):
    if m_str:
        for cid in m_str.split(','):
            true_pairs_set.add((s1_id, cid.strip()))

cand_set = set(zip(all_pairs['s1_id'], all_pairs['cand_id']))
found_true = len(true_pairs_set.intersection(cand_set))
cand_recall = found_true / len(true_pairs_set) if len(true_pairs_set) > 0 else 0.0
print(f"Candidate count: {total_pairs}")
print(f"Candidate recall: {cand_recall:.4f} ({found_true}/{len(true_pairs_set)}) in {time.time()-t_gt:.2f}s", flush=True)
del cand_set, true_pairs_set
gc.collect()

# Filter s1_store only for matched S1 IDs
matched_s1 = set(all_pairs['s1_id'].unique())
mask_s1 = s1_store['entity_id'].isin(matched_s1)
s1_lookup = dict(zip(s1_store['entity_id'][mask_s1], zip(s1_store['norm_name'][mask_s1], s1_store['norm_address'][mask_s1], s1_store['norm_country'][mask_s1])))
del s1_store, matched_s1, mask_s1
gc.collect()

# STEP 5: Feature calculation
print("\n--- STEP 5: FEATURE CALCULATION ---", flush=True)
t_feat = time.time()
s1_p = all_pairs['s1_id'].values
cand_p = all_pairs['cand_id'].values
del all_pairs
gc.collect()

scores = []
n_pairs = len(s1_p)
for i in range(n_pairs):
    sid = s1_p[i]
    cid = cand_p[i]
    
    n1, a1, c1 = s1_lookup[sid]
    n2, a2, c2 = target_lookup[cid]
    
    if c1 != c2 or not c1:
        scores.append(0.0)
        continue
        
    n_exact = (n1 == n2)
    a_exact = (a1 == a2)
    
    n_rat = 1.0 if n_exact else fuzz.ratio(n1, n2) / 100.0
    n_tok = 1.0 if n_exact else fuzz.token_sort_ratio(n1, n2) / 100.0
    n_score = max(n_rat, n_tok)
    
    a_rat = 1.0 if a_exact else fuzz.ratio(a1, a2) / 100.0
    a_tok = 1.0 if a_exact else fuzz.token_sort_ratio(a1, a2) / 100.0
    a_score = max(a_rat, a_tok)
    
    # Combined score
    comb = 0.55 * n_score + 0.45 * a_score
    if n_score >= 0.90 and a_score >= 0.50:
        comb = max(comb, 0.70 * n_score + 0.30 * a_score)
    if a_score >= 0.90 and n_score >= 0.50:
        comb = max(comb, 0.30 * n_score + 0.70 * a_score)
        
    scores.append(comb)

del s1_lookup, target_lookup
gc.collect()

scores = np.array(scores, dtype=np.float32)
print(f"Feature calculation completed for {n_pairs} pairs in {time.time()-t_feat:.2f}s, RAM avail: {psutil.virtual_memory().available / 1024**3:.2f} GB", flush=True)

# STEP 6: Threshold validation
print("\n--- STEP 6: THRESHOLD VALIDATION ---", flush=True)
thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

gt_s1 = gt['source1_entity_id'].values
gt_matched = gt['matched_entity_ids'].fillna('').values
n_gt = len(gt)
gt_sets = [set(m.split(',')) if m else set() for m in gt_matched]
del gt
gc.collect()

best_thr = None
best_f05 = -1.0
best_results = None

for thr in thresholds:
    t_th = time.time()
    mask = scores >= thr
    pass_s1 = s1_p[mask]
    pass_cand = cand_p[mask]
    
    preds_dict = {}
    for sid, cid in zip(pass_s1, pass_cand):
        if sid not in preds_dict:
            preds_dict[sid] = set()
        preds_dict[sid].add(cid)
        
    total_prec = 0.0
    total_rec = 0.0
    total_f05 = 0.0
    sing_corr = 0
    sing_tot = 0
    
    for idx, sid in enumerate(gt_s1):
        true_set = gt_sets[idx]
        pred_set = preds_dict.get(sid, set())
        
        p_len = len(pred_set)
        t_len = len(true_set)
        
        inter = len(pred_set.intersection(true_set)) if p_len > 0 and t_len > 0 else 0
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
                
    m_prec = total_prec / n_gt
    m_rec = total_rec / n_gt
    m_f05 = total_f05 / n_gt
    sing_acc = sing_corr / sing_tot if sing_tot > 0 else 0.0
    
    print(f"Threshold {thr:.2f} -> Precision: {m_prec:.4f}, Recall: {m_rec:.4f}, Macro F0.5: {m_f05:.4f}, Singletons: {sing_acc:.4f} ({sing_corr}/{sing_tot}) [took {time.time()-t_th:.2f}s]", flush=True)
    
    if m_f05 > best_f05:
        best_f05 = m_f05
        best_thr = thr
        best_results = (m_prec, m_rec, m_f05, sing_acc)

print(f"\n========================================", flush=True)
print(f"TRAIN VALIDATION BEST THRESHOLD: {best_thr:.2f}", flush=True)
print(f"Macro Precision: {best_results[0]:.4f}", flush=True)
print(f"Macro Recall:    {best_results[1]:.4f}", flush=True)
print(f"Macro F0.5:      {best_results[2]:.4f}", flush=True)
print(f"Singleton Acc:   {best_results[3]:.4f}", flush=True)
print(f"========================================", flush=True)
