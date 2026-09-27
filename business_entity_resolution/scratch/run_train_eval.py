import os
import sys
import time
import gc
import psutil
import pandas as pd
import numpy as np
import rapidfuzz.fuzz as fuzz

print("=== STARTING SUBMISSION 2 TRAIN VALIDATION ===", flush=True)
BASE_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution"
DATA_DIR = os.path.join(BASE_DIR, "dataset")

t_start = time.time()

# ---------------------------------------------------------
# STEP 1 & 2: Loading data & Vectorized Normalization (Train S1)
# ---------------------------------------------------------
print("\n--- STEP 1 & 2: Loading & Normalizing Train S1 ---", flush=True)
t0 = time.time()

def norm_name(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    suffixes = r"\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|plc|sarl|sa|gmbh)\b"
    s = s.str.replace(suffixes, ' ', regex=True)
    return s.str.replace(r'\s+', '', regex=True)

def norm_addr(s: pd.Series) -> pd.Series:
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
    return s.str.replace(r'\s+', '', regex=True)

df_s1 = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source1.tsv'), sep='\t', quoting=3, dtype=str)
n_name1 = norm_name(df_s1['business_name'])
n_addr1 = norm_addr(df_s1['business_address'])
c_ctry1 = df_s1['country'].fillna('').astype(str).str.upper().str.strip()

key_nc1 = n_name1 + "||" + c_ctry1
key_ac1 = n_addr1 + "||" + c_ctry1

valid_nc1 = (n_name1.str.len() >= 4) & (c_ctry1 != '')
valid_ac1 = (n_addr1.str.len() >= 6) & (c_ctry1 != '')

s1_nc = pd.DataFrame({'s1_id': df_s1['entity_id'][valid_nc1], 'key_nc': key_nc1[valid_nc1]})
s1_ac = pd.DataFrame({'s1_id': df_s1['entity_id'][valid_ac1], 'key_ac': key_ac1[valid_ac1]})

s1_nc_cnts = s1_nc['key_nc'].value_counts()
s1_ac_cnts = s1_ac['key_ac'].value_counts()

# Compact dictionary for S1 strings
s1_lookup = dict(zip(df_s1['entity_id'], zip(n_name1, n_addr1, c_ctry1)))

del df_s1, n_name1, n_addr1, c_ctry1, key_nc1, key_ac1, valid_nc1, valid_ac1
gc.collect()

print(f"STEP 1 & 2 completed in {time.time()-t0:.2f}s | RAM avail: {psutil.virtual_memory().available / 1024**3:.2f} GB", flush=True)

# ---------------------------------------------------------
# STEP 3: Candidate Blocking & STEP 4: Candidate Statistics
# ---------------------------------------------------------
print("\n--- STEP 3: Candidate Blocking ---", flush=True)
t_block = time.time()
MAX_BLOCK_PAIRS = 50

def match_target(fname, s1_nc_df, s1_ac_df, s1_nc_cnts, s1_ac_cnts):
    t_sub = time.time()
    fpath = os.path.join(DATA_DIR, 'train', fname)
    print(f"  Loading & indexing {fname}...", flush=True)
    df = pd.read_csv(fpath, sep='\t', quoting=3, dtype=str)
    
    n_name = norm_name(df['business_name'])
    n_addr = norm_addr(df['business_address'])
    c_ctry = df['country'].fillna('').astype(str).str.upper().str.strip()
    
    key_nc = n_name + "||" + c_ctry
    key_ac = n_addr + "||" + c_ctry
    
    # Name match
    cand_nc = pd.DataFrame({'cand_id': df['entity_id'][(n_name.str.len() >= 4) & (c_ctry != '')], 'key_nc': key_nc[(n_name.str.len() >= 4) & (c_ctry != '')]})
    nc_counts = cand_nc['key_nc'].value_counts()
    
    common_nc = set(s1_nc_cnts.index).intersection(set(nc_counts.index))
    oversized_nc = set()
    valid_nc_sizes = []
    for k in common_nc:
        prod = s1_nc_cnts[k] * nc_counts[k]
        if prod > MAX_BLOCK_PAIRS or nc_counts[k] > 20:
            oversized_nc.add(k)
        else:
            valid_nc_sizes.append(prod)
            
    cand_nc_valid = cand_nc[~cand_nc['key_nc'].isin(oversized_nc)]
    s1_nc_valid = s1_nc_df[~s1_nc_df['key_nc'].isin(oversized_nc)]
    m_nc = pd.merge(s1_nc_valid, cand_nc_valid, on='key_nc', how='inner')[['s1_id', 'cand_id']]
    del cand_nc, cand_nc_valid, s1_nc_valid
    gc.collect()
    
    # Address match
    cand_ac = pd.DataFrame({'cand_id': df['entity_id'][(n_addr.str.len() >= 6) & (c_ctry != '')], 'key_ac': key_ac[(n_addr.str.len() >= 6) & (c_ctry != '')]})
    ac_counts = cand_ac['key_ac'].value_counts()
    
    common_ac = set(s1_ac_cnts.index).intersection(set(ac_counts.index))
    oversized_ac = set()
    valid_ac_sizes = []
    for k in common_ac:
        prod = s1_ac_cnts[k] * ac_counts[k]
        if prod > MAX_BLOCK_PAIRS or ac_counts[k] > 20:
            oversized_ac.add(k)
        else:
            valid_ac_sizes.append(prod)
            
    cand_ac_valid = cand_ac[~cand_ac['key_ac'].isin(oversized_ac)]
    s1_ac_valid = s1_ac_df[~s1_ac_df['key_ac'].isin(oversized_ac)]
    m_ac = pd.merge(s1_ac_valid, cand_ac_valid, on='key_ac', how='inner')[['s1_id', 'cand_id']]
    del cand_ac, cand_ac_valid, s1_ac_valid
    gc.collect()
    
    pairs = pd.concat([m_nc, m_ac], ignore_index=True).drop_duplicates()
    del m_nc, m_ac
    gc.collect()
    
    # Keep lookup only for matched candidates
    matched_cands = set(pairs['cand_id'].unique())
    mask = df['entity_id'].isin(matched_cands)
    target_lookup = dict(zip(df['entity_id'][mask], zip(n_name[mask], n_addr[mask], c_ctry[mask])))
    
    del df, n_name, n_addr, c_ctry, key_nc, key_ac, matched_cands, mask
    gc.collect()
    
    print(f"  {fname} done in {time.time()-t_sub:.2f}s | Pairs: {len(pairs)} | Oversized: {len(oversized_nc)+len(oversized_ac)}", flush=True)
    return pairs, target_lookup, len(oversized_nc)+len(oversized_ac), valid_nc_sizes + valid_ac_sizes

pairs_s2, lookup_s2, over2, sizes2 = match_target('train_source2.tsv', s1_nc, s1_ac, s1_nc_cnts, s1_ac_cnts)
pairs_s3, lookup_s3, over3, sizes3 = match_target('train_source3.tsv', s1_nc, s1_ac, s1_nc_cnts, s1_ac_cnts)
del s1_nc, s1_ac, s1_nc_cnts, s1_ac_cnts
gc.collect()

all_pairs = pd.concat([pairs_s2, pairs_s3], ignore_index=True).drop_duplicates()
del pairs_s2, pairs_s3
gc.collect()

target_lookup = {**lookup_s2, **lookup_s3}
del lookup_s2, lookup_s3
gc.collect()

# Step 4: Statistics
all_sizes = sizes2 + sizes3
total_candidate_pairs = len(all_pairs)
largest_block = max(all_sizes) if all_sizes else 0
avg_block_size = np.mean(all_sizes) if all_sizes else 0
total_oversized = over2 + over3
del sizes2, sizes3, all_sizes
gc.collect()

print(f"\n--- STEP 4: Candidate Statistics ---", flush=True)
print(f"Total candidate pairs: {total_candidate_pairs}")
print(f"Largest block: {largest_block}")
print(f"Average block size: {avg_block_size:.2f}")
print(f"Number of oversized blocks: {total_oversized}")
print(f"Candidate blocking completed in {time.time()-t_block:.2f}s | RAM avail: {psutil.virtual_memory().available / 1024**3:.2f} GB", flush=True)

# Safety check
if total_candidate_pairs > 10000000:
    print(f"ERROR: Candidate pairs count {total_candidate_pairs} exceeds safety limit! Halting.", flush=True)
    sys.exit(1)

# Candidate Recall against Ground Truth
print("\n--- Evaluating Candidate Recall ---", flush=True)
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
print(f"Candidate count: {total_candidate_pairs}")
print(f"Candidate recall: {cand_recall:.4f} ({found_true}/{len(true_pairs_set)}) in {time.time()-t_gt:.2f}s", flush=True)
del cand_set, true_pairs_set
gc.collect()

# ---------------------------------------------------------
# STEP 5: Feature Calculation
# ---------------------------------------------------------
print("\n--- STEP 5: Feature Calculation (rapidfuzz) ---", flush=True)
t_feat = time.time()

s1_p = all_pairs['s1_id'].values
cand_p = all_pairs['cand_id'].values
del all_pairs
gc.collect()

n_pairs = len(s1_p)
scores = []

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
    
    # Combined score concept:
    # strong name + strong address OR very strong name + reasonable address OR very strong address + reasonable name
    comb = 0.55 * n_score + 0.45 * a_score
    if n_score >= 0.90 and a_score >= 0.50:
        comb = max(comb, 0.70 * n_score + 0.30 * a_score)
    if a_score >= 0.90 and n_score >= 0.50:
        comb = max(comb, 0.30 * n_score + 0.70 * a_score)
        
    scores.append(comb)

del s1_lookup, target_lookup
gc.collect()

scores = np.array(scores, dtype=np.float32)
print(f"STEP 5 completed in {time.time()-t_feat:.2f}s | RAM avail: {psutil.virtual_memory().available / 1024**3:.2f} GB", flush=True)

# ---------------------------------------------------------
# STEP 6: Threshold Validation
# ---------------------------------------------------------
print("\n--- STEP 6: Threshold Validation ---", flush=True)
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
print(f"BEST THRESHOLD: {best_thr:.2f}", flush=True)
print(f"Macro Precision: {best_results[0]:.4f}", flush=True)
print(f"Macro Recall:    {best_results[1]:.4f}", flush=True)
print(f"Macro F0.5:      {best_results[2]:.4f}", flush=True)
print(f"Singleton Acc:   {best_results[3]:.4f}", flush=True)
print(f"Total Validation Runtime: {time.time()-t_start:.2f}s", flush=True)
print(f"========================================", flush=True)
