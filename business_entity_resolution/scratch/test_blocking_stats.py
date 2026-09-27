import pandas as pd
import numpy as np
import time
import gc
import rapidfuzz.fuzz as fuzz

def normalize_series_name(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    suffixes = r"\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|plc|sarl|sa|gmbh)\b"
    s = s.str.replace(suffixes, ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def normalize_series_address(s: pd.Series) -> pd.Series:
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
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

t0 = time.time()
print("Reading S1...", flush=True)
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', quoting=3, dtype=str)
s1['norm_name'] = normalize_series_name(s1['business_name'])
s1['norm_address'] = normalize_series_address(s1['business_address'])
s1['norm_country'] = s1['country'].fillna('').astype(str).str.lower().str.strip()

s1['key_nc'] = s1['norm_name'] + "||" + s1['norm_country']
s1['key_ac'] = s1['norm_address'] + "||" + s1['norm_country']

# Valid masks
s1_nc_valid = s1['norm_name'].str.len() >= 3
s1_ac_valid = s1['norm_address'].str.len() >= 5

print(f"S1 processed in {time.time()-t0:.2f}s", flush=True)

def process_target(path):
    t_sub = time.time()
    print(f"Processing {path}...", flush=True)
    df = pd.read_csv(path, sep='\t', quoting=3, dtype=str)
    df['norm_name'] = normalize_series_name(df['business_name'])
    df['norm_address'] = normalize_series_address(df['business_address'])
    df['norm_country'] = df['country'].fillna('').astype(str).str.lower().str.strip()
    df['key_nc'] = df['norm_name'] + "||" + df['norm_country']
    df['key_ac'] = df['norm_address'] + "||" + df['norm_country']
    res = df[['entity_id', 'norm_name', 'norm_address', 'norm_country', 'key_nc', 'key_ac']].copy()
    del df
    gc.collect()
    print(f"Done in {time.time()-t_sub:.2f}s", flush=True)
    return res

s2 = process_target('dataset/train/train_source2.tsv')
s3 = process_target('dataset/train/train_source3.tsv')

s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3
gc.collect()
print(f"Total S2+S3: {len(s23)} rows", flush=True)

def analyze_and_merge(s1_sub, s23_sub, key_col, max_block_size=100):
    t_m = time.time()
    print(f"\nAnalyzing blocking key '{key_col}' (max_block_size={max_block_size})...", flush=True)
    
    s23_counts = s23_sub[key_col].value_counts()
    oversized_keys = set(s23_counts[s23_counts > max_block_size].index)
    
    s1_counts = s1_sub[key_col].value_counts()
    common_keys = set(s1_counts.index).intersection(set(s23_counts.index))
    num_blocks = len(common_keys)
    oversized_blocks = len([k for k in common_keys if k in oversized_keys])
    
    valid_s23_counts = s23_counts[(s23_counts.index.isin(common_keys)) & (~s23_counts.index.isin(oversized_keys))]
    largest_block = valid_s23_counts.max() if len(valid_s23_counts) > 0 else 0
    avg_block_size = valid_s23_counts.mean() if len(valid_s23_counts) > 0 else 0
    
    print(f"Total blocks (unique common keys): {num_blocks}")
    print(f"Oversized blocks (> {max_block_size}): {oversized_blocks}")
    print(f"Largest valid block size in S23: {largest_block}")
    print(f"Average valid block size in S23: {avg_block_size:.2f}")
    
    # Fast filtering with isin on the small set of oversized keys
    if oversized_keys:
        s1_filtered = s1_sub[~s1_sub[key_col].isin(oversized_keys)][['entity_id', key_col]].rename(columns={'entity_id': 's1_id'})
        s23_filtered = s23_sub[~s23_sub[key_col].isin(oversized_keys)][['entity_id', key_col]].rename(columns={'entity_id': 'cand_id'})
    else:
        s1_filtered = s1_sub[['entity_id', key_col]].rename(columns={'entity_id': 's1_id'})
        s23_filtered = s23_sub[['entity_id', key_col]].rename(columns={'entity_id': 'cand_id'})
        
    merged = pd.merge(s1_filtered, s23_filtered, on=key_col, how='inner')[['s1_id', 'cand_id']]
    print(f"Generated {len(merged)} candidate pairs in {time.time()-t_m:.2f}s", flush=True)
    return merged, largest_block, avg_block_size, oversized_blocks

s1_nc_sub = s1[s1_nc_valid]
s23_nc_sub = s23[s23['key_nc'].str.len() >= 3]
pairs_nc, max_b1, avg_b1, over_b1 = analyze_and_merge(s1_nc_sub, s23_nc_sub, 'key_nc', max_block_size=100)

s1_ac_sub = s1[s1_ac_valid]
s23_ac_sub = s23[s23['key_ac'].str.len() >= 5]
pairs_ac, max_b2, avg_b2, over_b2 = analyze_and_merge(s1_ac_sub, s23_ac_sub, 'key_ac', max_block_size=100)

print("\nCombining candidate pairs...", flush=True)
t_u = time.time()
all_pairs = pd.concat([pairs_nc, pairs_ac], ignore_index=True).drop_duplicates()
print(f"Total unique candidate pairs: {len(all_pairs)} in {time.time()-t_u:.2f}s", flush=True)

del pairs_nc, pairs_ac
gc.collect()

# Candidate recall on Ground Truth
print("\nLoading Ground Truth to calculate Candidate Recall...", flush=True)
t_gt = time.time()
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', quoting=3, dtype=str)

# Build set of true pairs
true_pairs_set = set()
for s1_id, m_str in zip(gt['source1_entity_id'], gt['matched_entity_ids'].fillna('')):
    if m_str:
        for cid in m_str.split(','):
            true_pairs_set.add((s1_id, cid.strip()))

total_true_pairs = len(true_pairs_set)
print(f"Total True GT pairs: {total_true_pairs} in {time.time()-t_gt:.2f}s", flush=True)

# Count how many true pairs are in all_pairs
cand_set = set(zip(all_pairs['s1_id'], all_pairs['cand_id']))
found_true = len(true_pairs_set.intersection(cand_set))
cand_recall = found_true / total_true_pairs if total_true_pairs > 0 else 0.0
print(f"Candidate Recall: {cand_recall:.4f} ({found_true}/{total_true_pairs})", flush=True)

# Now attach strings for candidate pairs to calculate fuzzy features
print("\nPreparing feature computation...", flush=True)
s1_dict = s1.set_index('entity_id')[['norm_name', 'norm_address', 'norm_country']].to_dict('index')
del s1
gc.collect()

# We only need s23 records that are in all_pairs
needed_cand_ids = set(all_pairs['cand_id'].unique())
s23_filtered = s23[s23['entity_id'].isin(needed_cand_ids)]
s23_dict = s23_filtered.set_index('entity_id')[['norm_name', 'norm_address', 'norm_country']].to_dict('index')
del s23, s23_filtered
gc.collect()

print("Calculating fuzzy similarity features for candidate pairs...", flush=True)
t_feat = time.time()

s1_ids = all_pairs['s1_id'].values
cand_ids = all_pairs['cand_id'].values
del all_pairs
gc.collect()

scores = []
batch_size = 500000
for i in range(0, len(s1_ids), batch_size):
    batch_s1 = s1_ids[i:i+batch_size]
    batch_c = cand_ids[i:i+batch_size]
    for sid, cid in zip(batch_s1, batch_c):
        s_row = s1_dict[sid]
        c_row = s23_dict[cid]
        
        n1, a1, c1 = s_row['norm_name'], s_row['norm_address'], s_row['norm_country']
        n2, a2, c2 = c_row['norm_name'], c_row['norm_address'], c_row['norm_country']
        
        if c1 != c2:
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

print(f"Features and combined scores calculated for {len(scores)} pairs in {time.time()-t_feat:.2f}s", flush=True)

scores = np.array(scores)

# Group predictions by threshold and evaluate
thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
print("\n--- SWEEPING THRESHOLDS ---", flush=True)

gt_s1 = gt['source1_entity_id'].values
gt_matched = gt['matched_entity_ids'].fillna('').values
n_gt = len(gt)

# Pre-parse ground truth sets
gt_sets = [set(m.split(',')) if m else set() for m in gt_matched]
s1_to_idx = {sid: idx for idx, sid in enumerate(gt_s1)}

for thr in thresholds:
    t_eval = time.time()
    mask = scores >= thr
    pass_s1 = s1_ids[mask]
    pass_cand = cand_ids[mask]
    
    # Build preds list
    preds_dict = {sid: set() for sid in gt_s1}
    for sid, cid in zip(pass_s1, pass_cand):
        preds_dict[sid].add(cid)
        
    total_prec = 0.0
    total_rec = 0.0
    total_f05 = 0.0
    sing_corr = 0
    sing_tot = 0
    
    for idx, sid in enumerate(gt_s1):
        true_set = gt_sets[idx]
        pred_set = preds_dict[sid]
        
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
    
    print(f"Threshold {thr:.2f}: Precision={m_prec:.4f}, Recall={m_rec:.4f}, Macro F0.5={m_f05:.4f}, Singletons={sing_acc:.4f} ({sing_corr}/{sing_tot}) in {time.time()-t_eval:.2f}s", flush=True)

