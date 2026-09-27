import os
import sys
import time
import gc
import psutil
import pandas as pd
import numpy as np

DATA_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\dataset"

t0 = time.time()
print("Reading S1...", flush=True)

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

# Load S1
df_s1 = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source1.tsv'), sep='\t', quoting=3, dtype=str)
n_name1 = norm_name(df_s1['business_name'])
n_addr1 = norm_addr(df_s1['business_address'])
c_ctry1 = df_s1['country'].fillna('').astype(str).str.upper().str.strip()

key_nc1 = n_name1 + "||" + c_ctry1
key_ac1 = n_addr1 + "||" + c_ctry1

# Valid keys: name >= 4 chars, addr >= 6 chars, ctry not empty
valid_nc_mask = (n_name1.str.len() >= 4) & (c_ctry1 != '')
valid_ac_mask = (n_addr1.str.len() >= 6) & (c_ctry1 != '')

s1_nc = pd.DataFrame({'s1_id': df_s1['entity_id'][valid_nc_mask], 'key_nc': key_nc1[valid_nc_mask]})
s1_ac = pd.DataFrame({'s1_id': df_s1['entity_id'][valid_ac_mask], 'key_ac': key_ac1[valid_ac_mask]})

# S1 key frequencies
s1_nc_counts = s1_nc['key_nc'].value_counts()
s1_ac_counts = s1_ac['key_ac'].value_counts()

del df_s1, n_name1, n_addr1, c_ctry1, key_nc1, key_ac1, valid_nc_mask, valid_ac_mask
gc.collect()

print(f"S1 processed in {time.time()-t0:.2f}s, RAM avail: {psutil.virtual_memory().available / 1024**3:.2f} GB", flush=True)

MAX_BLOCK_PAIRS = 50  # Maximum Cartesian product (s1_count * target_count) allowed per block

def match_target(fname, s1_nc_df, s1_ac_df, s1_nc_cnts, s1_ac_cnts):
    t_sub = time.time()
    fpath = os.path.join(DATA_DIR, 'train', fname)
    print(f"\nProcessing {fname}...", flush=True)
    df = pd.read_csv(fpath, sep='\t', quoting=3, dtype=str)
    
    n_name = norm_name(df['business_name'])
    n_addr = norm_addr(df['business_address'])
    c_ctry = df['country'].fillna('').astype(str).str.upper().str.strip()
    
    key_nc = n_name + "||" + c_ctry
    key_ac = n_addr + "||" + c_ctry
    
    # 1. Name match with safety control
    cand_nc = pd.DataFrame({'cand_id': df['entity_id'][(n_name.str.len() >= 4) & (c_ctry != '')], 'key_nc': key_nc[(n_name.str.len() >= 4) & (c_ctry != '')]})
    nc_counts = cand_nc['key_nc'].value_counts()
    
    # Identify oversized blocks
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
    
    # 2. Address match with safety control
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
    del cand_ac, cand_ac_valid, s1_ac_valid, df, n_name, n_addr, c_ctry, key_nc, key_ac
    gc.collect()
    
    pairs = pd.concat([m_nc, m_ac], ignore_index=True).drop_duplicates()
    del m_nc, m_ac
    gc.collect()
    
    print(f"  {fname}: {len(pairs)} pairs in {time.time()-t_sub:.2f}s (oversized NC: {len(oversized_nc)}, AC: {len(oversized_ac)})", flush=True)
    return pairs, len(oversized_nc) + len(oversized_ac), valid_nc_sizes + valid_ac_sizes

pairs_s2, over2, sizes2 = match_target('train_source2.tsv', s1_nc, s1_ac, s1_nc_counts, s1_ac_counts)
pairs_s3, over3, sizes3 = match_target('train_source3.tsv', s1_nc, s1_ac, s1_nc_counts, s1_ac_counts)

del s1_nc, s1_ac, s1_nc_counts, s1_ac_counts
gc.collect()

all_pairs = pd.concat([pairs_s2, pairs_s3], ignore_index=True).drop_duplicates()
del pairs_s2, pairs_s3
gc.collect()

all_sizes = sizes2 + sizes3
largest_block = max(all_sizes) if all_sizes else 0
avg_block_size = np.mean(all_sizes) if all_sizes else 0
total_oversized = over2 + over3

print(f"\n--- CANDIDATE STATISTICS ---", flush=True)
print(f"Total candidate pairs: {len(all_pairs)}")
print(f"Largest block: {largest_block}")
print(f"Average block size: {avg_block_size:.2f}")
print(f"Number of oversized blocks: {total_oversized}")
print(f"Total candidate generation time: {time.time()-t0:.2f}s", flush=True)
print(f"RAM available: {psutil.virtual_memory().available / 1024**3:.2f} GB", flush=True)

# Candidate Recall against Ground Truth
print("\nLoading Ground Truth to calculate Candidate Recall...", flush=True)
t_gt = time.time()
gt = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv'), sep='\t', quoting=3, dtype=str)
true_pairs_set = set()
for s1_id, m_str in zip(gt['source1_entity_id'], gt['matched_entity_ids'].fillna('')):
    if m_str:
        for cid in m_str.split(','):
            true_pairs_set.add((s1_id, cid.strip()))

cand_set = set(zip(all_pairs['s1_id'], all_pairs['cand_id']))
found = len(true_pairs_set.intersection(cand_set))
cand_recall = found / len(true_pairs_set) if len(true_pairs_set) > 0 else 0.0
print(f"Candidate count: {len(all_pairs)}")
print(f"Candidate recall: {cand_recall:.4f} ({found}/{len(true_pairs_set)}) in {time.time()-t_gt:.2f}s", flush=True)
