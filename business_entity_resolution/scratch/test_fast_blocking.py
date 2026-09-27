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
s1 = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source1.tsv'), sep='\t', quoting=3, dtype=str)

def fast_norm(s):
    return s.fillna('').astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

s1['key_nc'] = fast_norm(s1['business_name']) + "||" + s1['country'].fillna('').astype(str).str.upper().str.strip()
s1['key_ac'] = fast_norm(s1['business_address']) + "||" + s1['country'].fillna('').astype(str).str.upper().str.strip()

s1_nc = s1[s1['key_nc'].str.len() > 3][['entity_id', 'key_nc']].rename(columns={'entity_id': 's1_id'})
s1_ac = s1[s1['key_ac'].str.len() > 3][['entity_id', 'key_ac']].rename(columns={'entity_id': 's1_id'})

del s1
gc.collect()
print(f"S1 processed in {time.time()-t0:.2f}s, RAM avail: {psutil.virtual_memory().available / 1024**3:.2f} GB", flush=True)

def match_source(fname, s1_nc_df, s1_ac_df, max_block_size=100):
    t_sub = time.time()
    fpath = os.path.join(DATA_DIR, 'train', fname)
    print(f"Loading {fname}...", flush=True)
    df = pd.read_csv(fpath, sep='\t', quoting=3, dtype=str)
    
    df['key_nc'] = fast_norm(df['business_name']) + "||" + df['country'].fillna('').astype(str).str.upper().str.strip()
    df['key_ac'] = fast_norm(df['business_address']) + "||" + df['country'].fillna('').astype(str).str.upper().str.strip()
    
    # 1. Name merge
    cand_nc = df[df['key_nc'].str.len() > 3][['entity_id', 'key_nc']].rename(columns={'entity_id': 'cand_id'})
    nc_counts = cand_nc['key_nc'].value_counts()
    oversized_nc = set(nc_counts[nc_counts > max_block_size].index)
    
    cand_nc_valid = cand_nc[~cand_nc['key_nc'].isin(oversized_nc)]
    s1_nc_valid = s1_nc_df[~s1_nc_df['key_nc'].isin(oversized_nc)]
    m_nc = pd.merge(s1_nc_valid, cand_nc_valid, on='key_nc', how='inner')[['s1_id', 'cand_id']]
    del cand_nc, cand_nc_valid, s1_nc_valid
    gc.collect()
    
    # 2. Address merge
    cand_ac = df[df['key_ac'].str.len() > 3][['entity_id', 'key_ac']].rename(columns={'entity_id': 'cand_id'})
    ac_counts = cand_ac['key_ac'].value_counts()
    oversized_ac = set(ac_counts[ac_counts > max_block_size].index)
    
    cand_ac_valid = cand_ac[~cand_ac['key_ac'].isin(oversized_ac)]
    s1_ac_valid = s1_ac_df[~s1_ac_df['key_ac'].isin(oversized_ac)]
    m_ac = pd.merge(s1_ac_valid, cand_ac_valid, on='key_ac', how='inner')[['s1_id', 'cand_id']]
    del cand_ac, cand_ac_valid, s1_ac_valid, df
    gc.collect()
    
    pairs = pd.concat([m_nc, m_ac], ignore_index=True).drop_duplicates()
    del m_nc, m_ac
    gc.collect()
    
    print(f"Matched {fname}: {len(pairs)} pairs in {time.time()-t_sub:.2f}s, RAM avail: {psutil.virtual_memory().available / 1024**3:.2f} GB", flush=True)
    return pairs, len(oversized_nc) + len(oversized_ac)

pairs_s2, over2 = match_source('train_source2.tsv', s1_nc, s1_ac)
pairs_s3, over3 = match_source('train_source3.tsv', s1_nc, s1_ac)
del s1_nc, s1_ac
gc.collect()

all_pairs = pd.concat([pairs_s2, pairs_s3], ignore_index=True).drop_duplicates()
del pairs_s2, pairs_s3
gc.collect()

print(f"\nTotal Candidate Pairs: {len(all_pairs)} in {time.time()-t0:.2f}s", flush=True)

# Candidate Recall
gt = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv'), sep='\t', quoting=3, dtype=str)
true_pairs_set = set()
for s1_id, m_str in zip(gt['source1_entity_id'], gt['matched_entity_ids'].fillna('')):
    if m_str:
        for cid in m_str.split(','):
            true_pairs_set.add((s1_id, cid.strip()))

cand_set = set(zip(all_pairs['s1_id'], all_pairs['cand_id']))
found = len(true_pairs_set.intersection(cand_set))
print(f"Candidate Recall: {found / len(true_pairs_set):.4f} ({found}/{len(true_pairs_set)})", flush=True)
