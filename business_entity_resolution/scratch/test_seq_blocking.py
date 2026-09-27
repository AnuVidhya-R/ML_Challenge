import pandas as pd
import numpy as np
import time
import gc

def normalize_name(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    suffixes = r"\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|plc|sarl|sa|gmbh)\b"
    s = s.str.replace(suffixes, ' ', regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

def normalize_address(s: pd.Series) -> pd.Series:
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

t0 = time.time()
print("Reading and normalizing S1...", flush=True)
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', quoting=3, dtype=str)
s1['norm_name'] = normalize_name(s1['business_name'])
s1['norm_address'] = normalize_address(s1['business_address'])
s1['norm_country'] = s1['country'].fillna('').astype(str).str.lower().str.strip()
s1['key_nc'] = s1['norm_name'] + "||" + s1['norm_country']
s1['key_ac'] = s1['norm_address'] + "||" + s1['norm_country']

s1_nc = s1[s1['norm_name'].str.len() >= 3][['entity_id', 'key_nc']].rename(columns={'entity_id': 's1_id'})
s1_ac = s1[s1['norm_address'].str.len() >= 5][['entity_id', 'key_ac']].rename(columns={'entity_id': 's1_id'})
print(f"S1 ready in {time.time()-t0:.2f}s", flush=True)

def block_against_source(s_path, max_block_size=100):
    t_start = time.time()
    fname = s_path.split('/')[-1]
    print(f"\nProcessing {fname}...", flush=True)
    df = pd.read_csv(s_path, sep='\t', quoting=3, dtype=str)
    n_name = normalize_name(df['business_name'])
    n_addr = normalize_address(df['business_address'])
    n_ctry = df['country'].fillna('').astype(str).str.lower().str.strip()
    df['key_nc'] = n_name + "||" + n_ctry
    df['key_ac'] = n_addr + "||" + n_ctry
    
    # 1. key_nc
    s_nc = df[n_name.str.len() >= 3][['entity_id', 'key_nc']].rename(columns={'entity_id': 'cand_id'})
    nc_counts = s_nc['key_nc'].value_counts()
    oversized_nc = set(nc_counts[nc_counts > max_block_size].index)
    valid_nc = s_nc[~s_nc['key_nc'].isin(oversized_nc)]
    
    # Merge nc
    p_nc = pd.merge(s1_nc, valid_nc, on='key_nc', how='inner')[['s1_id', 'cand_id']]
    print(f"  key_nc pairs: {len(p_nc)}, oversized blocks: {len(oversized_nc)}")
    
    # 2. key_ac
    s_ac = df[n_addr.str.len() >= 5][['entity_id', 'key_ac']].rename(columns={'entity_id': 'cand_id'})
    ac_counts = s_ac['key_ac'].value_counts()
    oversized_ac = set(ac_counts[ac_counts > max_block_size].index)
    valid_ac = s_ac[~s_ac['key_ac'].isin(oversized_ac)]
    
    # Merge ac
    p_ac = pd.merge(s1_ac, valid_ac, on='key_ac', how='inner')[['s1_id', 'cand_id']]
    print(f"  key_ac pairs: {len(p_ac)}, oversized blocks: {len(oversized_ac)}")
    
    pairs = pd.concat([p_nc, p_ac], ignore_index=True).drop_duplicates()
    print(f"  Total unique pairs for {fname}: {len(pairs)} in {time.time()-t_start:.2f}s")
    
    del df, s_nc, s_ac, valid_nc, valid_ac, p_nc, p_ac
    gc.collect()
    return pairs, len(oversized_nc) + len(oversized_ac)

pairs_s2, over_s2 = block_against_source('dataset/train/train_source2.tsv', max_block_size=100)
pairs_s3, over_s3 = block_against_source('dataset/train/train_source3.tsv', max_block_size=100)

all_pairs = pd.concat([pairs_s2, pairs_s3], ignore_index=True).drop_duplicates()
print(f"\nTotal combined unique candidate pairs: {len(all_pairs)}", flush=True)

# Candidate recall on Ground Truth
print("\nLoading Ground Truth to calculate Candidate Recall...", flush=True)
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', quoting=3, dtype=str)
true_pairs_set = set()
for s1_id, m_str in zip(gt['source1_entity_id'], gt['matched_entity_ids'].fillna('')):
    if m_str:
        for cid in m_str.split(','):
            true_pairs_set.add((s1_id, cid.strip()))

total_true = len(true_pairs_set)
cand_set = set(zip(all_pairs['s1_id'], all_pairs['cand_id']))
found = len(true_pairs_set.intersection(cand_set))
print(f"Candidate Recall: {found / total_true:.4f} ({found} / {total_true})", flush=True)
