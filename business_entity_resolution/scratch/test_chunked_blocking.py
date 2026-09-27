import pandas as pd
import numpy as np
import time
import gc
import psutil

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
print("Loading and normalizing S1...", flush=True)
df_s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', quoting=3, dtype=str)
n_name1 = normalize_name(df_s1['business_name'])
n_addr1 = normalize_address(df_s1['business_address'])
ctry1 = df_s1['country'].fillna('').astype(str).str.lower().str.strip()

key_nc1 = n_name1 + "||" + ctry1
key_ac1 = n_addr1 + "||" + ctry1

# Only keep valid keys
s1_nc = pd.DataFrame({'s1_id': df_s1['entity_id'][n_name1.str.len() >= 3], 'key_nc': key_nc1[n_name1.str.len() >= 3]})
s1_ac = pd.DataFrame({'s1_id': df_s1['entity_id'][n_addr1.str.len() >= 5], 'key_ac': key_ac1[n_addr1.str.len() >= 5]})

del df_s1, n_name1, n_addr1, ctry1, key_nc1, key_ac1
gc.collect()

print(f"S1 blocking tables ready in {time.time()-t0:.2f}s, RAM avail: {psutil.virtual_memory().available / 1024**3:.2f} GB", flush=True)

# First pass on S2 and S3: count key occurrences for block size measurement & safety control
print("\nPass 1: Measuring block sizes across S2 and S3...", flush=True)
t_p1 = time.time()
counts_nc = {}
counts_ac = {}

for path in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    print(f"  Counting keys in {path}...", flush=True)
    for chunk in pd.read_csv(path, sep='\t', quoting=3, dtype=str, chunksize=1000000):
        nn = normalize_name(chunk['business_name'])
        na = normalize_address(chunk['business_address'])
        cc = chunk['country'].fillna('').astype(str).str.lower().str.strip()
        
        kn = (nn + "||" + cc)[nn.str.len() >= 3]
        ka = (na + "||" + cc)[na.str.len() >= 5]
        
        for k, v in kn.value_counts().items():
            counts_nc[k] = counts_nc.get(k, 0) + v
        for k, v in ka.value_counts().items():
            counts_ac[k] = counts_ac.get(k, 0) + v

print(f"Pass 1 finished in {time.time()-t_p1:.2f}s", flush=True)

# Measure block statistics with max_block_size = 100
MAX_BLOCK_SIZE = 100
s1_nc_keys = set(s1_nc['key_nc'])
s1_ac_keys = set(s1_ac['key_ac'])

common_nc = [counts_nc[k] for k in s1_nc_keys if k in counts_nc]
oversized_nc = sum(1 for c in common_nc if c > MAX_BLOCK_SIZE)
valid_nc_counts = [c for c in common_nc if c <= MAX_BLOCK_SIZE]

common_ac = [counts_ac[k] for k in s1_ac_keys if k in counts_ac]
oversized_ac = sum(1 for c in common_ac if c > MAX_BLOCK_SIZE)
valid_ac_counts = [c for c in common_ac if c <= MAX_BLOCK_SIZE]

all_valid_counts = valid_nc_counts + valid_ac_counts
largest_block = max(all_valid_counts) if all_valid_counts else 0
avg_block_size = np.mean(all_valid_counts) if all_valid_counts else 0
total_oversized = oversized_nc + oversized_ac

print("\n--- BLOCK STATISTICS ---", flush=True)
print(f"Total common blocks (nc + ac): {len(common_nc) + len(common_ac)}")
print(f"Oversized blocks (> {MAX_BLOCK_SIZE}): {total_oversized} (nc: {oversized_nc}, ac: {oversized_ac})")
print(f"Largest valid block size: {largest_block}")
print(f"Average valid block size: {avg_block_size:.2f}")

# Filter out oversized keys
oversized_nc_set = set(k for k in s1_nc_keys if counts_nc.get(k, 0) > MAX_BLOCK_SIZE)
oversized_ac_set = set(k for k in s1_ac_keys if counts_ac.get(k, 0) > MAX_BLOCK_SIZE)

del counts_nc, counts_ac, common_nc, common_ac, valid_nc_counts, valid_ac_counts, all_valid_counts
gc.collect()

# Filter S1
s1_nc = s1_nc[~s1_nc['key_nc'].isin(oversized_nc_set)]
s1_ac = s1_ac[~s1_ac['key_ac'].isin(oversized_ac_set)]
gc.collect()

# Pass 2: Merge in chunks
print("\nPass 2: Merging candidate pairs in chunks...", flush=True)
t_p2 = time.time()
pair_list = []

for path in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    print(f"  Merging chunks from {path}...", flush=True)
    for chunk in pd.read_csv(path, sep='\t', quoting=3, dtype=str, chunksize=1000000):
        nn = normalize_name(chunk['business_name'])
        na = normalize_address(chunk['business_address'])
        cc = chunk['country'].fillna('').astype(str).str.lower().str.strip()
        
        # nc
        kn = (nn + "||" + cc)
        mask_n = (nn.str.len() >= 3) & (~kn.isin(oversized_nc_set))
        chunk_nc = pd.DataFrame({'cand_id': chunk['entity_id'][mask_n], 'key_nc': kn[mask_n]})
        m_nc = pd.merge(s1_nc, chunk_nc, on='key_nc', how='inner')[['s1_id', 'cand_id']]
        if len(m_nc) > 0:
            pair_list.append(m_nc)
            
        # ac
        ka = (na + "||" + cc)
        mask_a = (na.str.len() >= 5) & (~ka.isin(oversized_ac_set))
        chunk_ac = pd.DataFrame({'cand_id': chunk['entity_id'][mask_a], 'key_ac': ka[mask_a]})
        m_ac = pd.merge(s1_ac, chunk_ac, on='key_ac', how='inner')[['s1_id', 'cand_id']]
        if len(m_ac) > 0:
            pair_list.append(m_ac)

del s1_nc, s1_ac, oversized_nc_set, oversized_ac_set
gc.collect()

print(f"Concatenating and deduplicating candidate pairs...", flush=True)
all_pairs = pd.concat(pair_list, ignore_index=True).drop_duplicates()
del pair_list
gc.collect()

print(f"Total candidate pairs generated: {len(all_pairs)} in {time.time()-t_p2:.2f}s", flush=True)
print(f"RAM available: {psutil.virtual_memory().available / 1024**3:.2f} GB", flush=True)

# Candidate Recall against Ground Truth
print("\nLoading Ground Truth to calculate Candidate Recall...", flush=True)
t_gt = time.time()
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', quoting=3, dtype=str)
true_pairs_set = set()
for s1_id, m_str in zip(gt['source1_entity_id'], gt['matched_entity_ids'].fillna('')):
    if m_str:
        for cid in m_str.split(','):
            true_pairs_set.add((s1_id, cid.strip()))

total_true = len(true_pairs_set)
cand_set = set(zip(all_pairs['s1_id'], all_pairs['cand_id']))
found = len(true_pairs_set.intersection(cand_set))
cand_recall = found / total_true if total_true > 0 else 0.0
print(f"Candidate Recall: {cand_recall:.4f} ({found} / {total_true}) in {time.time()-t_gt:.2f}s", flush=True)
