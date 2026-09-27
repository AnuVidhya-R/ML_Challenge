import os
import pandas as pd
import numpy as np

DATA_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\dataset\train"

def norm_name(s):
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    suffixes = r'\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|plc|sarl|sa|gmbh)\b'
    s = s.str.replace(suffixes, ' ', regex=True)
    return s.str.replace(r'\s+', '', regex=True)

def norm_addr(s):
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
    return s.str.replace(r'\s+', '', regex=True)

def norm_country(s):
    return s.fillna('').astype(str).str.upper().str.strip()

print("Loading data...")
s1 = pd.read_csv(os.path.join(DATA_DIR, "train_source1.tsv"), sep='\t', quoting=3, dtype=str, nrows=500_000)
s2 = pd.read_csv(os.path.join(DATA_DIR, "train_source2.tsv"), sep='\t', quoting=3, dtype=str, nrows=500_000)
gt = pd.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), sep='\t', quoting=3, dtype=str)

# Preprocess
for df in [s1, s2]:
    df['nn'] = norm_name(df['business_name'])
    df['na'] = norm_addr(df['business_address'])
    df['nc'] = norm_country(df['country'])

# Build GT set
true_pairs = set()
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    m_str = row['matched_entity_ids']
    if pd.notna(m_str) and m_str:
        for c in m_str.split(','):
            true_pairs.add((s1_id, c))
total_true_s2 = sum(1 for _, c in true_pairs if c.startswith('S2'))

def evaluate_rule(name, rule_df):
    pairs = set(zip(rule_df['s1_id'], rule_df['cand_id']))
    n_pairs = len(pairs)
    tp = len(pairs & true_pairs)
    fp = n_pairs - tp
    precision = tp / n_pairs if n_pairs > 0 else 0
    recall = tp / total_true_s2 if total_true_s2 > 0 else 0
    print(f"Rule: {name}")
    print(f"  Pairs: {n_pairs:,}")
    print(f"  TP: {tp:,} | FP: {fp:,}")
    print(f"  Precision: {precision:.4f}\n")

# Rule 1: Exact Name (len>=4) + Country
s1_valid = s1[s1['nn'].str.len() >= 4].copy()
s2_valid = s2[s2['nn'].str.len() >= 4].copy()
s1_valid['key'] = s1_valid['nn'] + '||' + s1_valid['nc']
s2_valid['key'] = s2_valid['nn'] + '||' + s2_valid['nc']
m1 = pd.merge(s1_valid[['entity_id', 'key']], s2_valid[['entity_id', 'key']], on='key')
m1.columns = ['s1_id', 'key', 'cand_id']
evaluate_rule("Exact Name + Country", m1)

# Rule 2: Exact Address (len>=6) + Country
s1_valid = s1[s1['na'].str.len() >= 6].copy()
s2_valid = s2[s2['na'].str.len() >= 6].copy()
s1_valid['key'] = s1_valid['na'] + '||' + s1_valid['nc']
s2_valid['key'] = s2_valid['na'] + '||' + s2_valid['nc']
m2 = pd.merge(s1_valid[['entity_id', 'key']], s2_valid[['entity_id', 'key']], on='key')
m2.columns = ['s1_id', 'key', 'cand_id']
evaluate_rule("Exact Address + Country", m2)

# Rule 3: Exact Name + Exact Address (no country)
s1_valid = s1[(s1['nn'].str.len() >= 4) & (s1['na'].str.len() >= 6)].copy()
s2_valid = s2[(s2['nn'].str.len() >= 4) & (s2['na'].str.len() >= 6)].copy()
s1_valid['key'] = s1_valid['nn'] + '||' + s1_valid['na']
s2_valid['key'] = s2_valid['nn'] + '||' + s2_valid['na']
m3 = pd.merge(s1_valid[['entity_id', 'key']], s2_valid[['entity_id', 'key']], on='key')
m3.columns = ['s1_id', 'key', 'cand_id']
evaluate_rule("Exact Name + Exact Address", m3)
