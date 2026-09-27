import pandas as pd
import numpy as np
import os
import gc

base_dir = r"d:\Working_project\ML Challenge\business_entity_resolution\dataset"

def inspect_file(file_path):
    print(f"\n--- Reading {os.path.basename(file_path)} ---")
    df = pd.read_csv(file_path, sep='\t', quoting=3)
    print(f"Row count: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("Missing values:")
    print(df.isnull().sum())
    
    id_col = df.columns[0]
    num_dupes = df[id_col].duplicated().sum()
    print(f"Duplicate IDs in '{id_col}': {num_dupes}")
    
    if 'country' in df.columns:
        print("Top 5 Countries:")
        print(df['country'].value_counts(dropna=False).head(5))
        
    if 'business_name' in df.columns:
        lens = df['business_name'].fillna('').astype(str).str.len()
        print(f"Business Name len - Min: {lens.min()}, Max: {lens.max()}, Mean: {lens.mean():.2f}, Median: {lens.median()}")
        
    if 'business_address' in df.columns:
        lens = df['business_address'].fillna('').astype(str).str.len()
        print(f"Business Address len - Min: {lens.min()}, Max: {lens.max()}, Mean: {lens.mean():.2f}, Median: {lens.median()}")
    
    del df
    gc.collect()

print("=== STEP 1: TRAIN FILES ===")
inspect_file(os.path.join(base_dir, 'train', 'train_source1.tsv'))
inspect_file(os.path.join(base_dir, 'train', 'train_source2.tsv'))
inspect_file(os.path.join(base_dir, 'train', 'train_source3.tsv'))

print("\n=== GROUND TRUTH STATS ===")
gt_path = os.path.join(base_dir, 'train', 'train_ground_truth.tsv')
gt = pd.read_csv(gt_path, sep='\t', quoting=3)
print(f"Total S1 in Ground Truth: {len(gt)}")
print(f"Null matched_entity_ids count: {gt['matched_entity_ids'].isnull().sum()}")

# Process matched_entity_ids chunk-wise / efficiently
def count_matches(val):
    if pd.isna(val) or not str(val).strip():
        return 0
    return len(str(val).split(','))

match_counts = gt['matched_entity_ids'].apply(count_matches)
singletons = (match_counts == 0).sum()
print(f"Singletons (0 matches): {singletons} ({singletons / len(gt) * 100:.2f}%)")
print(f"1 match: {(match_counts == 1).sum()} ({(match_counts == 1).mean() * 100:.2f}%)")
print(f">1 match: {(match_counts > 1).sum()} ({(match_counts > 1).mean() * 100:.2f}%)")
print(f"Max matches for a single S1: {match_counts.max()}")

del gt
gc.collect()

print("\n=== STEP 1: TEST FILES ===")
inspect_file(os.path.join(base_dir, 'test', 'test_source1.tsv'))
inspect_file(os.path.join(base_dir, 'test', 'test_source2.tsv'))
inspect_file(os.path.join(base_dir, 'test', 'test_source3.tsv'))
