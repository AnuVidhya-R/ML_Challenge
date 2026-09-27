# -*- coding: utf-8 -*-
import pandas as pd
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

BASE = r'D:\Working_project\ML Challenge\business_entity_resolution'

s1 = pd.read_csv(os.path.join(BASE, 'dataset/train/train_source1.tsv'), sep='\t', quoting=3, dtype=str, nrows=10)
print('=== S1 COLUMNS:', list(s1.columns))
print(s1.head(3).to_string())
print(f'S1 entity_id prefix: {s1["entity_id"].iloc[0][:3]}')

s2 = pd.read_csv(os.path.join(BASE, 'dataset/train/train_source2.tsv'), sep='\t', quoting=3, dtype=str, nrows=10)
print('\n=== S2 COLUMNS:', list(s2.columns))
print(s2.head(3).to_string())
print(f'S2 entity_id prefix: {s2["entity_id"].iloc[0][:3]}')

s3 = pd.read_csv(os.path.join(BASE, 'dataset/train/train_source3.tsv'), sep='\t', quoting=3, dtype=str, nrows=10)
print('\n=== S3 COLUMNS:', list(s3.columns))
print(s3.head(3).to_string())
print(f'S3 entity_id prefix: {s3["entity_id"].iloc[0][:3]}')

gt = pd.read_csv(os.path.join(BASE, 'dataset/train/train_ground_truth.tsv'), sep='\t', quoting=3, dtype=str, nrows=20)
print('\n=== GT COLUMNS:', list(gt.columns))
print(gt.head(5).to_string())

# Check GT stats
gt_full = pd.read_csv(os.path.join(BASE, 'dataset/train/train_ground_truth.tsv'), sep='\t', quoting=3, dtype=str)
print(f'\nGT total rows: {len(gt_full):,}')
n_empty = gt_full['matched_entity_ids'].isna().sum() + (gt_full['matched_entity_ids'] == '').sum()
print(f'GT empty matches (singletons): {n_empty:,}')
n_nonempty = len(gt_full) - n_empty
print(f'GT non-empty matches: {n_nonempty:,}')

# Count multi-match
multi = 0
for v in gt_full['matched_entity_ids'].dropna():
    if ',' in str(v):
        multi += 1
print(f'GT multi-match rows: {multi:,}')

# Sample matched
print('\nSample matched entries:')
matched = gt_full[gt_full['matched_entity_ids'].notna() & (gt_full['matched_entity_ids'] != '')]
for i, row in matched.head(10).iterrows():
    print(f'  {row["source1_entity_id"]}: {row["matched_entity_ids"]}')

# Check S1/S2/S3 row counts
s1_full_count = sum(1 for _ in open(os.path.join(BASE, 'dataset/train/train_source1.tsv'), encoding='utf-8')) - 1
s2_full_count = sum(1 for _ in open(os.path.join(BASE, 'dataset/train/train_source2.tsv'), encoding='utf-8')) - 1
s3_full_count = sum(1 for _ in open(os.path.join(BASE, 'dataset/train/train_source3.tsv'), encoding='utf-8')) - 1
print(f'\nTrain S1 rows: {s1_full_count:,}')
print(f'Train S2 rows: {s2_full_count:,}')
print(f'Train S3 rows: {s3_full_count:,}')

# Test counts
ts1 = sum(1 for _ in open(os.path.join(BASE, 'dataset/test/test_source1.tsv'), encoding='utf-8')) - 1
ts2 = sum(1 for _ in open(os.path.join(BASE, 'dataset/test/test_source2.tsv'), encoding='utf-8')) - 1
ts3 = sum(1 for _ in open(os.path.join(BASE, 'dataset/test/test_source3.tsv'), encoding='utf-8')) - 1
print(f'Test S1 rows: {ts1:,}')
print(f'Test S2 rows: {ts2:,}')
print(f'Test S3 rows: {ts3:,}')
