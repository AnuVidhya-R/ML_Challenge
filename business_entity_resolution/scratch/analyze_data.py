# -*- coding: utf-8 -*-
"""Quick analysis of Sub#2 weaknesses on a sample of training data."""
import pandas as pd
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

BASE = r'D:\Working_project\ML Challenge\business_entity_resolution'

# Load GT
gt = pd.read_csv(os.path.join(BASE, 'dataset/train/train_ground_truth.tsv'), sep='\t', quoting=3, dtype=str)
print(f'GT rows: {len(gt):,}')

# Check country distribution in S1
s1 = pd.read_csv(os.path.join(BASE, 'dataset/train/train_source1.tsv'), sep='\t', quoting=3, dtype=str,
                  usecols=['entity_id', 'country'])
print(f'\nS1 country distribution:')
print(s1['country'].value_counts().head(20).to_string())
print(f'\nUnique countries: {s1["country"].nunique()}')

# How many GT entries have matches vs singletons
n_singleton = gt['matched_entity_ids'].isna().sum() + (gt['matched_entity_ids'] == '').sum()
n_matched = len(gt) - n_singleton
print(f'\nSingletons: {n_singleton:,} ({100*n_singleton/len(gt):.1f}%)')
print(f'Matched: {n_matched:,} ({100*n_matched/len(gt):.1f}%)')

# Analyze match counts
match_counts = gt['matched_entity_ids'].dropna().apply(lambda x: len(x.split(',')) if x else 0)
print(f'\nMatch count distribution:')
print(match_counts.value_counts().sort_index().head(15).to_string())

# Check if matches cross S2/S3
sample = gt[gt['matched_entity_ids'].notna()].head(1000)
has_s2 = 0
has_s3 = 0
has_both = 0
for _, row in sample.iterrows():
    ids = row['matched_entity_ids'].split(',')
    s2_ids = [x for x in ids if x.startswith('S2-')]
    s3_ids = [x for x in ids if x.startswith('S3-')]
    if s2_ids: has_s2 += 1
    if s3_ids: has_s3 += 1
    if s2_ids and s3_ids: has_both += 1
print(f'\nOf 1000 matched samples:')
print(f'  Has S2 matches: {has_s2}')
print(f'  Has S3 matches: {has_s3}')
print(f'  Has both S2+S3: {has_both}')

# Check name length distribution
s1_full = pd.read_csv(os.path.join(BASE, 'dataset/train/train_source1.tsv'), sep='\t', quoting=3, dtype=str,
                       nrows=100000)
name_lens = s1_full['business_name'].fillna('').str.len()
print(f'\nS1 name length stats (sample 100k):')
print(f'  Mean: {name_lens.mean():.1f}')
print(f'  Median: {name_lens.median():.1f}')
print(f'  Min: {name_lens.min()}')
print(f'  Max: {name_lens.max()}')

addr_lens = s1_full['business_address'].fillna('').str.len()
print(f'\nS1 address length stats (sample 100k):')
print(f'  Mean: {addr_lens.mean():.1f}')
print(f'  Median: {addr_lens.median():.1f}')

# Check missing values
print(f'\nMissing values in S1 sample:')
print(f'  business_name: {s1_full["business_name"].isna().sum()}')
print(f'  business_address: {s1_full["business_address"].isna().sum()}')
print(f'  country: {s1_full["country"].isna().sum()}')

del s1_full
print('\nDone.')
