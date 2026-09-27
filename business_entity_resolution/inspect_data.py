import os
import pandas as pd

base_dir = r"d:\Working_project\ML Challenge\business_entity_resolution\dataset"

print("--- DATASET INSPECTION ---")
for split in ['train', 'test']:
    split_dir = os.path.join(base_dir, split)
    print(f"\n=== {split.upper()} ===")
    for f in sorted(os.listdir(split_dir)):
        if f.endswith('.tsv'):
            path = os.path.join(split_dir, f)
            df = pd.read_csv(path, sep='\t')
            print(f"\nFile: {f}")
            print(f"Row count: {len(df)}")
            print(f"Columns: {list(df.columns)}")
            print(f"Missing values:\n{df.isnull().sum()}")
            if 'country' in df.columns:
                print(f"Country top 5:\n{df['country'].value_counts(dropna=False).head(5)}")
            if 'name' in df.columns:
                name_lengths = df['name'].fillna('').str.len()
                print(f"Name length - Min: {name_lengths.min()}, Max: {name_lengths.max()}, Mean: {name_lengths.mean():.2f}")
            if 'address' in df.columns:
                addr_lengths = df['address'].fillna('').str.len()
                print(f"Address length - Min: {addr_lengths.min()}, Max: {addr_lengths.max()}, Mean: {addr_lengths.mean():.2f}")
            
            # ID column check for duplicates
            id_cols = [c for c in df.columns if 'id' in c.lower() and c != 'matched_entity_ids']
            for id_c in id_cols:
                dupes = df[id_c].duplicated().sum()
                print(f"Duplicate IDs in '{id_c}': {dupes}")

# Ground truth stats
gt_path = os.path.join(base_dir, 'train', 'train_ground_truth.tsv')
if os.path.exists(gt_path):
    gt_df = pd.read_csv(gt_path, sep='\t')
    print("\n=== GROUND TRUTH STATS ===")
    print(f"Total rows in GT: {len(gt_df)}")
    
    # parse matched_entity_ids
    # could be string of space/comma separated IDs or empty/NaN
    gt_df['match_list'] = gt_df['matched_entity_ids'].fillna('').apply(lambda x: [i.strip() for i in str(x).split(',') if i.strip()])
    match_counts = gt_df['match_list'].apply(len)
    
    print(f"S1 with 0 matches (singletons): {(match_counts == 0).sum()} ({(match_counts == 0).mean()*100:.2f}%)")
    print(f"S1 with 1 match: {(match_counts == 1).sum()}")
    print(f"S1 with >1 match: {(match_counts > 1).sum()}")
    
    all_matched_ids = [m for sublist in gt_df['match_list'] for m in sublist]
    s2_matches = sum(1 for m in all_matched_ids if 's2' in m.lower() or m.startswith('S2') or 'source2' in m.lower())
    s3_matches = sum(1 for m in all_matched_ids if 's3' in m.lower() or m.startswith('S3') or 'source3' in m.lower())
    print(f"Total matched IDs: {len(all_matched_ids)}")
    print(f"Sample matched IDs: {all_matched_ids[:10]}")
