import os
import sys
import pandas as pd

BASE_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution"
TEST_S1_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source1.tsv")
TEST_S2_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source2.tsv")
TEST_S3_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source3.tsv")

def validate_submission_fast(sub_file, is_candidate=False):
    print(f"\n==================================================")
    print(f"FAST VALIDATION: {os.path.basename(sub_file)}")
    print(f"==================================================")
    
    if not os.path.exists(sub_file):
        print(f"FAIL: File does not exist: {sub_file}")
        return False

    # 1. Load valid Source 1 IDs & Source 2/3 IDs set efficiently
    print("Loading test entity ID sets...")
    df_s1 = pd.read_csv(TEST_S1_PATH, sep='\t', quoting=3, usecols=['entity_id'], dtype=str)
    expected_s1_ids = set(df_s1['entity_id'].dropna().str.strip())
    expected_count = len(expected_s1_ids)
    print(f"Expected test Source1 entity count: {expected_count:,}")

    df_s2 = pd.read_csv(TEST_S2_PATH, sep='\t', quoting=3, usecols=['entity_id'], dtype=str)
    df_s3 = pd.read_csv(TEST_S3_PATH, sep='\t', quoting=3, usecols=['entity_id'], dtype=str)
    valid_s23_ids = set(df_s2['entity_id'].dropna().str.strip()).union(set(df_s3['entity_id'].dropna().str.strip()))
    print(f"Valid Target Source 2 & 3 entity count: {len(valid_s23_ids):,}")

    # 2. Check Header
    df_head = pd.read_csv(sub_file, sep='\t', quoting=3, nrows=2, dtype=str)
    cols = list(df_head.columns)
    if 'candidate_entity_ids' in cols:
        target_col = 'candidate_entity_ids'
    else:
        target_col = 'matched_entity_ids'
    expected_cols = ['source1_entity_id', target_col]
    
    errors = []
    if cols != expected_cols:
        errors.append(f"Header mismatch! Expected {expected_cols}, got {cols}")

    # 3. Process submission in chunks
    seen_s1_ids = set()
    total_rows = 0
    dup_s1_count = 0
    invalid_target_ids = 0
    self_match_count = 0
    non_empty_rows = 0
    empty_rows = 0

    target_col = expected_cols[1]

    for chunk in pd.read_csv(sub_file, sep='\t', quoting=3, chunksize=250000, dtype=str):
        if len(chunk.columns) != 2:
            errors.append(f"Row chunk has wrong column count: {len(chunk.columns)}")
            break
            
        total_rows += len(chunk)
        
        # S1 IDs
        chunk_s1 = chunk['source1_entity_id'].fillna('').str.strip()
        
        # Check unknown S1 IDs
        for s1_id in chunk_s1:
            if s1_id in seen_s1_ids:
                dup_s1_count += 1
            seen_s1_ids.add(s1_id)
            if s1_id not in expected_s1_ids:
                errors.append(f"Unknown source1_entity_id found: '{s1_id}'")
                if len(errors) > 20: break
        
        # Target predictions validation
        chunk_target = chunk[target_col].fillna('').str.strip()
        for s1_id, t_str in zip(chunk_s1, chunk_target):
            if not t_str:
                empty_rows += 1
                continue
            empty_rows if not t_str else None
            non_empty_rows += 1
            
            t_ids = [t.strip() for t in t_str.split(',') if t.strip()]
            if len(t_ids) != len(set(t_ids)):
                errors.append(f"Duplicate target IDs in row for S1={s1_id}")
                if len(errors) > 20: break
                
            for t_id in t_ids:
                if t_id == s1_id or t_id in expected_s1_ids:
                    self_match_count += 1
                    errors.append(f"S1 ID found in target predictions for S1={s1_id}: {t_id}")
                elif t_id not in valid_s23_ids:
                    invalid_target_ids += 1
                    errors.append(f"Invalid entity ID predicted for S1={s1_id}: {t_id}")
                if len(errors) > 20: break

    # Final checks
    if total_rows != expected_count:
        errors.append(f"Row count mismatch! Expected {expected_count:,}, got {total_rows:,}")
    if len(seen_s1_ids) != expected_count:
        errors.append(f"Unique S1 ID count mismatch! Expected {expected_count:,}, got {len(seen_s1_ids):,}")
    if dup_s1_count > 0:
        errors.append(f"Found {dup_s1_count} duplicate source1_entity_id entries!")

    # Summary Output
    print("\n--- VALIDATION SUMMARY ---")
    print(f"Total Rows Checked       : {total_rows:,}")
    print(f"Unique Source1 IDs Seen  : {len(seen_s1_ids):,}")
    print(f"Duplicate Source1 IDs    : {dup_s1_count}")
    print(f"Non-empty Predictions    : {non_empty_rows:,}")
    print(f"Empty Predictions        : {empty_rows:,}")
    print(f"Self / Invalid Targets   : {self_match_count + invalid_target_ids}")
    
    if errors:
        print(f"\nSTATUS: FAIL ({len(errors)} error(s) found)")
        for err in errors[:10]:
            print(f" - {err}")
        return False
    else:
        print("\nSTATUS: PASS (All validation checks successfully passed!)")
        return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        target_path = os.path.join(BASE_DIR, "output", "matching_results.tsv")
    validate_submission_fast(target_path)
