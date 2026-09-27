import os
import pandas as pd
import numpy as np

def validate_submission(matching_file, candidate_file, test_s1_file, test_s2_file, test_s3_file):
    print("\n=== RUNNING SUBMISSION VALIDATOR ===")
    errors = []
    warnings = []
    
    # 1. Check file existence
    if not os.path.exists(matching_file):
        errors.append(f"Matching file missing: {matching_file}")
        return False, errors, warnings
    if not os.path.exists(candidate_file):
        errors.append(f"Candidate file missing: {candidate_file}")
        return False, errors, warnings

    # Read files
    df_match = pd.read_csv(matching_file, sep='\t', quoting=3, dtype=str)
    df_cand = pd.read_csv(candidate_file, sep='\t', quoting=3, dtype=str)
    df_s1 = pd.read_csv(test_s1_file, sep='\t', quoting=3, dtype=str)
    df_s2 = pd.read_csv(test_s2_file, sep='\t', quoting=3, dtype=str)
    df_s3 = pd.read_csv(test_s3_file, sep='\t', quoting=3, dtype=str)

    s1_ids = set(df_s1['entity_id'].unique())
    s23_ids = set(df_s2['entity_id'].unique()).union(set(df_s3['entity_id'].unique()))

    # Check Columns
    if list(df_match.columns) != ['source1_entity_id', 'matched_entity_ids']:
        errors.append(f"Incorrect columns in matching_results.tsv: {list(df_match.columns)}")
    if list(df_cand.columns) != ['source1_entity_id', 'candidate_entity_ids']:
        errors.append(f"Incorrect columns in candidate_pairs.tsv: {list(df_cand.columns)}")

    # Check S1 Alignment
    match_s1_ids = df_match['source1_entity_id'].tolist()
    if len(match_s1_ids) != len(s1_ids):
        errors.append(f"matching_results row count ({len(match_s1_ids)}) does not match test S1 count ({len(s1_ids)})")
    if set(match_s1_ids) != s1_ids:
        errors.append("matching_results source1_entity_id set does not match test S1 set")
    if len(match_s1_ids) != len(set(match_s1_ids)):
        errors.append("Duplicate source1_entity_id rows found in matching_results")

    cand_s1_ids = df_cand['source1_entity_id'].tolist()
    if set(cand_s1_ids) != s1_ids:
        errors.append("candidate_pairs source1_entity_id set does not match test S1 set")
    if len(cand_s1_ids) != len(set(cand_s1_ids)):
        errors.append("Duplicate source1_entity_id rows found in candidate_pairs")

    # Match ID validations
    cand_dict = {}
    for _, row in df_cand.iterrows():
        s1 = row['source1_entity_id']
        c_str = str(row['candidate_entity_ids']) if pd.notna(row['candidate_entity_ids']) else ""
        c_list = [c.strip() for c in c_str.split(',') if c.strip()]
        if len(c_list) != len(set(c_list)):
            errors.append(f"Duplicate candidate IDs in row S1={s1}")
        cand_dict[s1] = set(c_list)

    for _, row in df_match.iterrows():
        s1 = row['source1_entity_id']
        m_str = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) else ""
        m_list = [m.strip() for m in m_str.split(',') if m.strip()]
        
        # Check duplicate matches
        if len(m_list) != len(set(m_list)):
            errors.append(f"Duplicate matched IDs in row S1={s1}")
            
        # Check invalid IDs & S1 in matches
        for m in m_list:
            if m in s1_ids:
                errors.append(f"S1 ID found in matched predictions for S1={s1}: {m}")
            if m not in s23_ids:
                errors.append(f"Invalid entity ID predicted for S1={s1}: {m}")
            # Check match present in candidates
            if s1 in cand_dict and m not in cand_dict[s1]:
                errors.append(f"Matched ID {m} for S1={s1} is NOT present in candidate_pairs")

    is_valid = len(errors) == 0
    if is_valid:
        print("VALIDATION SUCCESS: All submission requirements and checks passed!")
    else:
        print(f"VALIDATION FAILED with {len(errors)} error(s):")
        for err in errors[:10]:
            print(f" - {err}")
        if len(errors) > 10:
            print(f" ... and {len(errors)-10} more errors.")
            
    return is_valid, errors, warnings

if __name__ == "__main__":
    base_dir = r"d:\Working_project\ML Challenge\business_entity_resolution"
    validate_submission(
        os.path.join(base_dir, 'output', 'matching_results.tsv'),
        os.path.join(base_dir, 'output', 'candidate_pairs.tsv'),
        os.path.join(base_dir, 'dataset', 'test', 'test_source1.tsv'),
        os.path.join(base_dir, 'dataset', 'test', 'test_source2.tsv'),
        os.path.join(base_dir, 'dataset', 'test', 'test_source3.tsv')
    )
