import pandas as pd
import numpy as np
import os
import gc
import sys

DATA_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\dataset"
OUTPUT_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\output"
SUBMISSIONS_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\submissions\day1"
EXP_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\experiments"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

print("Starting Fast Optimized Day 1 Submission 1 Pipeline...", flush=True)

def quick_norm(s):
    return s.fillna('').astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

def build_and_match(s1_path, s2_path, s3_path):
    print(f"Reading S1: {os.path.basename(s1_path)}...", flush=True)
    df_s1 = pd.read_csv(s1_path, sep='\t', quoting=3, dtype=str)
    
    print(f"Reading S2: {os.path.basename(s2_path)}...", flush=True)
    df_s2 = pd.read_csv(s2_path, sep='\t', quoting=3, dtype=str)
    
    print(f"Reading S3: {os.path.basename(s3_path)}...", flush=True)
    df_s3 = pd.read_csv(s3_path, sep='\t', quoting=3, dtype=str)
    
    # Create exact key (alphanumeric string concatenation)
    df_s1['key_nc'] = quick_norm(df_s1['business_name']) + "||" + df_s1['country'].fillna('').astype(str).str.upper().str.strip()
    df_s1['key_ac'] = quick_norm(df_s1['business_address']) + "||" + df_s1['country'].fillna('').astype(str).str.upper().str.strip()
    
    df_s2['key_nc'] = quick_norm(df_s2['business_name']) + "||" + df_s2['country'].fillna('').astype(str).str.upper().str.strip()
    df_s2['key_ac'] = quick_norm(df_s2['business_address']) + "||" + df_s2['country'].fillna('').astype(str).str.upper().str.strip()
    
    df_s3['key_nc'] = quick_norm(df_s3['business_name']) + "||" + df_s3['country'].fillna('').astype(str).str.upper().str.strip()
    df_s3['key_ac'] = quick_norm(df_s3['business_address']) + "||" + df_s3['country'].fillna('').astype(str).str.upper().str.strip()

    df_s23 = pd.concat([
        df_s2[['entity_id', 'key_nc', 'key_ac']],
        df_s3[['entity_id', 'key_nc', 'key_ac']]
    ], ignore_index=True)
    
    del df_s2, df_s3
    gc.collect()

    print("Matching Rule 1: Name + Country...", flush=True)
    m1 = pd.merge(
        df_s1[['entity_id', 'key_nc']].rename(columns={'entity_id': 's1_id'}),
        df_s23[df_s23['key_nc'].str.len() > 3].rename(columns={'entity_id': 'cand_id'}),
        on='key_nc',
        how='inner'
    )[['s1_id', 'cand_id']]

    print("Matching Rule 2: Address + Country...", flush=True)
    m2 = pd.merge(
        df_s1[['entity_id', 'key_ac']].rename(columns={'entity_id': 's1_id'}),
        df_s23[df_s23['key_ac'].str.len() > 3].rename(columns={'entity_id': 'cand_id'}),
        on='key_ac',
        how='inner'
    )[['s1_id', 'cand_id']]

    del df_s23
    gc.collect()

    print("Combining match pairs...", flush=True)
    all_m = pd.concat([m1, m2], ignore_index=True).drop_duplicates()
    
    print("Aggregating matched entity IDs...", flush=True)
    grouped = all_m.groupby('s1_id')['cand_id'].apply(lambda x: ','.join(sorted(x))).reset_index()
    grouped.columns = ['source1_entity_id', 'matched_entity_ids']

    res = pd.merge(df_s1[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'}), grouped, on='source1_entity_id', how='left')
    res['matched_entity_ids'] = res['matched_entity_ids'].fillna('')
    return res

# 1. Validation Split (Sample 200,000 S1)
print("\n=== STEP 1: VALIDATION EVALUATION ===", flush=True)
tr_s1_path = os.path.join(DATA_DIR, 'train', 'train_source1.tsv')
tr_s2_path = os.path.join(DATA_DIR, 'train', 'train_source2.tsv')
tr_s3_path = os.path.join(DATA_DIR, 'train', 'train_source3.tsv')
gt_path = os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv')

val_predictions = build_and_match(tr_s1_path, tr_s2_path, tr_s3_path)

print("Evaluating validation performance...", flush=True)
gt_df = pd.read_csv(gt_path, sep='\t', quoting=3, dtype=str)
eval_df = pd.merge(gt_df, val_predictions, on='source1_entity_id', suffixes=('_true', '_pred'))

def eval_row(row):
    t_str = str(row['matched_entity_ids_true']) if pd.notna(row['matched_entity_ids_true']) else ""
    p_str = str(row['matched_entity_ids_pred']) if pd.notna(row['matched_entity_ids_pred']) else ""
    
    true_set = set([x.strip() for x in t_str.split(',') if x.strip()])
    pred_set = set([x.strip() for x in p_str.split(',') if x.strip()])
    
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0, 1.0, 1.0, 0, 0
    
    tp = len(true_set.intersection(pred_set))
    prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    rec = tp / len(true_set) if len(true_set) > 0 else 0.0
    
    if prec + rec == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
        
    return prec, rec, f05, len(true_set), len(pred_set)

eval_results = [eval_row(row) for _, row in eval_df.iterrows()]
res_df = pd.DataFrame(eval_results, columns=['prec', 'rec', 'f05', 'true_cnt', 'pred_cnt'])

macro_prec = res_df['prec'].mean()
macro_rec = res_df['rec'].mean()
macro_f05 = res_df['f05'].mean()

singletons_df = res_df[res_df['true_cnt'] == 0]
total_singletons = len(singletons_df)
correct_singletons = (singletons_df['pred_cnt'] == 0).sum()
incorrect_singletons = (singletons_df['pred_cnt'] > 0).sum()

print(f"\n--- VALIDATION RESULTS ---", flush=True)
print(f"Macro Precision: {macro_prec:.4f}", flush=True)
print(f"Macro Recall:    {macro_rec:.4f}", flush=True)
print(f"Macro F0.5:      {macro_f05:.4f}", flush=True)
print(f"Singletons Total: {total_singletons}", flush=True)
print(f"Singletons Correct: {correct_singletons}", flush=True)
print(f"Singletons Incorrect: {incorrect_singletons}", flush=True)

del val_predictions, gt_df, eval_df, res_df
gc.collect()

# 2. Test Inference
print("\n=== STEP 2: TEST INFERENCE ===", flush=True)
te_s1_path = os.path.join(DATA_DIR, 'test', 'test_source1.tsv')
te_s2_path = os.path.join(DATA_DIR, 'test', 'test_source2.tsv')
te_s3_path = os.path.join(DATA_DIR, 'test', 'test_source3.tsv')

test_predictions = build_and_match(te_s1_path, te_s2_path, te_s3_path)

candidate_pairs = test_predictions.copy()
candidate_pairs.columns = ['source1_entity_id', 'candidate_entity_ids']

matching_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
candidate_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
sub_path = os.path.join(SUBMISSIONS_DIR, 'submission_01.tsv')

print(f"Saving matching_results.tsv to {matching_path}...", flush=True)
test_predictions.to_csv(matching_path, sep='\t', index=False)

print(f"Saving candidate_pairs.tsv to {candidate_path}...", flush=True)
candidate_pairs.to_csv(candidate_path, sep='\t', index=False)

print(f"Saving submission copy to {sub_path}...", flush=True)
test_predictions.to_csv(sub_path, sep='\t', index=False)

test_s1_count = len(test_predictions)
test_matched_count = (test_predictions['matched_entity_ids'].fillna('').str.len() > 0).sum()
test_sing_count = (test_predictions['matched_entity_ids'].fillna('').str.len() == 0).sum()

print(f"\n--- TEST PREDICTIONS SUMMARY ---", flush=True)
print(f"Test S1 Rows:            {test_s1_count}", flush=True)
print(f"Test S1 with Matches:    {test_matched_count}", flush=True)
print(f"Test Singleton S1:       {test_sing_count}", flush=True)

# 3. Create Experiment Log
log_content = f"""# Experiment Log: DAY1_SUBMISSION1

## Method
Fast Alphanumeric Match Engine (Name+Country, Addr+Country)

## Normalization
- Strip non-alphanumeric characters
- Lowercase
- Append uppercase country code

## Validation Results
- Macro Precision: {macro_prec:.4f}
- Macro Recall:    {macro_rec:.4f}
- Macro F0.5:      {macro_f05:.4f}

## Singletons Summary
- Total Singletons: {total_singletons}
- Correctly Predicted Singletons: {correct_singletons}
- Incorrectly Matched Singletons: {incorrect_singletons}

## Test Outputs
- Test S1 Total Rows: {test_s1_count}
- Predicted Match S1 Count: {test_matched_count}
- Predicted Singleton S1 Count: {test_sing_count}
"""

with open(os.path.join(EXP_DIR, 'experiment_log.md'), 'w') as f:
    f.write(log_content)

print("\nPipeline execution complete!", flush=True)
