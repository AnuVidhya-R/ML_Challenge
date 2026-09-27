import os
import sys
import time
import unicodedata
import re
import pandas as pd
import numpy as np

DATA_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\dataset"
OUTPUT_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\output"
SUBMISSIONS_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\submissions\day1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

def normalize_series(s):
    # Vectorized string cleaning
    s = s.fillna('').astype(str).str.lower()
    # Replace non-alphanumeric characters with space
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    # Strip business suffixes
    suffixes = r'\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|plc|sarl|sa|gmbh)\b'
    s = s.str.replace(suffixes, ' ', regex=True)
    # Strip address abbreviations
    abbrs = [
        (r'\bst\b', 'street'), (r'\brd\b', 'road'), (r'\bave\b', 'avenue'),
        (r'\bblvd\b', 'boulevard'), (r'\bdr\b', 'drive'), (r'\bln\b', 'lane'),
        (r'\bste\b', 'suite'), (r'\bapt\b', 'apartment'), (r'\bfl\b', 'floor'),
        (r'\bpo box\b', 'pobox')
    ]
    for pat, repl in abbrs:
        s = s.str.replace(pat, repl, regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def load_and_normalize(split='train'):
    print(f"Loading and normalizing {split} datasets...", flush=True)
    t0 = time.time()
    s1_path = os.path.join(DATA_DIR, split, f"{split}_source1.tsv")
    s2_path = os.path.join(DATA_DIR, split, f"{split}_source2.tsv")
    s3_path = os.path.join(DATA_DIR, split, f"{split}_source3.tsv")
    
    s1 = pd.read_csv(s1_path, sep='\t')
    s2 = pd.read_csv(s2_path, sep='\t')
    s3 = pd.read_csv(s3_path, sep='\t')
    
    print(f"Normalizing S1 (rows={len(s1)})...", flush=True)
    s1['norm_name'] = normalize_series(s1['business_name'])
    s1['norm_address'] = normalize_series(s1['business_address'])
    s1['norm_country'] = s1['country'].fillna('').astype(str).str.lower().str.strip()
    
    print(f"Normalizing S2 (rows={len(s2)})...", flush=True)
    s2['norm_name'] = normalize_series(s2['business_name'])
    s2['norm_address'] = normalize_series(s2['business_address'])
    s2['norm_country'] = s2['country'].fillna('').astype(str).str.lower().str.strip()
    
    print(f"Normalizing S3 (rows={len(s3)})...", flush=True)
    s3['norm_name'] = normalize_series(s3['business_name'])
    s3['norm_address'] = normalize_series(s3['business_address'])
    s3['norm_country'] = s3['country'].fillna('').astype(str).str.lower().str.strip()
    
    gt = None
    if split == 'train':
        gt_path = os.path.join(DATA_DIR, split, f"{split}_ground_truth.tsv")
        gt = pd.read_csv(gt_path, sep='\t')
        
    print(f"Loaded and normalized in {time.time()-t0:.2f}s", flush=True)
    return s1, s2, s3, gt

def build_index_and_match(s1, s2, s3):
    print("Building lookup indices for S2 and S3...", flush=True)
    t0 = time.time()
    
    # Pre-filter non-empty keys
    s2_valid_name = s2[s2['norm_name'].str.len() >= 3]
    s2_valid_addr = s2[s2['norm_address'].str.len() >= 5]
    
    s3_valid_name = s3[s3['norm_name'].str.len() >= 3]
    s3_valid_addr = s3[s3['norm_address'].str.len() >= 5]
    
    # Dict indexing for high performance
    s2_name_map = s2_valid_name.groupby(['norm_name', 'norm_country'])['entity_id'].apply(list).to_dict()
    s2_addr_map = s2_valid_addr.groupby(['norm_address', 'norm_country'])['entity_id'].apply(list).to_dict()
    
    s3_name_map = s3_valid_name.groupby(['norm_name', 'norm_country'])['entity_id'].apply(list).to_dict()
    s3_addr_map = s3_valid_addr.groupby(['norm_address', 'norm_country'])['entity_id'].apply(list).to_dict()
    
    print(f"Indices built in {time.time()-t0:.2f}s. Matching S1 records...", flush=True)
    t1 = time.time()
    
    predictions = {}
    candidate_pair_count = 0
    
    s1_names = s1['norm_name'].values
    s1_addrs = s1['norm_address'].values
    s1_ctrys = s1['norm_country'].values
    s1_ids = s1['entity_id'].values
    
    for i in range(len(s1)):
        eid = s1_ids[i]
        n_name = s1_names[i]
        n_addr = s1_addrs[i]
        n_ctry = s1_ctrys[i]
        
        mset = set()
        key_name = (n_name, n_ctry)
        key_addr = (n_addr, n_ctry)
        
        if len(n_name) >= 3:
            if key_name in s2_name_map:
                mset.update(s2_name_map[key_name])
            if key_name in s3_name_map:
                mset.update(s3_name_map[key_name])
                
        if len(n_addr) >= 5:
            if key_addr in s2_addr_map:
                mset.update(s2_addr_map[key_addr])
            if key_addr in s3_addr_map:
                mset.update(s3_addr_map[key_addr])
                
        predictions[eid] = mset
        candidate_pair_count += len(mset)
        
    print(f"Matching finished in {time.time()-t1:.2f}s", flush=True)
    return predictions, candidate_pair_count

def evaluate_predictions(predictions, gt):
    print("Evaluating predictions...", flush=True)
    gt_dict = {}
    gt_s1 = gt['source1_entity_id'].values
    gt_matched = gt['matched_entity_ids'].fillna('').values
    
    for i in range(len(gt)):
        s1_id = gt_s1[i]
        m_str = gt_matched[i]
        if not m_str:
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(m_str.split(','))
            
    total_prec = 0.0
    total_rec = 0.0
    total_f05 = 0.0
    
    singleton_correct = 0
    singleton_total = 0
    
    n_entities = len(gt_dict)
    
    for s1_id, true_set in gt_dict.items():
        pred_set = predictions.get(s1_id, set())
        
        p_len = len(pred_set)
        t_len = len(true_set)
        inter = len(pred_set.intersection(true_set)) if p_len > 0 and t_len > 0 else 0
        
        # Precision
        prec = 1.0 if p_len == 0 else inter / p_len
        # Recall
        rec = (1.0 if p_len == 0 else 0.0) if t_len == 0 else inter / t_len
            
        # F0.5
        if prec + rec == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
            
        total_prec += prec
        total_rec += rec
        total_f05 += f05
        
        if t_len == 0:
            singleton_total += 1
            if p_len == 0:
                singleton_correct += 1
                
    macro_prec = total_prec / n_entities
    macro_rec = total_rec / n_entities
    macro_f05 = total_f05 / n_entities
    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 0.0
    
    return {
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'macro_f05': macro_f05,
        'singleton_acc': singleton_acc,
        'singleton_correct': singleton_correct,
        'singleton_total': singleton_total,
        'predicted_matches_count': sum(len(v) for v in predictions.values())
    }

def main():
    start_time = time.time()
    
    print("=== STEP 1: VALIDATION ON TRAIN SET ===", flush=True)
    tr_s1, tr_s2, tr_s3, tr_gt = load_and_normalize(split='train')
    val_preds, val_candidate_pairs = build_index_and_match(tr_s1, tr_s2, tr_s3)
    val_metrics = evaluate_predictions(val_preds, tr_gt)
    
    print("\n--- TRAIN VALIDATION RESULTS ---", flush=True)
    print(f"Validation Macro Precision: {val_metrics['macro_precision']:.6f}")
    print(f"Validation Macro Recall:    {val_metrics['macro_recall']:.6f}")
    print(f"Validation Macro F0.5:      {val_metrics['macro_f05']:.6f}")
    print(f"Singleton Performance:      {val_metrics['singleton_acc']:.6f} ({val_metrics['singleton_correct']}/{val_metrics['singleton_total']})")
    print(f"Total Predicted Match Pairs: {val_metrics['predicted_matches_count']}")
    
    print("\n=== STEP 2: GENERATING TEST SUBMISSION ===", flush=True)
    te_s1, te_s2, te_s3, _ = load_and_normalize(split='test')
    test_preds, test_candidate_pairs = build_index_and_match(te_s1, te_s2, te_s3)
    
    print("Formatting submission TSV...", flush=True)
    te_s1_ids = te_s1['entity_id'].values
    sub_rows = []
    for eid in te_s1_ids:
        matched = sorted(list(test_preds.get(eid, set())))
        matched_str = ",".join(matched)
        sub_rows.append((eid, matched_str))
        
    sub_df = pd.DataFrame(sub_rows, columns=['source1_entity_id', 'matched_entity_ids'])
    
    matching_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    submission_path = os.path.join(SUBMISSIONS_DIR, "submission_01.tsv")
    
    sub_df.to_csv(matching_path, sep='\t', index=False)
    sub_df.to_csv(submission_path, sep='\t', index=False)
    
    candidate_summary_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    with open(candidate_summary_path, "w", encoding="utf-8") as f:
        f.write("split\ttotal_s1_entities\ttotal_candidate_pairs\n")
        f.write(f"train\t{len(tr_s1)}\t{val_candidate_pairs}\n")
        f.write(f"test\t{len(te_s1)}\t{test_candidate_pairs}\n")
        
    print("\n=== STEP 3: SUBMISSION VALIDATION ===", flush=True)
    print(f"Submission row count: {len(sub_df)}")
    print(f"Expected test S1 count: {len(te_s1)}")
    assert len(sub_df) == len(te_s1), f"Row count mismatch! Expected {len(te_s1)}, got {len(sub_df)}"
    assert sub_df['source1_entity_id'].nunique() == len(te_s1), "Duplicate S1 IDs in submission!"
    assert list(sub_df.columns) == ['source1_entity_id', 'matched_entity_ids'], "Column header mismatch!"
    print("Submission validation PASSED successfully!")
    
    exp_log_path = os.path.join(SUBMISSIONS_DIR, "experiment_log.tsv")
    log_exists = os.path.exists(exp_log_path)
    with open(exp_log_path, "a", encoding="utf-8") as f:
        if not log_exists:
            f.write("experiment_id\tnormalization_rules\tmatching_rules\tval_precision\tval_recall\tval_f05\tpredicted_matches\tsingleton_acc\n")
        f.write(f"submission_01\tconservative_lowercase_unicode_suffixes_address\texact_name_ctry_and_addr_ctry\t{val_metrics['macro_precision']:.6f}\t{val_metrics['macro_recall']:.6f}\t{val_metrics['macro_f05']:.6f}\t{val_metrics['predicted_matches_count']}\t{val_metrics['singleton_acc']:.6f}\n")
        
    print(f"\nTotal Elapsed time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

