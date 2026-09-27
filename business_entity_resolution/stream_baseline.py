import os
import sys
import time
import collections
import re
import pandas as pd

DATA_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\dataset"
OUTPUT_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\output"
SUBMISSIONS_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\submissions\day1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# Precompiled regexes
RE_NON_ALPHA = re.compile(r'[^a-z0-9\s]')
RE_SUFFIXES = re.compile(r'\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|plc|sarl|sa|gmbh)\b')
RE_SPACE = re.compile(r'\s+')

ADDR_ABBR = [
    (re.compile(r'\bst\b'), 'street'),
    (re.compile(r'\brd\b'), 'road'),
    (re.compile(r'\bave\b'), 'avenue'),
    (re.compile(r'\bblvd\b'), 'boulevard'),
    (re.compile(r'\bdr\b'), 'drive'),
    (re.compile(r'\bln\b'), 'lane'),
    (re.compile(r'\bste\b'), 'suite'),
    (re.compile(r'\bapt\b'), 'apartment'),
    (re.compile(r'\bfl\b'), 'floor'),
    (re.compile(r'\bpo box\b'), 'pobox')
]

def fast_norm_name(text):
    if not text or pd.isna(text):
        return ""
    s = str(text).lower()
    s = RE_NON_ALPHA.sub(' ', s)
    s = RE_SUFFIXES.sub(' ', s)
    s = RE_SPACE.sub(' ', s).strip()
    return s

def fast_norm_addr(text):
    if not text or pd.isna(text):
        return ""
    s = str(text).lower()
    s = RE_NON_ALPHA.sub(' ', s)
    for pat, repl in ADDR_ABBR:
        s = pat.sub(repl, s)
    s = RE_SPACE.sub(' ', s).strip()
    return s

def process_file(file_path, is_s1=False):
    t0 = time.time()
    print(f"Reading {os.path.basename(file_path)}...", flush=True)
    df = pd.read_csv(file_path, sep='\t')
    
    names = df['business_name'].values
    addrs = df['business_address'].values
    ctrys = df['country'].fillna('').astype(str).str.lower().str.strip().values
    eids = df['entity_id'].values
    
    print(f"Normalizing {len(df)} rows...", flush=True)
    norm_names = [fast_norm_name(x) for x in names]
    norm_addrs = [fast_norm_addr(x) for x in addrs]
    
    print(f"Processed {os.path.basename(file_path)} in {time.time()-t0:.2f}s", flush=True)
    return eids, norm_names, norm_addrs, ctrys

def run_pipeline(split='train'):
    print(f"\n==========================================", flush=True)
    print(f"   STARTING PIPELINE FOR {split.upper()} SPLIT", flush=True)
    print(f"==========================================", flush=True)
    
    t_start = time.time()
    
    s1_path = os.path.join(DATA_DIR, split, f"{split}_source1.tsv")
    s2_path = os.path.join(DATA_DIR, split, f"{split}_source2.tsv")
    s3_path = os.path.join(DATA_DIR, split, f"{split}_source3.tsv")
    
    s1_eids, s1_names, s1_addrs, s1_ctrys = process_file(s1_path, is_s1=True)
    s2_eids, s2_names, s2_addrs, s2_ctrys = process_file(s2_path)
    s3_eids, s3_names, s3_addrs, s3_ctrys = process_file(s3_path)
    
    print("Building indexing maps for S2...", flush=True)
    t0 = time.time()
    s2_name_map = collections.defaultdict(list)
    s2_addr_map = collections.defaultdict(list)
    for eid, name, addr, ctry in zip(s2_eids, s2_names, s2_addrs, s2_ctrys):
        if len(name) >= 3 and ctry:
            s2_name_map[(name, ctry)].append(eid)
        if len(addr) >= 5 and ctry:
            s2_addr_map[(addr, ctry)].append(eid)
    print(f"S2 maps built in {time.time()-t0:.2f}s (Name keys: {len(s2_name_map)}, Addr keys: {len(s2_addr_map)})", flush=True)
    
    print("Building indexing maps for S3...", flush=True)
    t0 = time.time()
    s3_name_map = collections.defaultdict(list)
    s3_addr_map = collections.defaultdict(list)
    for eid, name, addr, ctry in zip(s3_eids, s3_names, s3_addrs, s3_ctrys):
        if len(name) >= 3 and ctry:
            s3_name_map[(name, ctry)].append(eid)
        if len(addr) >= 5 and ctry:
            s3_addr_map[(addr, ctry)].append(eid)
    print(f"S3 maps built in {time.time()-t0:.2f}s (Name keys: {len(s3_name_map)}, Addr keys: {len(s3_addr_map)})", flush=True)
    
    print("Performing baseline exact matching for S1 entities...", flush=True)
    t0 = time.time()
    predictions = {}
    total_candidate_pairs = 0
    
    for eid, name, addr, ctry in zip(s1_eids, s1_names, s1_addrs, s1_ctrys):
        mset = set()
        key_name = (name, ctry)
        key_addr = (addr, ctry)
        
        if len(name) >= 3 and ctry:
            if key_name in s2_name_map:
                mset.update(s2_name_map[key_name])
            if key_name in s3_name_map:
                mset.update(s3_name_map[key_name])
                
        if len(addr) >= 5 and ctry:
            if key_addr in s2_addr_map:
                mset.update(s2_addr_map[key_addr])
            if key_addr in s3_addr_map:
                mset.update(s3_addr_map[key_addr])
                
        predictions[eid] = mset
        total_candidate_pairs += len(mset)
        
    print(f"Matched {len(predictions)} S1 entities in {time.time()-t0:.2f}s. Total matched pairs: {total_candidate_pairs}", flush=True)
    
    metrics = None
    if split == 'train':
        print("Evaluating predictions against Ground Truth...", flush=True)
        t0 = time.time()
        gt_path = os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv')
        gt = pd.read_csv(gt_path, sep='\t')
        
        gt_s1 = gt['source1_entity_id'].values
        gt_matched = gt['matched_entity_ids'].fillna('').values
        
        total_prec = 0.0
        total_rec = 0.0
        total_f05 = 0.0
        singleton_correct = 0
        singleton_total = 0
        n_entities = len(gt)
        
        for s1_id, m_str in zip(gt_s1, gt_matched):
            true_set = set(m_str.split(',')) if m_str else set()
            pred_set = predictions.get(s1_id, set())
            
            p_len = len(pred_set)
            t_len = len(true_set)
            
            inter = len(pred_set.intersection(true_set)) if p_len > 0 and t_len > 0 else 0
            
            prec = 1.0 if p_len == 0 else inter / p_len
            rec = (1.0 if p_len == 0 else 0.0) if t_len == 0 else inter / t_len
            
            f05 = 0.0 if prec + rec == 0 else (1.25 * prec * rec) / (0.25 * prec + rec)
            
            total_prec += prec
            total_rec += rec
            total_f05 += f05
            
            if t_len == 0:
                singleton_total += 1
                if p_len == 0:
                    singleton_correct += 1
                    
        metrics = {
            'macro_precision': total_prec / n_entities,
            'macro_recall': total_rec / n_entities,
            'macro_f05': total_f05 / n_entities,
            'singleton_acc': singleton_correct / singleton_total,
            'singleton_correct': singleton_correct,
            'singleton_total': singleton_total,
            'predicted_matches_count': total_candidate_pairs
        }
        print(f"Evaluated in {time.time()-t0:.2f}s", flush=True)
        
    print(f"Split {split} completed in {time.time()-t_start:.2f}s", flush=True)
    return s1_eids, predictions, total_candidate_pairs, metrics

def main():
    start_total_time = time.time()
    
    # 1. Train Split Validation
    train_eids, train_preds, train_pairs, val_metrics = run_pipeline(split='train')
    
    print("\n==========================================", flush=True)
    print("      REAL TRAIN VALIDATION RESULTS", flush=True)
    print("==========================================", flush=True)
    print(f"Validation Macro Precision: {val_metrics['macro_precision']:.6f}", flush=True)
    print(f"Validation Macro Recall:    {val_metrics['macro_recall']:.6f}", flush=True)
    print(f"Validation Macro F0.5:      {val_metrics['macro_f05']:.6f}", flush=True)
    print(f"Singleton Performance:      {val_metrics['singleton_acc']:.6f} ({val_metrics['singleton_correct']}/{val_metrics['singleton_total']})", flush=True)
    print(f"Number of Predictions:      {val_metrics['predicted_matches_count']}", flush=True)
    
    # 2. Test Split Prediction
    test_eids, test_preds, test_pairs, _ = run_pipeline(split='test')
    
    # Write output files
    print("\nGenerating output/matching_results.tsv and submission_01.tsv...", flush=True)
    sub_rows = []
    for eid in test_eids:
        m_list = sorted(list(test_preds.get(eid, set())))
        sub_rows.append((eid, ",".join(m_list)))
        
    sub_df = pd.DataFrame(sub_rows, columns=['source1_entity_id', 'matched_entity_ids'])
    
    matching_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    submission_path = os.path.join(SUBMISSIONS_DIR, "submission_01.tsv")
    
    sub_df.to_csv(matching_path, sep='\t', index=False)
    sub_df.to_csv(submission_path, sep='\t', index=False)
    
    # Candidate pairs summary file
    candidate_summary_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    with open(candidate_summary_path, "w", encoding="utf-8") as f:
        f.write("split\ttotal_s1_entities\ttotal_candidate_pairs\n")
        f.write(f"train\t{len(train_eids)}\t{train_pairs}\n")
        f.write(f"test\t{len(test_eids)}\t{test_pairs}\n")
        
    # Validation checks
    print("\nValidating submission format...", flush=True)
    assert len(sub_df) == len(test_eids), f"Row count mismatch! Expected {len(test_eids)}, got {len(sub_df)}"
    assert sub_df['source1_entity_id'].nunique() == len(test_eids), "Duplicate S1 IDs found!"
    assert list(sub_df.columns) == ['source1_entity_id', 'matched_entity_ids'], "Column header mismatch!"
    print("Submission validation PASSED perfectly!", flush=True)
    
    # Experiment Log
    exp_log_path = os.path.join(SUBMISSIONS_DIR, "experiment_log.tsv")
    log_exists = os.path.exists(exp_log_path)
    with open(exp_log_path, "a", encoding="utf-8") as f:
        if not log_exists:
            f.write("experiment_id\tnormalization_rules\tmatching_rules\tval_precision\tval_recall\tval_f05\tpredicted_matches\tsingleton_acc\n")
        f.write(f"submission_01\tconservative_lowercase_unicode_suffixes_address\texact_name_ctry_and_addr_ctry\t{val_metrics['macro_precision']:.6f}\t{val_metrics['macro_recall']:.6f}\t{val_metrics['macro_f05']:.6f}\t{val_metrics['predicted_matches_count']}\t{val_metrics['singleton_acc']:.6f}\n")
        
    print(f"\n==========================================", flush=True)
    print(f"  TOTAL EXECUTION TIME: {time.time() - start_total_time:.2f}s", flush=True)
    print(f"==========================================", flush=True)

if __name__ == "__main__":
    main()
