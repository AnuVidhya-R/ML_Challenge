import os
import sys
import time
import collections
import pandas as pd

DATA_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\dataset"
OUTPUT_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\output"
SUBMISSIONS_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution\submissions\day1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

def norm_series_name(s):
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s = s.str.replace(r'\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|plc|sarl|sa|gmbh)\b', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def norm_series_addr(s):
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
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

def process_tsv(fpath):
    t0 = time.time()
    fname = os.path.basename(fpath)
    print(f"Loading {fname}...", flush=True)
    df = pd.read_csv(fpath, sep='\t')
    
    print(f"Vectorized normalizing {fname} ({len(df)} rows)...", flush=True)
    n_name = norm_series_name(df['business_name']).values
    n_addr = norm_series_addr(df['business_address']).values
    ctry = df['country'].fillna('').astype(str).str.lower().str.strip().values
    eids = df['entity_id'].values
    
    print(f"Finished {fname} in {time.time()-t0:.2f}s", flush=True)
    return eids, n_name, n_addr, ctry

def run_split(split='train'):
    print(f"\n==========================================", flush=True)
    print(f"   RUNNING FAST PIPELINE FOR {split.upper()}", flush=True)
    print(f"==========================================", flush=True)
    t_start = time.time()
    
    s1_path = os.path.join(DATA_DIR, split, f"{split}_source1.tsv")
    s2_path = os.path.join(DATA_DIR, split, f"{split}_source2.tsv")
    s3_path = os.path.join(DATA_DIR, split, f"{split}_source3.tsv")
    
    s1_ids, s1_names, s1_addrs, s1_ctrys = process_tsv(s1_path)
    s2_ids, s2_names, s2_addrs, s2_ctrys = process_tsv(s2_path)
    s3_ids, s3_names, s3_addrs, s3_ctrys = process_tsv(s3_path)
    
    print("Building dictionary indices for S2...", flush=True)
    t0 = time.time()
    s2_name_map = collections.defaultdict(list)
    s2_addr_map = collections.defaultdict(list)
    for eid, name, addr, ctry in zip(s2_ids, s2_names, s2_addrs, s2_ctrys):
        if len(name) >= 3 and ctry:
            s2_name_map[(name, ctry)].append(eid)
        if len(addr) >= 5 and ctry:
            s2_addr_map[(addr, ctry)].append(eid)
    print(f"S2 index built in {time.time()-t0:.2f}s", flush=True)
    
    print("Building dictionary indices for S3...", flush=True)
    t0 = time.time()
    s3_name_map = collections.defaultdict(list)
    s3_addr_map = collections.defaultdict(list)
    for eid, name, addr, ctry in zip(s3_ids, s3_names, s3_addrs, s3_ctrys):
        if len(name) >= 3 and ctry:
            s3_name_map[(name, ctry)].append(eid)
        if len(addr) >= 5 and ctry:
            s3_addr_map[(addr, ctry)].append(eid)
    print(f"S3 index built in {time.time()-t0:.2f}s", flush=True)
    
    print("Performing baseline exact matching...", flush=True)
    t0 = time.time()
    predictions = {}
    total_pairs = 0
    for eid, name, addr, ctry in zip(s1_ids, s1_names, s1_addrs, s1_ctrys):
        mset = set()
        k_name = (name, ctry)
        k_addr = (addr, ctry)
        
        if len(name) >= 3 and ctry:
            if k_name in s2_name_map:
                mset.update(s2_name_map[k_name])
            if k_name in s3_name_map:
                mset.update(s3_name_map[k_name])
                
        if len(addr) >= 5 and ctry:
            if k_addr in s2_addr_map:
                mset.update(s2_addr_map[k_addr])
            if k_addr in s3_addr_map:
                mset.update(s3_addr_map[k_addr])
                
        predictions[eid] = mset
        total_pairs += len(mset)
        
    print(f"Matched {len(predictions)} entities in {time.time()-t0:.2f}s. Total candidate pairs: {total_pairs}", flush=True)
    
    metrics = None
    if split == 'train':
        print("Evaluating validation metrics against Ground Truth...", flush=True)
        t0 = time.time()
        gt = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv'), sep='\t')
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
            'predicted_matches_count': total_pairs
        }
        print(f"Evaluated in {time.time()-t0:.2f}s", flush=True)
        
    print(f"Total time for {split}: {time.time()-t_start:.2f}s", flush=True)
    return s1_ids, predictions, total_pairs, metrics

def main():
    t_main = time.time()
    
    # Train validation
    tr_ids, tr_preds, tr_pairs, val_metrics = run_split('train')
    
    print("\n==========================================", flush=True)
    print("      REAL TRAIN VALIDATION RESULTS", flush=True)
    print("==========================================", flush=True)
    print(f"Validation Macro Precision: {val_metrics['macro_precision']:.6f}", flush=True)
    print(f"Validation Macro Recall:    {val_metrics['macro_recall']:.6f}", flush=True)
    print(f"Validation Macro F0.5:      {val_metrics['macro_f05']:.6f}", flush=True)
    print(f"Singleton Performance:      {val_metrics['singleton_acc']:.6f} ({val_metrics['singleton_correct']}/{val_metrics['singleton_total']})", flush=True)
    print(f"Number of Predictions:      {val_metrics['predicted_matches_count']}", flush=True)
    
    # Test submission
    te_ids, te_preds, te_pairs, _ = run_split('test')
    
    print("\nGenerating submission files...", flush=True)
    sub_rows = []
    for eid in te_ids:
        m_list = sorted(list(te_preds.get(eid, set())))
        sub_rows.append((eid, ",".join(m_list)))
        
    sub_df = pd.DataFrame(sub_rows, columns=['source1_entity_id', 'matched_entity_ids'])
    
    matching_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    submission_path = os.path.join(SUBMISSIONS_DIR, "submission_01.tsv")
    
    sub_df.to_csv(matching_path, sep='\t', index=False)
    sub_df.to_csv(submission_path, sep='\t', index=False)
    
    candidate_summary_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    with open(candidate_summary_path, "w", encoding="utf-8") as f:
        f.write("split\ttotal_s1_entities\ttotal_candidate_pairs\n")
        f.write(f"train\t{len(tr_ids)}\t{tr_pairs}\n")
        f.write(f"test\t{len(te_ids)}\t{te_pairs}\n")
        
    print("\nValidating submission file...", flush=True)
    assert len(sub_df) == len(te_ids), f"Row count mismatch! Expected {len(te_ids)}, got {len(sub_df)}"
    assert sub_df['source1_entity_id'].nunique() == len(te_ids), "Duplicate S1 IDs!"
    assert list(sub_df.columns) == ['source1_entity_id', 'matched_entity_ids'], "Header mismatch!"
    print("Submission validation PASSED successfully!", flush=True)
    
    exp_log_path = os.path.join(SUBMISSIONS_DIR, "experiment_log.tsv")
    log_exists = os.path.exists(exp_log_path)
    with open(exp_log_path, "a", encoding="utf-8") as f:
        if not log_exists:
            f.write("experiment_id\tnormalization_rules\tmatching_rules\tval_precision\tval_recall\tval_f05\tpredicted_matches\tsingleton_acc\n")
        f.write(f"submission_01\tconservative_lowercase_unicode_suffixes_address\texact_name_ctry_and_addr_ctry\t{val_metrics['macro_precision']:.6f}\t{val_metrics['macro_recall']:.6f}\t{val_metrics['macro_f05']:.6f}\t{val_metrics['predicted_matches_count']}\t{val_metrics['singleton_acc']:.6f}\n")
        
    print(f"\nPipeline finished in {time.time()-t_main:.2f}s", flush=True)

if __name__ == "__main__":
    main()
