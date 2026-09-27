import os
import glob
import hashlib
import pandas as pd
from collections import defaultdict

BASE_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution"
DATA_DIR = os.path.join(BASE_DIR, "dataset")
SUB_DIR = os.path.join(BASE_DIR, "submissions")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def count_lines(filepath):
    """Fast line count (subtracting header = row count)"""
    with open(filepath, 'rb') as f:
        lines = 0
        buf_size = 1024 * 1024
        read_res = f.raw.read if hasattr(f, 'raw') else f.read
        buf = read_res(buf_size)
        while buf:
            lines += buf.count(b'\n')
            buf = read_res(buf_size)
    return max(0, lines - 1)

def compute_file_hash(filepath, chunk_size=10*1024*1024):
    """Compute MD5 hash of file to find identical files."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def inspect_submission(filepath):
    """Chunked inspection of submission TSV file."""
    total_rows = 0
    non_empty_preds = 0
    empty_preds = 0
    s1_ids_seen = set()
    has_duplicates = False
    dup_count = 0
    
    # Read first chunk to inspect header
    df_head = pd.read_csv(filepath, sep='\t', quoting=3, nrows=2, dtype=str)
    cols = list(df_head.columns)
    valid_cols = (cols == ['source1_entity_id', 'matched_entity_ids'])
    
    for chunk in pd.read_csv(filepath, sep='\t', quoting=3, chunksize=200000, dtype=str):
        total_rows += len(chunk)
        
        # Check matched_entity_ids column (col index 1)
        matched_col = chunk['matched_entity_ids'].fillna('').str.strip()
        non_empty = (matched_col != '').sum()
        empty = len(chunk) - non_empty
        non_empty_preds += non_empty
        empty_preds += empty
        
        # S1 IDs
        s1_series = chunk['source1_entity_id'].astype(str)
        # Check duplicates in chunk vs seen
        chunk_unique = set(s1_series)
        if len(chunk_unique) < len(s1_series):
            dup_count += (len(s1_series) - len(chunk_unique))
        
        overlap = chunk_unique.intersection(s1_ids_seen)
        if overlap:
            dup_count += len(overlap)
        s1_ids_seen.update(chunk_unique)
        
    return {
        'total_rows': total_rows,
        'cols_valid': valid_cols,
        'cols': cols,
        'non_empty_preds': non_empty_preds,
        'empty_preds': empty_preds,
        'unique_s1_count': len(s1_ids_seen),
        'duplicate_s1_count': dup_count
    }

def main():
    print("=== AUDITING DATASET FILES ===")
    dataset_info = {}
    for split in ['train', 'test']:
        split_dir = os.path.join(DATA_DIR, split)
        for fname in sorted(os.listdir(split_dir)):
            if fname.endswith('.tsv'):
                fpath = os.path.join(split_dir, fname)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                rows = count_lines(fpath)
                dataset_info[fname] = {'rows': rows, 'size_mb': round(size_mb, 2)}
                print(f"Dataset {fname}: {rows:,} rows, {size_mb:.2f} MB")

    print("\n=== AUDITING SUBMISSION FILES ===")
    sub_files = []
    for root, dirs, files in os.walk(SUB_DIR):
        for f in files:
            if f.endswith('.tsv') and f != 'experiment_log.tsv':
                sub_files.append(os.path.join(root, f))
    sub_files = sorted(sub_files)

    hashes = {}
    sub_results = {}

    for fpath in sub_files:
        rel_path = os.path.relpath(fpath, BASE_DIR)
        print(f"Auditing {rel_path}...")
        f_hash = compute_file_hash(fpath)
        hashes[rel_path] = f_hash
        
        info = inspect_submission(fpath)
        info['rel_path'] = rel_path
        info['hash'] = f_hash
        info['file_size_mb'] = round(os.path.getsize(fpath) / (1024*1024), 2)
        sub_results[rel_path] = info

    # Group identical files by hash
    hash_to_files = defaultdict(list)
    for rel_path, h in hashes.items():
        hash_to_files[h].append(rel_path)

    # Output report
    report_path = os.path.join(REPORTS_DIR, "final_submission_audit.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=============================================================\n")
        f.write("FINAL SUBMISSION AUDIT REPORT\n")
        f.write("=============================================================\n\n")

        f.write("1. DATASET METRICS\n")
        f.write("-" * 60 + "\n")
        for fname, info in dataset_info.items():
            f.write(f"{fname:<25}: {info['rows']:>10,} rows ({info['size_mb']:>7.2f} MB)\n")

        f.write("\n2. SUBMISSION FILES AUDIT\n")
        f.write("-" * 60 + "\n")
        header = f"{'Submission File':<32} | {'Rows':<9} | {'Dup S1':<6} | {'Non-Empty':<9} | {'Empty':<9} | {'Valid Cols':<10}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        for rel_path, info in sub_results.items():
            f.write(f"{rel_path:<32} | {info['total_rows']:<9,} | {info['duplicate_s1_count']:<6} | {info['non_empty_preds']:<9,} | {info['empty_preds']:<9,} | {str(info['cols_valid']):<10}\n")

        f.write("\n3. IDENTICAL SUBMISSIONS (HASH COMPARISON)\n")
        f.write("-" * 60 + "\n")
        group_idx = 1
        for h, f_list in hash_to_files.items():
            if len(f_list) > 1:
                f.write(f"Group {group_idx} (Identical MD5: {h}):\n")
                for item in f_list:
                    f.write(f"  - {item}\n")
                group_idx += 1
            else:
                f.write(f"Unique submission: {f_list[0]} (MD5: {h})\n")

    print(f"\nAudit completed! Report saved to {report_path}")

if __name__ == "__main__":
    main()
