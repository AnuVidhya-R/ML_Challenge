import os
import sys
import pandas as pd

BASE_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution"
TEST_S1_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source1.tsv")
TEST_S2_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source2.tsv")
TEST_S3_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source3.tsv")

def main():
    print("=== LOADING VALIDATION TEST SET INDEXES ===")
    df_s1 = pd.read_csv(TEST_S1_PATH, sep='\t', quoting=3, usecols=['entity_id'], dtype=str)
    expected_s1_ids = set(df_s1['entity_id'].dropna().str.strip())
    expected_count = len(expected_s1_ids)

    df_s2 = pd.read_csv(TEST_S2_PATH, sep='\t', quoting=3, usecols=['entity_id'], dtype=str)
    df_s3 = pd.read_csv(TEST_S3_PATH, sep='\t', quoting=3, usecols=['entity_id'], dtype=str)
    valid_s23_ids = set(df_s2['entity_id'].dropna().str.strip()).union(set(df_s3['entity_id'].dropna().str.strip()))

    candidates = [
        ("Day1 #1", os.path.join(BASE_DIR, "submissions", "day1", "submission_01.tsv")),
        ("Day1 #2", os.path.join(BASE_DIR, "submissions", "day1", "submission_02.tsv")),
        ("Day1 #3 / Day2 #2", os.path.join(BASE_DIR, "submissions", "day1", "submission_03.tsv")),
        ("Day1 #4 / #5", os.path.join(BASE_DIR, "submissions", "day1", "submission_04.tsv")),
        ("Day2 #1 / #3 / #4 / #5", os.path.join(BASE_DIR, "submissions", "day2", "submission_01.tsv")),
    ]

    print("\n=== RUNNING FAST VALIDATION ACROSS ALL CANDIDATE SUBMISSIONS ===")
    results = {}

    for label, sub_path in candidates:
        print(f"\nValidating {label}: {os.path.basename(sub_path)}...")
        errors = []
        total_rows = 0
        dup_s1 = 0
        seen_s1 = set()
        non_empty = 0
        empty = 0
        invalid_target_count = 0

        # Check header
        df_head = pd.read_csv(sub_path, sep='\t', quoting=3, nrows=2, dtype=str)
        if list(df_head.columns) != ['source1_entity_id', 'matched_entity_ids']:
            errors.append(f"Header invalid: {list(df_head.columns)}")

        for chunk in pd.read_csv(sub_path, sep='\t', quoting=3, chunksize=500000, dtype=str):
            total_rows += len(chunk)
            s1_col = chunk['source1_entity_id'].fillna('').str.strip()
            t_col = chunk['matched_entity_ids'].fillna('').str.strip()

            for s1_id, t_str in zip(s1_col, t_col):
                if s1_id in seen_s1:
                    dup_s1 += 1
                seen_s1.add(s1_id)

                if not t_str:
                    empty += 1
                else:
                    non_empty += 1
                    t_ids = [t.strip() for t in t_str.split(',') if t.strip()]
                    for t_id in t_ids:
                        if t_id in expected_s1_ids or t_id not in valid_s23_ids:
                            invalid_target_count += 1
                            if len(errors) < 5:
                                errors.append(f"Invalid target ID {t_id} for S1={s1_id}")

        if total_rows != expected_count:
            errors.append(f"Total rows {total_rows} != expected {expected_count}")
        if len(seen_s1) != expected_count:
            errors.append(f"Unique S1 count {len(seen_s1)} != expected {expected_count}")
        if dup_s1 > 0:
            errors.append(f"Duplicate S1 count: {dup_s1}")
        if invalid_target_count > 0:
            errors.append(f"Invalid target IDs count: {invalid_target_count}")

        status = "PASS" if len(errors) == 0 else f"FAIL ({len(errors)} errors)"
        results[label] = {
            'status': status,
            'total_rows': total_rows,
            'unique_s1': len(seen_s1),
            'dup_s1': dup_s1,
            'non_empty': non_empty,
            'empty': empty,
            'invalid_targets': invalid_target_count,
            'errors': errors
        }
        print(f"Result for {label}: {status} (Non-empty: {non_empty:,}, Empty: {empty:,})")

    print("\n================ FINAL VALIDATION SUMMARY ================")
    for label, res in results.items():
        print(f"{label:<25} | Status: {res['status']:<15} | Non-empty: {res['non_empty']:>10,} | Empty: {res['empty']:>10,}")

if __name__ == "__main__":
    main()
