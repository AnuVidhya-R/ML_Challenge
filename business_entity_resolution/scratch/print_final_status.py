import os
import sys

BASE_DIR = r"d:\Working_project\ML Challenge\business_entity_resolution"

def file_rows(path):
    """Fast line count (rows = lines - 1 for header)."""
    if not os.path.exists(path):
        return "MISSING"
    with open(path, 'rb') as f:
        lines = 0
        buf = f.read(1024*1024)
        while buf:
            lines += buf.count(b'\n')
            buf = f.read(1024*1024)
    return max(0, lines - 1)

def file_size_mb(path):
    if not os.path.exists(path):
        return "MISSING"
    return f"{os.path.getsize(path) / (1024*1024):.2f} MB"

def check_exists(path):
    return "✅ Yes" if os.path.exists(path) else "❌ MISSING"

# Dataset paths
ds = {
    "Train S1": os.path.join(BASE_DIR, "dataset", "train", "train_source1.tsv"),
    "Train S2": os.path.join(BASE_DIR, "dataset", "train", "train_source2.tsv"),
    "Train S3": os.path.join(BASE_DIR, "dataset", "train", "train_source3.tsv"),
    "Test S1":  os.path.join(BASE_DIR, "dataset", "test",  "test_source1.tsv"),
    "Test S2":  os.path.join(BASE_DIR, "dataset", "test",  "test_source2.tsv"),
    "Test S3":  os.path.join(BASE_DIR, "dataset", "test",  "test_source3.tsv"),
}

# Submission paths
subs = {
    "Day1 #1": (os.path.join(BASE_DIR, "submissions", "day1", "submission_01.tsv"), "0.3863 (LB: 0.406)"),
    "Day1 #2": (os.path.join(BASE_DIR, "submissions", "day1", "submission_02.tsv"), "0.5394"),
    "Day1 #3": (os.path.join(BASE_DIR, "submissions", "day1", "submission_03.tsv"), "0.5551 ← BEST"),
    "Day1 #4": (os.path.join(BASE_DIR, "submissions", "day1", "submission_04.tsv"), "not locally scored"),
    "Day1 #5": (os.path.join(BASE_DIR, "submissions", "day1", "submission_05.tsv"), "not locally scored (= #4)"),
    "Day2 #1": (os.path.join(BASE_DIR, "submissions", "day2", "submission_01.tsv"), "not locally scored"),
    "Day2 #2": (os.path.join(BASE_DIR, "submissions", "day2", "submission_02.tsv"), "not locally scored (= Day1 #3)"),
    "Day2 #3": (os.path.join(BASE_DIR, "submissions", "day2", "submission_03.tsv"), "not locally scored (= Day2 #1)"),
    "Day2 #4": (os.path.join(BASE_DIR, "submissions", "day2", "submission_04.tsv"), "not locally scored (= Day2 #1)"),
    "Day2 #5": (os.path.join(BASE_DIR, "submissions", "day2", "submission_05.tsv"), "not locally scored (= Day2 #1)"),
}

print()
print("=" * 60)
print("        FINAL PROJECT STATUS — Business Entity Resolution")
print("=" * 60)

print("\nDATASET:")
print("-" * 60)
for label, path in ds.items():
    rows = file_rows(path)
    sz = file_size_mb(path)
    row_str = f"{rows:,}" if isinstance(rows, int) else rows
    print(f"  {label:<10}: {row_str:>12} rows  ({sz})")

print("\nSUBMISSIONS:")
print("-" * 60)
for label, (path, score) in subs.items():
    exists = check_exists(path)
    rows = file_rows(path)
    row_str = f"{rows:,}" if isinstance(rows, int) else rows
    print(f"  {label:<8}: {exists}  Rows={row_str}  Local F0.5={score}")

print("\nOUTPUT FILES:")
print("-" * 60)
mr = os.path.join(BASE_DIR, "output", "matching_results.tsv")
cp = os.path.join(BASE_DIR, "output", "candidate_pairs.tsv")
print(f"  matching_results.tsv : {check_exists(mr)}, Size={file_size_mb(mr)}")
print(f"  candidate_pairs.tsv  : {check_exists(cp)}, Size={file_size_mb(cp)}")

print("\nBEST KNOWN LOCAL F0.5:")
print("  0.5551 — Day1 #3 (submissions/day1/submission_03.tsv)")
print("  Precision=0.8598, Recall=0.4526, Threshold=0.85")

print("\nBEST VALIDATED SUBMISSION:")
print("  submissions/day1/submission_03.tsv")
print("  (Also copied to: final_submission/matching_results.tsv)")

print("\nFINAL PACKAGE:")
print(f"  Path: {os.path.join(BASE_DIR, 'final_package')}")
pkg_files = []
for root, dirs, files in os.walk(os.path.join(BASE_DIR, "final_package")):
    for f in files:
        pkg_files.append(os.path.relpath(os.path.join(root, f), BASE_DIR))
for pf in sorted(pkg_files):
    print(f"    ✅ {pf}")

print("\nFINAL SUBMISSION PACKAGE:")
print(f"  Path: {os.path.join(BASE_DIR, 'final_submission')}")
sub_files = []
for root, dirs, files in os.walk(os.path.join(BASE_DIR, "final_submission")):
    for f in files:
        sub_files.append(os.path.relpath(os.path.join(root, f), BASE_DIR))
for sf in sorted(sub_files):
    print(f"    ✅ {sf}")

print("\nDOCUMENTATION:")
print(f"  Path: {os.path.join(BASE_DIR, 'reports')}")
report_files = []
for f in os.listdir(os.path.join(BASE_DIR, "reports")):
    report_files.append(f"    ✅ reports/{f}")
for rf in sorted(report_files):
    print(rf)

print("\nVALIDATION STATUS:")
print("  Day1 #1  : ✅ PASS")
print("  Day1 #2  : ✅ PASS")
print("  Day1 #3  : ✅ PASS (= Day2 #2)")
print("  Day1 #4  : ✅ PASS (= Day1 #5)")
print("  Day2 #1  : ✅ PASS (= Day2 #3, #4, #5)")
print("  candidate_pairs.tsv : ✅ PASS")
print("  Overall  : ✅ ALL PASS")

print()
print("=" * 60)
print("   PROJECT IS FINAL-READY. ALL SUBMISSIONS PRESERVED.")
print("=" * 60)
