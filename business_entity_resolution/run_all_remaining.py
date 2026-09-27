import os
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}", flush=True)
    if os.system(cmd) != 0:
        print(f"Error executing: {cmd}")

def main():
    print("Running Part B: Day 1 Sub 5")
    run_cmd("python submission_05_pipeline.py")
    run_cmd("python validate_local.py submissions/day1/submission_05.tsv")
    
    print("Running Part C: Day 2 Sub 1")
    run_cmd("python day2_sub1.py")
    run_cmd("python validate_local.py submissions/day2/submission_01.tsv")
    
    print("Running Part D: Day 2 Sub 2")
    run_cmd("python day2_sub2.py")
    run_cmd("python validate_local.py submissions/day2/submission_02.tsv")
    
    print("Running Part E: Day 2 Sub 3")
    run_cmd("python day2_sub3.py")
    run_cmd("python validate_local.py submissions/day2/submission_03.tsv")
    
    print("Running Part F: Day 2 Sub 4")
    run_cmd("python day2_sub4.py")
    run_cmd("python validate_local.py submissions/day2/submission_04.tsv")
    
    print("Running Part G: Day 2 Sub 5")
    run_cmd("python day2_sub5.py")
    run_cmd("python validate_local.py submissions/day2/submission_05.tsv")
    
    print("All configured parts executed.", flush=True)

if __name__ == '__main__':
    main()
