"""
Master Runner — Full Replication Pipeline
==========================================
Paper: "Efficient RL for Optimizing Multi-turn Student Outcomes with LLM Tutors"

Run this to execute all steps in order:
  python run_all.py

Or run individual steps:
  python run_all.py --step 1
  python run_all.py --step 2
  ...

Prerequisites:
  pip install anthropic datasets scikit-learn numpy pandas matplotlib d3rlpy torch
  export ANTHROPIC_API_KEY="sk-ant-..."
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    (1, "step1_download_datasets.py",    "Download GSM8K + CoMTA datasets"),
    (2, "step2_generate_dialogues.py",   "Generate 3,000 synthetic dialogues"),
    (3, "step3_extract_states_actions.py","Extract 25-dim states + action labels"),
    (4, "step4_train_rl_policies.py",    "Train BC, FQI, CQL policies"),
    (5, "step5a_augment_data.py",        "Optimism-guided data augmentation (D+)"),
    (6, "step5b_evaluate.py",            "Student simulator evaluation → Figure 3"),
]

def run_step(script: str, description: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Script: {script}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n  DONE in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  FAILED (exit code {result.returncode})")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=None,
                        help="Run only this step (1-6). Omit to run all.")
    parser.add_argument("--from-step", type=int, default=1,
                        help="Start from this step.")
    args = parser.parse_args()

    steps_to_run = STEPS
    if args.step:
        steps_to_run = [s for s in STEPS if s[0] == args.step]
    elif args.from_step > 1:
        steps_to_run = [s for s in STEPS if s[0] >= args.from_step]

    print("\nReplication Pipeline — Starting")
    print(f"Steps to run: {[s[0] for s in steps_to_run]}\n")

    for step_num, script, desc in steps_to_run:
        print(f"\n[Step {step_num}/6] {desc}")
        success = run_step(script, desc)
        if not success:
            print(f"\nPipeline stopped at step {step_num}.")
            print("Fix the error above and resume with:")
            print(f"  python run_all.py --from-step {step_num}")
            sys.exit(1)

    print("\n" + "="*60)
    print("  REPLICATION COMPLETE")
    print("  Results in: results/evaluation_results.json")
    print("  Figure 3:   results/figure3_reproduction.png")
    print("="*60)

if __name__ == "__main__":
    main()
