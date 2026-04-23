"""
Step 1: Download & Explore Datasets
====================================
Paper: "Efficient RL for Optimizing Multi-turn Student Outcomes with LLM Tutors"

Datasets needed:
  1. GSM8K     - Math problems (used as tutoring scenarios + generalization eval)
  2. CoMTA     - Real student-tutor dialogues (used as few-shot examples)
"""

# ── Install required libraries ──────────────────────────────────────────────
# Run this in your terminal first:
#   pip install datasets pandas

from datasets import load_dataset
import pandas as pd
import json
import os

os.makedirs("data", exist_ok=True)

# ============================================================
# DATASET 1: GSM8K
# ============================================================
print("=" * 60)
print("Loading GSM8K dataset...")
print("=" * 60)

gsm8k = load_dataset("gsm8k", "main")

# Basic info
print(f"\nSplits available: {list(gsm8k.keys())}")
print(f"  Train size : {len(gsm8k['train'])} problems")
print(f"  Test  size : {len(gsm8k['test'])}  problems")

# Preview a sample
print("\n--- Sample problem (train[0]) ---")
sample = gsm8k["train"][0]
print(f"Question : {sample['question']}")
print(f"Answer   : {sample['answer']}")

# The 7 specific test problems used in the paper (Appendix 15)
# Question indices: 7, 12, 13, 15, 20, 37, 46
paper_eval_indices = [7, 12, 13, 15, 20, 37, 46]

print("\n--- 7 GSM8K test problems used in the paper ---")
paper_eval_problems = []
for idx in paper_eval_indices:
    problem = gsm8k["test"][idx]
    paper_eval_problems.append({
        "gsm8k_index": idx,
        "question": problem["question"],
        "answer": problem["answer"],
    })
    print(f"\nQ{idx}: {problem['question'][:100]}...")
    print(f"Answer: {problem['answer'].split('####')[-1].strip()}")

# Save eval problems
with open("data/gsm8k_eval_problems.json", "w") as f:
    json.dump(paper_eval_problems, f, indent=2)
print("\nSaved: data/gsm8k_eval_problems.json")

# Save full test set as CSV for exploration
gsm8k_test_df = pd.DataFrame(gsm8k["test"])
gsm8k_test_df.to_csv("data/gsm8k_test.csv", index=True)
print("Saved: data/gsm8k_test.csv")

# ============================================================
# DATASET 2: CoMTA (Khan Academy student-tutor dialogues)
# ============================================================
print("\n" + "=" * 60)
print("Loading CoMTA dataset...")
print("=" * 60)

# NOTE: If 'millerls/CoMTA' is not available on HuggingFace,
# see the alternative loading section below.
try:
    comta = load_dataset("millerls/CoMTA")
    print(f"\nSplits available: {list(comta.keys())}")

    # Show a sample dialogue
    sample_dialogue = comta["train"][0]
    print("\n--- Sample CoMTA dialogue ---")
    print(f"Columns: {list(sample_dialogue.keys())}")
    print(f"Sample: {str(sample_dialogue)[:400]}...")

    # Save for later use as few-shot examples
    comta_df = pd.DataFrame(comta["train"])
    comta_df.to_csv("data/comta_dialogues.csv", index=False)
    print("Saved: data/comta_dialogues.csv")

except Exception as e:
    print(f"\nCould not load CoMTA from HuggingFace: {e}")
    print("\nAlternative: The CoMTA dataset is from this paper:")
    print("  Miller & DiCerbo (2024) - 'LLM Based Math Tutoring: Challenges and Dataset'")
    print("  You may need to request it directly from the authors.")
    print("\nFor now, we will create a placeholder few-shot example structure")
    print("matching the format used in the paper's prompts.\n")

    # Create placeholder few-shot examples matching paper's format
    placeholder_examples = [
        {
            "dialogue_id": "placeholder_1",
            "problem": "If x + 5 = 12, what is x?",
            "conversation": [
                {"role": "Student", "text": "I'm stuck on this problem."},
                {"role": "Tutor",   "text": "Let's think about it together. What does the equation tell you?"},
                {"role": "Student", "text": "It says x plus 5 equals 12?"},
                {"role": "Tutor",   "text": "Exactly! If you take away 5 from both sides, what do you get?"},
                {"role": "Student", "text": "x equals 7?"},
                {"role": "Tutor",   "text": "That's right! Great work."},
            ]
        },
        {
            "dialogue_id": "placeholder_2",
            "problem": "What is 30% of 200?",
            "conversation": [
                {"role": "Student", "text": "How do I solve this?"},
                {"role": "Tutor",   "text": "What does 'percent' mean to you?"},
                {"role": "Student", "text": "Out of 100?"},
                {"role": "Tutor",   "text": "Correct. So 30% means 30 out of 100. Can you write that as a fraction?"},
                {"role": "Student", "text": "30/100, so 0.3?"},
                {"role": "Tutor",   "text": "Perfect! Now multiply 0.3 × 200."},
                {"role": "Student", "text": "That's 60!"},
                {"role": "Tutor",   "text": "Excellent job!"},
            ]
        },
    ]

    with open("data/comta_placeholder.json", "w") as f:
        json.dump(placeholder_examples, f, indent=2)
    print("Saved placeholder few-shot examples: data/comta_placeholder.json")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 1 COMPLETE — Dataset Summary")
print("=" * 60)
print(f"  GSM8K Train      : {len(gsm8k['train'])} problems -> used to generate 3,000 synthetic dialogues")
print(f"  GSM8K Test (7)   : saved to data/gsm8k_eval_problems.json -> used for generalization eval")
print(f"  CoMTA            : real student-tutor dialogues -> used as few-shot examples in prompts")
print("\nNext step -> Step 2: Synthetic Dialogue Generation")
