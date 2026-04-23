import os
import json
import time
import random
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from mistralai import Mistral

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "B3lMQNsxQMCSe6BrmkbRy7ICigJavCwV")

MODEL_NAME      = "mistral-small-latest"
NUM_DIALOGUES   = 3000
MAX_TURNS       = 12
DISCOUNT_FACTOR = 0.9

OUTPUT_DIR      = Path("data/dialogues")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PATH       = OUTPUT_DIR / "synthetic_dialogues.jsonl"

COMTA_CSV        = Path("data/comta_dialogues.csv")
COMTA_CACHE_JSON = Path("data/comta_fewshot.json")

client = Mistral(api_key=MISTRAL_API_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# Few-shot loader
# ──────────────────────────────────────────────────────────────────────────────
def _placeholder_shots() -> list[str]:
    return [
        (
            "Tutor: Let's think about this step by step. "
            "What information do you have?\n"
            "Student: The car went from A to B at 30 mph.\n"
            "Tutor: Good! And what else does the problem tell you?\n"
            "Student: It came back at 50 mph.\n"
            "Tutor: Exactly. So can you write an equation for the time?\n"
            "Student: Time equals distance divided by speed?\n"
            "Tutor: Perfect — now apply that to each leg of the journey.\n"
            "Student: d/30 plus d/50!\n"
            "Tutor: That's right — great work!"
        ),
        (
            "Tutor: What does the problem ask you to find?\n"
            "Student: I'm not sure, I'm confused.\n"
            "Tutor: No worries! Let's read it together. "
            "What's the key word in the question?\n"
            "Student: Total?\n"
            "Tutor: Yes! So we need to find the total. "
            "What values do we start with?\n"
            "Student: 15 and 23?\n"
            "Tutor: Great — now what operation gives us the total?\n"
            "Student: Addition! So 15 plus 23 equals 38.\n"
            "Tutor: Excellent work!"
        ),
    ]


def load_few_shots(n_examples: int = 5) -> list[str]:
    if COMTA_CACHE_JSON.exists():
        with open(COMTA_CACHE_JSON, "r", encoding="utf-8") as f:
            shots = json.load(f)
        print(f"Loaded {len(shots)} CoMTA few-shot examples from cache.")
        return shots[:n_examples]

    if not COMTA_CSV.exists():
        print("WARNING: data/comta_dialogues.csv not found.")
        print("Using placeholder few-shot examples.")
        return _placeholder_shots()

    df = pd.read_csv(COMTA_CSV)
    print(f"Reading CoMTA CSV with {len(df)} rows")

    shots = []

    cols      = [c.lower() for c in df.columns]
    orig_cols = list(df.columns)

    def col(keyword):
        for i, c in enumerate(cols):
            if keyword in c:
                return orig_cols[i]
        return None

    tutor_col   = col("tutor")
    student_col = col("student")
    role_col    = col("role") or col("speaker") or col("who")
    msg_col     = col("message") or col("text") or col("content") or col("utterance")
    id_col      = col("dialogue_id") or col("session") or col("id")
    text_col    = col("dialogue") or col("conversation")

    if tutor_col and student_col:
        for _, row in df.iterrows():
            tutor_text   = str(row[tutor_col]).strip()
            student_text = str(row[student_col]).strip()

            if tutor_text and student_text and tutor_text != "nan" and student_text != "nan":
                shots.append(f"Tutor: {tutor_text}\nStudent: {student_text}")

            if len(shots) >= n_examples:
                break

    elif id_col and role_col and msg_col:
        for dial_id, grp in df.groupby(id_col):
            lines = []

            for _, row in grp.iterrows():
                role = str(row[role_col]).lower().strip()
                msg  = str(row[msg_col]).strip()

                if not msg or msg == "nan":
                    continue

                if any(k in role for k in ["tutor", "teacher", "assistant", "agent"]):
                    lines.append(f"Tutor: {msg}")
                elif any(k in role for k in ["student", "learner", "user"]):
                    lines.append(f"Student: {msg}")

            if lines:
                shots.append("\n".join(lines))

            if len(shots) >= n_examples:
                break

    elif text_col:
        for _, row in df.head(n_examples).iterrows():
            text = str(row[text_col]).strip()
            if text and text != "nan":
                shots.append(text)

    if not shots:
        print("Could not parse CSV correctly. Using placeholder examples.")
        return _placeholder_shots()

    with open(COMTA_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(shots, f, indent=2, ensure_ascii=False)

    print(f"Loaded {len(shots)} few-shot examples from CSV.")
    return shots[:n_examples]


# ──────────────────────────────────────────────────────────────────────────────
# Mistral helper
# ──────────────────────────────────────────────────────────────────────────────
def call_mistral(prompt: str, max_tokens: int = 1200) -> str:
    for attempt in range(5):
        try:
            response = client.chat.complete(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            err = str(e).lower()

            if "429" in err or "rate" in err or "limit" in err:
                wait_time = 30 * (attempt + 1)
                print(f"Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Mistral error: {e}")
                time.sleep(5)

    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder (single API call)
# ──────────────────────────────────────────────────────────────────────────────
def build_dialogue_prompt(problem: str, few_shots: list[str]) -> str:
    few_shot_str = "\n\n".join(few_shots)

    return f"""
You are generating synthetic tutoring conversations for reinforcement learning research.

Your task:
1. First invent ONE realistic math mistake a weak sixth-grade student may make.
2. Then generate a dialogue between a Tutor and Student.
3. The student struggles with the problem and makes that mistake.
4. The tutor asks questions, identifies the misunderstanding, and helps the student solve it.
5. The tutor should not directly reveal the final answer immediately.
6. The student should eventually reach the correct answer.
7. Keep the dialogue under {MAX_TURNS} student turns.
8. Use exactly this format:

Mistake: <short description>

Tutor: ...
Student: ...
Tutor: ...
Student: ...

Math problem:
{problem}

Example tutoring dialogues:
{few_shot_str}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Dialogue parser
# ──────────────────────────────────────────────────────────────────────────────
def parse_dialogue(raw_text: str, correct_answer: str) -> dict:
    mistake_match = re.search(r"Mistake:\s*(.+)", raw_text)
    mistake = (
        mistake_match.group(1).strip()
        if mistake_match
        else "The student makes a calculation mistake."
    )

    dialogue_text = re.sub(r"Mistake:\s*.+", "", raw_text, count=1).strip()

    turns = []
    current_role = None

    parts = re.split(r"(Tutor:|Student:)", dialogue_text)

    for part in parts:
        part = part.strip()

        if part == "Tutor:":
            current_role = "tutor"

        elif part == "Student:":
            current_role = "student"

        elif part and current_role:
            reward = 0

            if current_role == "student":
                numbers = re.findall(r"-?\d+\.?\d*", part)

                clean_answer = re.sub(
                    r"[^0-9.]",
                    "",
                    correct_answer.split("####")[-1].strip()
                )

                if numbers and clean_answer and clean_answer in numbers:
                    reward = 1

            turns.append({
                "role": current_role,
                "text": part,
                "reward": reward
            })

            current_role = None

    turns = turns[:MAX_TURNS * 2]

    solved = any(
        t["reward"] == 1
        for t in turns
        if t["role"] == "student"
    )

    discounted_value = 0.0
    student_turns = 0

    for t in turns:
        if t["role"] == "student":
            student_turns += 1
            discounted_value += (DISCOUNT_FACTOR ** student_turns) * t["reward"]

    return {
        "student_mistake": mistake,
        "turns": turns,
        "solved": solved,
        "final_reward": 1 if solved else -1,
        "discounted_value": round(discounted_value, 4),
        "num_turns": student_turns,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Generate one dialogue
# ──────────────────────────────────────────────────────────────────────────────
def generate_dialogue(idx: int, problem: dict, few_shot_dialogues: list[str]) -> dict | None:
    question = problem["question"]
    answer   = problem["answer"]

    prompt = build_dialogue_prompt(question, few_shot_dialogues)

    raw_dialogue = call_mistral(prompt, max_tokens=1200)

    if not raw_dialogue:
        return None

    parsed = parse_dialogue(raw_dialogue, answer)

    return {
        "dialogue_id": f"dial_{idx:05d}",
        "gsm8k_question": question,
        "correct_answer": answer,
        "student_mistake": parsed["student_mistake"],
        "raw_dialogue": raw_dialogue,
        "turns": parsed["turns"],
        "solved": parsed["solved"],
        "final_reward": parsed["final_reward"],
        "discounted_value": parsed["discounted_value"],
        "num_turns": parsed["num_turns"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Resume support
# ──────────────────────────────────────────────────────────────────────────────
def load_existing_ids() -> set:
    if not SAVE_PATH.exists():
        return set()

    existing_ids = set()

    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                existing_ids.add(row["dialogue_id"])
            except Exception:
                pass

    return existing_ids


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading GSM8K dataset...")
    gsm8k = load_dataset("gsm8k", "main")
    train_problems = list(gsm8k["train"])

    print("Loading CoMTA few-shot examples...")
    few_shot_dialogues = load_few_shots(n_examples=5)

    print("\nFew-shot preview:")
    for i, shot in enumerate(few_shot_dialogues[:2]):
        print(f"\nExample {i+1}:\n{shot[:200]}...")

    existing_ids = load_existing_ids()

    print(f"\nAlready completed: {len(existing_ids)} dialogues")
    print(f"Target total      : {NUM_DIALOGUES}")

    random.seed(42)
    sampled_problems = random.choices(train_problems, k=NUM_DIALOGUES)

    with open(SAVE_PATH, "a", encoding="utf-8") as f:
        for idx, problem in enumerate(sampled_problems):
            dialogue_id = f"dial_{idx:05d}"

            if dialogue_id in existing_ids:
                print(f"[{idx+1}/{NUM_DIALOGUES}] Skipping {dialogue_id}")
                continue

            print(f"[{idx+1}/{NUM_DIALOGUES}] Generating {dialogue_id}...")

            dialogue = generate_dialogue(
                idx=idx,
                problem=problem,
                few_shot_dialogues=few_shot_dialogues
            )

            if dialogue is None:
                print("  Failed. Skipping.")
                time.sleep(3)
                continue

            f.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

            status = "SOLVED" if dialogue["solved"] else "FAILED"

            print(
                f"  {status} | turns={dialogue['num_turns']} "
                f"| reward={dialogue['final_reward']}"
            )

            time.sleep(1.2)

    print("\nLoading saved dialogues for summary...")

    dialogues = []

    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                dialogues.append(json.loads(line))
            except Exception:
                pass

    solved = [d for d in dialogues if d["solved"]]

    avg_turns = (
        sum(d["num_turns"] for d in dialogues) / max(len(dialogues), 1)
    )

    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)
    print(f"Total dialogues : {len(dialogues)}")
    print(f"Solved dialogues: {len(solved)}")
    print(f"Solve rate      : {100 * len(solved) / max(len(dialogues), 1):.2f}%")
    print(f"Average turns   : {avg_turns:.2f}")
    print(f"Saved to        : {SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()