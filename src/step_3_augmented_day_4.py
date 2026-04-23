"""
Step 3 — DAY 4 VERSION (Augmented Data)
========================================
Reads:  data/dialogues/augmented_dialogues.jsonl
Saves:  data/rl_dataset_augmented.csv

Key fix: augmented dialogues store conversation differently
  - Original format  → has 'turns' list directly
  - Augmented format → has 'partial_history' + 'continuation' text

This script handles augmented format correctly and saves
to a SEPARATE CSV so original rl_dataset.csv is not overwritten.

Run:
  python step3_day4_augmented.py

Then run step4_day4_combined.py to train CQL on D+.
"""

import os
import json
import time
import re
import numpy as np
import pandas as pd
from pathlib import Path
from mistralai import Mistral

# ── Configuration ──────────────────────────────────────────────────────────────
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "B3lMQNsxQMCSe6BrmkbRy7ICigJavCwV")
MODEL_NAME      = "mistral-small-latest"

# ── Input / Output paths ───────────────────────────────────────────────────────
INPUT_JSONL  = Path("data/dialogues/augmented_dialogues.jsonl")   # augmented only
OUTPUT_CSV   = Path("data/rl_dataset_augmented.csv")              # separate file
CHECKPOINT   = Path("data/.checkpoint_aug_step3.json")            # resume tracking

client    = Mistral(api_key=MISTRAL_API_KEY)
N_ACTIONS = 4
ACTION_LABELS = {
    "instruct": 0, "encourage": 1,
    "refocus":  2, "ask_question": 3,
}
ACTION_NAMES = {v: k for k, v in ACTION_LABELS.items()}


# ── Mistral API helper ─────────────────────────────────────────────────────────
def call_mistral(prompt: str, max_tokens: int = 80) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.complete(
                model       = MODEL_NAME,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = max_tokens,
                temperature = 0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in err:
                wait = 30 * (attempt + 1)
                print(f"  Rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Mistral error: {e}")
                time.sleep(5)
    return ""


# ── KEY FIX: Parse augmented dialogue format ───────────────────────────────────
def parse_augmented_turns(dialogue: dict) -> list[dict]:
    """
    Augmented dialogues don't have a 'turns' list.
    They have 'partial_history' (str) + 'continuation' (str).
    This function rebuilds the turns list from those two text fields.
    """

    # If original format with turns list — use directly
    if "turns" in dialogue and isinstance(dialogue["turns"], list) \
            and len(dialogue["turns"]) > 0:
        return dialogue["turns"]

    # Augmented format — rebuild from text
    raw_text = ""
    if "partial_history" in dialogue:
        raw_text += dialogue["partial_history"].strip()
    if "continuation" in dialogue:
        raw_text += "\n" + dialogue["continuation"].strip()

    if not raw_text.strip():
        return []

    # Clean up generation tags
    raw_text = re.sub(r"\[Generation\]\s*", "", raw_text)

    # Split on Tutor:/Student: markers
    parts        = re.split(r"((?:^|\n)(?:Tutor|Student)\s*:)", raw_text,
                            flags=re.MULTILINE)
    turns        = []
    current_role = None

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^Tutor\s*:$", part):
            current_role = "tutor"
        elif re.match(r"^Student\s*:$", part):
            current_role = "student"
        elif current_role:
            # Check reward for student turns
            reward  = 0
            correct = dialogue.get("correct_answer", "")
            if current_role == "student" and correct:
                clean   = re.sub(r"[^0-9.]", "",
                                 correct.split("####")[-1].strip())
                numbers = re.findall(r"-?\d+\.?\d*", part)
                if clean and clean in numbers:
                    reward = 1
            turns.append({
                "role":   current_role,
                "text":   part,
                "reward": reward,
            })
            current_role = None

    return turns


# ── State extraction (same as Step 3 original) ────────────────────────────────
STATE_QUESTIONS = [
    "Is the student producing math-related content?",
    "Has the student solved the problem correctly?",
    "Is the student asking the tutor to re-explain a concept?",
    "Is the student repeating what the tutor already said?",
    "Is the student going off-topic?",
    "Is the student's utterance unrelated to the math problem?",
    "Is the student explicitly asking the tutor a question?",
    "Is the student describing what they are stuck on?",
    "Has the tutor asked diagnostic questions about the student's math level?",
    "Is the student expressing frustration?",
    "Is the student expressing uncertainty or low confidence?",
    "Is the student expressing positive sentiment?",
    "Is the student asking for a break from tutoring?",
    "Is the student talking about the problem at hand?",
    "Is the student talking about their general math background?",
    "Is the student discussing other math concepts related to the problem?",
    "Has the student written down an equation?",
    "Is the tutor asking a question to the student?",
    "Did the student make a mistake in the current turn?",
    "Has the tutor tried to bring the student's focus back to the problem?",
]


def extract_state_vector(history: str, student_text: str,
                         turn_n: int, tq: int, sq: int) -> np.ndarray:
    numbered = "\n".join(
        f"Q{i+1}: {q} Answer yes or no."
        for i, q in enumerate(STATE_QUESTIONS)
    )
    prompt = (
        "Evaluate this tutoring dialogue.\n\n"
        f"Dialogue:\n{history}\n\n"
        f"Student's latest utterance: \"{student_text}\"\n\n"
        "Answer ONLY yes or no, one per line:\n\n"
        f"{numbered}"
    )
    raw     = call_mistral(prompt, max_tokens=60)
    answers = []
    for line in raw.strip().split("\n"):
        line = re.sub(r"^[qQ\d]+[\.\:\)]\s*", "", line.strip().lower())
        if "yes" in line:
            answers.append(1)
        elif "no" in line:
            answers.append(0)
        if len(answers) == 20:
            break
    while len(answers) < 20:
        answers.append(0)

    # Dims 24-25: heuristic math density + reasoning
    math_tokens = re.findall(r"[\d\+\-\*\/\=\%\^\(\)]", student_text)
    words       = student_text.split()
    density     = round(len(math_tokens) / max(len(words), 1), 4)
    reason_kw   = {"because","so","therefore","since","if",
                   "then","means","equals","thus"}
    reasoning   = round(
        min(len(set(student_text.lower().split()) & reason_kw) / 5, 1.0), 4
    )

    return np.array(
        answers + [float(tq), float(sq), float(turn_n), density, reasoning],
        dtype=float,
    )


def extract_action(history: str, tutor_text: str) -> int:
    prompt = (
        "Label this tutor utterance. Reply with ONE word only:\n"
        "  instruct     — explains or corrects a concept\n"
        "  encourage    — motivates or praises the student\n"
        "  refocus      — brings distracted student back on topic\n"
        "  ask_question — asks the student a question\n\n"
        f"Dialogue:\n{history}\n\n"
        f"Tutor: \"{tutor_text}\"\n\n"
        "Reply with ONLY the label word."
    )
    raw = call_mistral(prompt, max_tokens=10).lower()
    for name, idx in ACTION_LABELS.items():
        if name in raw:
            return idx
    return 0


# ── Process one dialogue → RL tuples ──────────────────────────────────────────
def process_dialogue(dialogue: dict) -> list[dict]:
    turns = parse_augmented_turns(dialogue)
    if not turns:
        return []

    records         = []
    history_lines   = []
    tutor_q_count   = 0
    student_q_count = 0
    turn_number     = 0

    # Pair tutor → student
    paired = []
    i = 0
    while i < len(turns) - 1:
        if (turns[i]["role"] == "tutor" and
                turns[i+1]["role"] == "student"):
            paired.append((turns[i], turns[i+1]))
            i += 2
        else:
            i += 1

    if not paired:
        return []

    for pair_idx, (tutor_turn, student_turn) in enumerate(paired):
        turn_number  += 1
        tutor_text    = tutor_turn["text"]
        student_text  = student_turn["text"]
        reward        = student_turn["reward"]
        history_str   = "\n".join(history_lines) if history_lines else "(start)"

        # State S_n
        state_n  = extract_state_vector(
            history_str, student_text,
            turn_number, tutor_q_count, student_q_count,
        )

        # Action A_n
        action_n = extract_action(history_str, tutor_text)

        # Update counters from this turn's state
        if state_n[17] == 1:
            tutor_q_count   += 1
        if state_n[6] == 1:
            student_q_count += 1

        history_lines.append(f"Tutor: {tutor_text}")
        history_lines.append(f"Student: {student_text}")

        # State S_{n+1}
        if pair_idx + 1 < len(paired):
            ns_text    = paired[pair_idx + 1][1]["text"]
            ns_history = "\n".join(history_lines)
            state_n1   = extract_state_vector(
                ns_history, ns_text,
                turn_number + 1, tutor_q_count, student_q_count,
            )
        else:
            state_n1 = np.zeros(25)

        records.append({
            "dialogue_id": dialogue["dialogue_id"],
            "turn":        turn_number,
            "state":       state_n.tolist(),
            "action":      action_n,
            "action_name": ACTION_NAMES[action_n],
            "reward":      reward,
            "next_state":  state_n1.tolist(),
            "done":        (pair_idx + 1 == len(paired)),
        })

        time.sleep(1.2)   # Mistral 1 req/sec pacing

    return records


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def load_checkpoint() -> set:
    if not CHECKPOINT.exists():
        return set()
    with open(CHECKPOINT) as f:
        return set(json.load(f))


def save_checkpoint(done_ids: set):
    with open(CHECKPOINT, "w") as f:
        json.dump(list(done_ids), f)


# ── Records → flat CSV rows ────────────────────────────────────────────────────
def to_csv_row(rec: dict) -> dict:
    row = {
        "dialogue_id": rec["dialogue_id"],
        "turn":        rec["turn"],
        "action":      rec["action"],
        "action_name": rec["action_name"],
        "reward":      rec["reward"],
        "done":        rec["done"],
    }
    for i, v in enumerate(rec["state"]):
        row[f"s{i}"] = v
    for i, v in enumerate(rec["next_state"]):
        row[f"ns{i}"] = v
    return row


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Step 3 — DAY 4: Augmented Data State & Action Extraction")
    print("=" * 60)

    # Load augmented dialogues
    dialogues = []
    with open(INPUT_JSONL) as f:
        for line in f:
            try:
                dialogues.append(json.loads(line))
            except Exception:
                pass
    print(f"\nLoaded {len(dialogues)} augmented dialogues from {INPUT_JSONL}")

    # Load checkpoint
    done_ids   = load_checkpoint()
    remaining  = [d for d in dialogues
                  if d.get("dialogue_id") not in done_ids]
    print(f"Already done : {len(done_ids)}")
    print(f"Remaining    : {len(remaining)}\n")

    total_tuples = 0
    write_header = not OUTPUT_CSV.exists()

    with open(OUTPUT_CSV, "a", newline="") as csv_file:
        for d_idx, dialogue in enumerate(remaining):
            dial_id = dialogue.get("dialogue_id", f"aug_unknown_{d_idx}")
            print(f"[{d_idx+1}/{len(remaining)}] {dial_id}")

            try:
                records = process_dialogue(dialogue)

                if not records:
                    print(f"  No valid turn pairs found — skipping.")
                    done_ids.add(dial_id)
                    save_checkpoint(done_ids)
                    continue

                rows = [to_csv_row(r) for r in records]
                df   = pd.DataFrame(rows)
                df.to_csv(csv_file, index=False, header=write_header)
                csv_file.flush()

                write_header  = False
                total_tuples += len(records)
                done_ids.add(dial_id)
                save_checkpoint(done_ids)

                print(f"  → {len(records)} RL tuples saved. "
                      f"(total so far: {total_tuples})")

            except Exception as e:
                print(f"  Error: {e} — skipping {dial_id}")
                time.sleep(2)

    # Final stats
    print("\n" + "=" * 60)
    print("STEP 3 DAY 4 COMPLETE")
    print("=" * 60)
    print(f"  Augmented RL tuples : {total_tuples}")
    print(f"  Saved to            : {OUTPUT_CSV}")

    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
        actions = df["action"].tolist()
        print(f"\n  Action distribution:")
        for name, idx in ACTION_LABELS.items():
            count = actions.count(idx)
            print(f"    {name:15s}: {count:4d} "
                  f"({100*count/max(len(actions),1):.1f}%)")

    print("\nNext → python step4_day4_combined.py")


if __name__ == "__main__":
    main()