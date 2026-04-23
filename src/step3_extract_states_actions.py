"""
Step 3: State Representation & Action Extraction — MISTRAL VERSION (Free)
=========================================================================
Uses Mistral AI free API for extracting 25-dim student states
and classifying tutor actions.

Setup:
  pip install mistralai numpy pandas
  export MISTRAL_API_KEY="your-mistral-key-here"
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
DIALOGUE_PATH   = Path("data/dialogues/synthetic_dialogues.jsonl")
OUTPUT_JSONL    = Path("data/rl_dataset.jsonl")
OUTPUT_CSV      = Path("data/rl_dataset.csv")

client    = Mistral(api_key=MISTRAL_API_KEY)
N_ACTIONS = 4
ACTION_LABELS = {
    "instruct":     0,
    "encourage":    1,
    "refocus":      2,
    "ask_question": 3,
}
ACTION_NAMES = {v: k for k, v in ACTION_LABELS.items()}


# ── Mistral API helper ─────────────────────────────────────────────────────────
def call_mistral(prompt: str, max_tokens: int = 200) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.complete(
                model       = MODEL_NAME,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = max_tokens,
                temperature = 0.0,   # deterministic for classification
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


# ══════════════════════════════════════════════════════════════════════════════
# PART A — 25-DIM STATE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

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


def build_state_prompt(history: str, student_utterance: str) -> str:
    numbered = "\n".join(
        f"Q{i+1}: {q} Answer yes or no."
        for i, q in enumerate(STATE_QUESTIONS)
    )
    return (
        "You are evaluating a math tutoring dialogue.\n\n"
        f"Dialogue so far:\n{history}\n\n"
        f"Student's latest utterance: \"{student_utterance}\"\n\n"
        "Answer each question with ONLY 'yes' or 'no' — one per line, "
        "no extra text, no explanations.\n\n"
        f"{numbered}"
    )


def parse_yes_no(raw: str, n: int = 20) -> list[int]:
    answers = []
    for line in raw.strip().split("\n"):
        line = re.sub(r"^[qQ\d]+[\.\:\)]\s*", "", line.strip().lower())
        if "yes" in line:
            answers.append(1)
        elif "no" in line:
            answers.append(0)
        if len(answers) == n:
            break
    while len(answers) < n:
        answers.append(0)
    return answers[:n]


def math_density(text: str) -> float:
    tokens = re.findall(r"[\d\+\-\*\/\=\%\^\(\)]", text)
    words  = text.split()
    return round(len(tokens) / max(len(words), 1), 4)


def math_reasoning(text: str) -> float:
    keywords = {"because", "so", "therefore", "since", "if",
                "then", "means", "equals", "thus", "hence"}
    hits = len(set(text.lower().split()) & keywords)
    return round(min(hits / 5.0, 1.0), 4)


def extract_state_vector(
    history: str,
    student_utterance: str,
    turn_number: int,
    tutor_q_count: int,
    student_q_count: int,
) -> np.ndarray:
    raw    = call_mistral(build_state_prompt(history, student_utterance),
                          max_tokens=60)
    binary = parse_yes_no(raw, n=20)
    state  = np.array(
        binary + [
            float(tutor_q_count),
            float(student_q_count),
            float(turn_number),
            math_density(student_utterance),
            math_reasoning(student_utterance),
        ],
        dtype=float,
    )
    return state


# ══════════════════════════════════════════════════════════════════════════════
# PART B — ACTION EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

ACTION_EXAMPLES = {
    "instruct":     "Tutor: Let's think about this step by step. What formula would you use?",
    "encourage":    "Tutor: Great work! You're really close — keep going!",
    "refocus":      "Tutor: Let's get back to the problem. What were we finding?",
    "ask_question": "Tutor: Good thinking! Now what happens when you divide both sides by 2?",
}


def build_action_prompt(history: str, tutor_utterance: str) -> str:
    examples = "\n".join(
        f"Label: {name}\n{text}"
        for name, text in ACTION_EXAMPLES.items()
    )
    return (
        "Label the tutor's utterance in this math tutoring dialogue.\n\n"
        f"Dialogue:\n{history}\n\n"
        f"Tutor's utterance: \"{tutor_utterance}\"\n\n"
        "Choose exactly ONE label:\n"
        "  instruct      — explains or corrects a concept\n"
        "  encourage     — motivates or praises the student\n"
        "  refocus       — brings distracted student back to lesson\n"
        "  ask_question  — probes student understanding\n\n"
        f"Examples:\n{examples}\n\n"
        "Reply with ONLY the label word. Nothing else."
    )


def extract_action(history: str, tutor_utterance: str) -> int:
    raw = call_mistral(
        build_action_prompt(history, tutor_utterance),
        max_tokens=10,
    ).lower()
    for name, idx in ACTION_LABELS.items():
        if name in raw:
            return idx
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# PART C — PROCESS DIALOGUE INTO RL TUPLES
# ══════════════════════════════════════════════════════════════════════════════

def process_dialogue(dialogue: dict) -> list[dict]:
    turns           = dialogue["turns"]
    records         = []
    history_lines   = []
    tutor_q_count   = 0
    student_q_count = 0
    turn_number     = 0

    # Pair tutor → student turns
    paired = []
    i = 0
    while i < len(turns) - 1:
        if turns[i]["role"] == "tutor" and turns[i+1]["role"] == "student":
            paired.append((turns[i], turns[i+1]))
            i += 2
        else:
            i += 1

    for pair_idx, (tutor_turn, student_turn) in enumerate(paired):
        turn_number  += 1
        tutor_text    = tutor_turn["text"]
        student_text  = student_turn["text"]
        reward        = student_turn["reward"]
        history_str   = "\n".join(history_lines) if history_lines else "(start)"

        # State S_n
        state_n = extract_state_vector(
            history_str, student_text,
            turn_number, tutor_q_count, student_q_count,
        )

        # Action A_n
        action_n = extract_action(history_str, tutor_text)

        # Update counters
        if state_n[17] == 1:
            tutor_q_count += 1
        if state_n[6] == 1:
            student_q_count += 1

        history_lines.append(f"Tutor: {tutor_text}")
        history_lines.append(f"Student: {student_text}")

        # State S_{n+1}
        if pair_idx + 1 < len(paired):
            next_student = paired[pair_idx + 1][1]["text"]
            next_history = "\n".join(history_lines)
            state_n1 = extract_state_vector(
                next_history, next_student,
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

        time.sleep(1.2)   # safe pacing for 1 req/sec

    return records


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Step 3 (Mistral): State & Action Extraction")
    print("=" * 60)

    dialogues = []
    with open(DIALOGUE_PATH) as f:
        for line in f:
            try:
                dialogues.append(json.loads(line))
            except Exception:
                pass
    print(f"Loaded {len(dialogues)} dialogues.\n")

    all_records = []
    with open(OUTPUT_JSONL, "w") as out:
        for d_idx, dialogue in enumerate(dialogues):
            print(f"[{d_idx+1}/{len(dialogues)}] {dialogue['dialogue_id']} "
                  f"— {len(dialogue['turns'])//2} turn pairs")
            try:
                records = process_dialogue(dialogue)
                for rec in records:
                    out.write(json.dumps(rec) + "\n")
                all_records.extend(records)
                print(f"  → {len(records)} RL tuples extracted.")
            except Exception as e:
                print(f"  Error: {e} — skipping.")
            time.sleep(1)

    # Save CSV
    rows = []
    for rec in all_records:
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
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 60)
    print("STEP 3 COMPLETE")
    print("=" * 60)
    print(f"  Total RL tuples : {len(all_records)}")
    print(f"  Saved CSV       : {OUTPUT_CSV}")

    if all_records:
        actions = [r["action"] for r in all_records]
        print("\n  Action distribution:")
        for name, idx in ACTION_LABELS.items():
            count = actions.count(idx)
            print(f"    {name:15s}: {count:4d} "
                  f"({100*count/max(len(actions),1):.1f}%)")

    print("\nNext → step4_train_rl_policies.py")


if __name__ == "__main__":
    main()