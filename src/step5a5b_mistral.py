"""
Step 5a + 5b: Data Augmentation & Evaluation — MISTRAL VERSION (Free)
======================================================================
Uses Mistral AI free API for:
  5a — Optimism-guided data augmentation (generating D+)
  5b — Student simulator evaluation (reproducing Figure 3)

Setup:
  pip install mistralai scikit-learn numpy pandas matplotlib datasets d3rlpy torch
  export MISTRAL_API_KEY="your-mistral-key-here"

Run order:
  python step5a5b_mistral.py --mode augment    # runs Step 5a
  python step5a5b_mistral.py --mode evaluate   # runs Step 5b
  python step5a5b_mistral.py --mode both       # runs 5a then 5b
"""

import os
import json
import time
import pickle
import random
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from mistralai import Mistral
from datasets import load_dataset

# ── Configuration ──────────────────────────────────────────────────────────────
MISTRAL_API_KEY      = os.environ.get("MISTRAL_API_KEY", "B3lMQNsxQMCSe6BrmkbRy7ICigJavCwV")
MODEL_NAME           = "mistral-small-latest"
DATA_CSV             = Path("data/rl_dataset.csv")
DIALOGUE_JSONL       = Path("data/dialogues/synthetic_dialogues.jsonl")
MODEL_DIR            = Path("models")
AUG_OUTPUT           = Path("data/dialogues/augmented_dialogues.jsonl")
RESULTS_DIR          = Path("results")
AUG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TOP_N_CANDIDATES     = 500
SCENARIOS_PER        = 5
N_EVAL_CONVERSATIONS = 300
MAX_TURNS            = 20
DISCOUNT_FACTOR      = 0.9
N_ACTIONS            = 4

ACTION_NAMES = {0: "instruct", 1: "encourage", 2: "refocus", 3: "ask_question"}
ACTION_DESCRIPTIONS = {
    0: "teaching or instructing — explain or correct a concept",
    1: "encouraging — give positive, supportive remarks",
    2: "refocusing — bring the distracted student back to the lesson",
    3: "asking a question — probe the student's understanding",
}
COLORS = {"D": "#4a90d9", "D+": "#f5a623", "Prompt": "#5BAD72"}

client = Mistral(api_key=MISTRAL_API_KEY)


# ── Mistral API helper ─────────────────────────────────────────────────────────
def call_mistral(prompt: str, max_tokens: int = 1500,
                 temperature: float = 0.7) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.complete(
                model       = MODEL_NAME,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = max_tokens,
                temperature = temperature,
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
# STEP 5a — OPTIMISM-GUIDED DATA AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def load_rl_data():
    df          = pd.read_csv(DATA_CSV)
    state_cols  = [f"s{i}"  for i in range(25)]
    ns_cols     = [f"ns{i}" for i in range(25)]
    return (
        df[state_cols].values.astype(np.float32),
        df["action"].values.astype(np.int32),
        df["reward"].values.astype(np.float32),
        df[ns_cols].values.astype(np.float32),
        df["done"].values.astype(np.float32),
        df["dialogue_id"].values,
        df["turn"].values,
    )


def load_dialogues() -> dict:
    dialogues = {}
    with open(DIALOGUE_JSONL) as f:
        for line in f:
            try:
                d = json.loads(line)
                dialogues[d["dialogue_id"]] = d
            except Exception:
                pass
    return dialogues


def compute_optimism_scores(states, bc_model, fqi_model, action_encoder):
    def encode_sa(s_batch, a_batch):
        a_hot = action_encoder.transform(a_batch.reshape(-1, 1))
        return np.hstack([s_batch, a_hot])

    n            = len(states)
    scores       = np.zeros(n)
    best_actions = np.zeros(n, dtype=np.int32)

    for i in range(0, n, 256):
        s  = states[i:i+256]
        b  = len(s)
        Qs = np.stack([
            fqi_model.predict(encode_sa(s, np.full(b, a, dtype=np.int32)))
            for a in range(N_ACTIONS)
        ], axis=1)
        max_q  = Qs.max(axis=1)
        a_star = Qs.argmax(axis=1)
        bc_a   = bc_model.predict(s)
        bc_q   = np.array([Qs[j, bc_a[j]] for j in range(b)])
        scores[i:i+b]       = max_q - bc_q
        best_actions[i:i+b] = a_star

    return scores, best_actions


def generate_augmented_dialogue(original: dict,
                                turn_index: int,
                                optimal_action: int) -> dict | None:
    turns   = original["turns"]
    problem = original["gsm8k_question"]
    correct = original["correct_answer"]

    # Reconstruct history up to turn_index
    lines, pair_count = [], 0
    for t in turns:
        lines.append(f"{'Tutor' if t['role']=='tutor' else 'Student'}: {t['text']}")
        if t["role"] == "student":
            pair_count += 1
            if pair_count >= turn_index:
                break
    history = "\n".join(lines)

    # Intervene with optimal action
    desc = ACTION_DESCRIPTIONS[optimal_action]
    raw  = call_mistral(
        f"You are a math tutor. {desc.capitalize()}.\n\n"
        f"Dialogue so far:\n{history}\n\n"
        f"Begin your response with '[Generation] Tutor:' and {desc}. Be concise.",
        max_tokens=200,
    )
    if not raw:
        return None
    intervention = re.sub(r"\[Generation\]\s*Tutor:\s*", "", raw).strip()
    intervention = re.sub(r"^Tutor:\s*", "", intervention).strip()

    # Continue the rest of the dialogue
    partial      = history + f"\nTutor: {intervention}\n"
    continuation = call_mistral(
        f"Continue this tutoring dialogue. Problem: {problem}\n\n"
        f"Do NOT give the answer directly. End when the student solves it.\n\n"
        f"Dialogue:\n{partial}\n\nBegin with 'Student:' and continue until solved.",
        max_tokens=1200,
    )
    if not continuation:
        return None

    numbers = re.findall(r"-?\d+\.?\d*", continuation)
    clean   = re.sub(r"[^0-9.]", "", correct.split("####")[-1].strip())
    solved  = clean in numbers if clean else False

    return {
        "dialogue_id":       f"aug_{original['dialogue_id']}_t{turn_index}_a{optimal_action}",
        "source_dialogue":   original["dialogue_id"],
        "intervention_turn": turn_index,
        "optimal_action":    optimal_action,
        "action_name":       ACTION_NAMES[optimal_action],
        "gsm8k_question":    problem,
        "correct_answer":    correct,
        "continuation":      continuation,
        "solved":            solved,
        "final_reward":      1 if solved else -1,
    }


def run_augmentation():
    print("=" * 60)
    print("Step 5a (Mistral): Optimism-guided Data Augmentation")
    print("=" * 60)

    (states, actions, rewards,
     next_states, terminals,
     dial_ids, turn_nums) = load_rl_data()
    dialogues              = load_dialogues()

    with open(MODEL_DIR / "bc_policy.pkl",  "rb") as f:
        bc_model = pickle.load(f)
    with open(MODEL_DIR / "fqi_policy.pkl", "rb") as f:
        fqi_model, action_encoder = pickle.load(f)

    print(f"  {len(states)} transitions | {len(dialogues)} dialogues\n")
    print("  Computing optimism scores...")
    scores, best_actions = compute_optimism_scores(
        states, bc_model, fqi_model, action_encoder
    )
    top_idx = np.argsort(scores)[::-1][:TOP_N_CANDIDATES]
    print(f"  Top-{TOP_N_CANDIDATES} selected. "
          f"Score range: {scores[top_idx].min():.4f} – {scores[top_idx].max():.4f}\n")

    aug_count = success_count = 0
    with open(AUG_OUTPUT, "w") as out:
        for rank, idx in enumerate(top_idx):
            dial_id  = dial_ids[idx]
            turn_num = int(turn_nums[idx])
            a_star   = int(best_actions[idx])
            if dial_id not in dialogues:
                continue
            for _ in range(SCENARIOS_PER):
                aug = generate_augmented_dialogue(dialogues[dial_id], turn_num, a_star)
                if aug:
                    out.write(json.dumps(aug) + "\n")
                    aug_count += 1
                    if aug["solved"]:
                        success_count += 1
                time.sleep(1.2)

            if (rank + 1) % 50 == 0:
                print(f"  [{rank+1}/{TOP_N_CANDIDATES}] "
                      f"{aug_count} dialogues | "
                      f"success: {100*success_count/max(aug_count,1):.1f}%")

    print(f"\n  Done: {aug_count} augmented dialogues → {AUG_OUTPUT}")
    print(f"  Success rate: {100*success_count/max(aug_count,1):.1f}%"
          f" (paper target: ~82.83%)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5b — STUDENT SIMULATOR EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

STUDENT_SYSTEM = (
    "You are a sixth-grade student who is not good at math. "
    "You struggle with problems and make mistakes. "
    "You are easily distracted. Keep responses short."
)


def sample_mistake(problem: str) -> str:
    raw = call_mistral(
        f"Problem: {problem}\n"
        "Name one specific mistake a sixth-grader might make. One sentence.",
        max_tokens=60,
    )
    return raw or "The student makes a calculation error."


def student_respond(history: list[dict], mistake: str, problem: str) -> str:
    context = "\n".join(
        f"{'Tutor' if m['role']=='assistant' else 'Student'}: {m['content']}"
        for m in history[-6:]
    )
    return call_mistral(
        f"{STUDENT_SYSTEM}\nProblem: {problem}\nYour mistake: {mistake}\n\n"
        f"Conversation:\n{context}\n\nStudent:",
        max_tokens=150, temperature=0.8,
    )


def extract_answer(student_text: str, correct_answer: str) -> bool:
    clean    = re.sub(r"[^0-9.]", "",
                      correct_answer.split("####")[-1].strip())
    numbers  = re.findall(r"-?\d+\.?\d*", student_text)
    confused = {"not sure", "confused", "don't know", "maybe"}
    return (clean in numbers and
            not any(w in student_text.lower() for w in confused))


def prompt_tutor(history: list[dict], problem: str) -> str:
    context = "\n".join(
        f"{'Tutor' if m['role']=='assistant' else 'Student'}: {m['content']}"
        for m in history[-6:]
    )
    return call_mistral(
        "You are a math tutor. Guide the student — do NOT give the answer. "
        "Ask questions and correct mistakes. Keep it concise.\n\n"
        f"{context}\n\nTutor:",
        max_tokens=200,
    )


def extract_state_fast(text: str, turn: int,
                       tq: int, sq: int) -> np.ndarray:
    words    = set(text.lower().split())
    math_w   = {"equation","solve","calculate","formula","equals","answer","number"}
    frust    = {"frustrated","confused","hard","can't","stuck","lost"}
    positive = {"yes","got it","understand","okay","right","thanks"}
    nums     = re.findall(r"-?\d+\.?\d*", text)
    return np.array([
        float(bool(words & math_w)),
        float(len(nums) > 0),
        float("again" in text.lower() or "repeat" in text.lower()),
        0., float(any(w in text.lower() for w in ["anyway","lunch","game"])),
        0., float("?" in text),
        float(bool(words & {"stuck","confused","which","where"})),
        0., float(bool(words & frust)),
        float("not sure" in text.lower() or "maybe" in text.lower()),
        float(bool(words & positive)),
        float(any(w in text.lower() for w in ["break","tired","later"])),
        float(bool(words & math_w)), 0., 0.,
        float(bool(re.search(r"[a-zA-Z]\s*=", text))),
        0., float(len(nums) > 0 and "=" in text), 0.,
        float(tq), float(sq), float(turn),
        float(len(words & math_w)) / max(len(words), 1),
        float(bool(words & {"because","so","therefore","since"})),
    ], dtype=np.float32)


def rl_tutor(history, problem, policy, policy_type,
             student_text, turn, tq, sq) -> tuple[str, int]:
    state  = extract_state_fast(student_text, turn, tq, sq)
    action = (int(policy.predict(state.reshape(1, -1))[0])
              if policy_type == "cql"
              else int(policy.predict(state)))
    context = "\n".join(
        f"{'Tutor' if m['role']=='assistant' else 'Student'}: {m['content']}"
        for m in history[-6:]
    )
    response = call_mistral(
        f"You are a math tutor. Goal: {ACTION_DESCRIPTIONS[action]}.\n"
        f"Do NOT give the answer. Be concise.\n\n{context}\n\nTutor:",
        max_tokens=200,
    )
    return response, action


def run_conversation(problem, tutor_fn, policy=None,
                     policy_type="prompt") -> dict:
    question = problem["question"]
    correct  = problem["answer"]
    mistake  = sample_mistake(question)
    history  = []
    tq = sq  = 0
    solved   = False
    act_log  = []

    history.append({"role": "assistant",
                    "content": "Hi! What math problem do you need help with?"})
    history.append({"role": "user",
                    "content": f"I need help with: {question}"})

    for turn in range(1, MAX_TURNS + 1):
        last_student = history[-1]["content"]
        if policy_type == "prompt":
            tutor_text  = tutor_fn(history, question)
            act         = None
        else:
            tutor_text, act = tutor_fn(
                history, question, policy, policy_type,
                last_student, turn, tq, sq,
            )
            act_log.append(act)

        if "?" in tutor_text:
            tq += 1
        history.append({"role": "assistant", "content": tutor_text})

        student_text = student_respond(history, mistake, question)
        if "?" in student_text:
            sq += 1
        history.append({"role": "user", "content": student_text})

        if extract_answer(student_text, correct):
            solved = True
            break
        time.sleep(1.2)

    return {
        "solved":     solved,
        "turns":      turn,
        "disc_reward":(DISCOUNT_FACTOR ** turn) if solved else -1.0,
        "action_log": act_log,
    }


def evaluate_policy(name, tutor_fn, problems, n=N_EVAL_CONVERSATIONS,
                    policy=None, policy_type="prompt") -> dict:
    print(f"\n  Evaluating {name} ({n} conversations)...")
    results = []
    sampled = random.choices(problems, k=n)
    for i, prob in enumerate(sampled):
        results.append(run_conversation(prob, tutor_fn, policy, policy_type))
        if (i + 1) % 50 == 0:
            sr = 100 * sum(r["solved"] for r in results) / len(results)
            print(f"    [{i+1}/{n}] running success: {sr:.1f}%")

    sr      = 100 * sum(r["solved"] for r in results) / len(results)
    avg_t   = np.mean([r["turns"] for r in results])
    all_a   = [a for r in results for a in r["action_log"]]
    act_dist = {ACTION_NAMES[a]: round(100*all_a.count(a)/max(len(all_a),1),1)
                for a in range(N_ACTIONS)} if all_a else {}
    print(f"    {name}: {sr:.2f}% success | avg turns={avg_t:.1f}")
    return {"policy": name, "success_rate": sr, "avg_turns": avg_t,
            "action_dist": act_dist, "n_eval": n}


def plot_results(results):
    names  = [r["policy"] for r in results]
    scores = [r["success_rate"] for r in results]
    colors = [COLORS["D+"] if "D+" in n else
              COLORS["Prompt"] if n == "Prompt" else COLORS["D"]
              for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars    = ax.bar(names, scores, color=colors, width=0.55, zorder=3)
    ax.axhline(36.0,  color=COLORS["Prompt"], linestyle="--", linewidth=1.2)
    ax.axhline(60.33, color=COLORS["D+"],     linestyle="--", linewidth=1.2)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, s+0.8,
                f"{s:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    patches = [
        mpatches.Patch(color=COLORS["D"],      label="Trained on D"),
        mpatches.Patch(color=COLORS["D+"],     label="Trained on D+"),
        mpatches.Patch(color=COLORS["Prompt"], label="Prompt engineering"),
    ]
    ax.legend(handles=patches, fontsize=10)
    ax.set_ylabel("Student success rate (%)", fontsize=12)
    ax.set_title("Figure 3 reproduction — Mistral version", fontsize=13)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path = RESULTS_DIR / "figure3_mistral.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved → {save_path}")
    plt.close()


def run_evaluation():
    print("=" * 60)
    print("Step 5b (Mistral): Student Simulator Evaluation")
    print("=" * 60)

    gsm8k    = load_dataset("gsm8k", "main")
    problems = list(gsm8k["test"])
    random.seed(42)

    with open(MODEL_DIR / "bc_policy.pkl",  "rb") as f:
        bc_model = pickle.load(f)
    with open(MODEL_DIR / "fqi_policy.pkl", "rb") as f:
        fqi_model, _ = pickle.load(f)

    cql_D = cql_Dp = None
    try:
        import d3rlpy
        cql_D = d3rlpy.algos.DiscreteCQL.from_json(
            str(MODEL_DIR / "cql_D_policy/params.json"))
        cql_D.load_model(str(MODEL_DIR / "cql_D_policy/model.pt"))
    except Exception as e:
        print(f"  CQL(D) not loaded: {e}")
    try:
        import d3rlpy
        cql_Dp = d3rlpy.algos.DiscreteCQL.from_json(
            str(MODEL_DIR / "cql_Dp_policy/params.json"))
        cql_Dp.load_model(str(MODEL_DIR / "cql_Dp_policy/model.pt"))
    except Exception as e:
        print(f"  CQL(D+) not loaded: {e}")

    all_results = []
    all_results.append(evaluate_policy("Prompt",  prompt_tutor, problems,
                                       policy_type="prompt"))
    all_results.append(evaluate_policy("BC(D)",   rl_tutor, problems,
                                       policy=bc_model,  policy_type="bc"))
    all_results.append(evaluate_policy("FQI(D)",  rl_tutor, problems,
                                       policy=fqi_model, policy_type="fqi"))
    if cql_D:
        all_results.append(evaluate_policy("CQL(D)",  rl_tutor, problems,
                                           policy=cql_D,  policy_type="cql"))
    if cql_Dp:
        all_results.append(evaluate_policy("CQL(D+)", rl_tutor, problems,
                                           policy=cql_Dp, policy_type="cql"))

    with open(RESULTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    plot_results(all_results)

    paper_ref = {"Prompt":36.0,"BC(D)":"<36","FQI(D)":"~44",
                 "CQL(D)":48.67,"CQL(D+)":60.33}
    print("\n" + "=" * 60)
    print(f"  {'Policy':<12} {'Ours':>10} {'Paper':>10}")
    print(f"  {'-'*34}")
    for r in all_results:
        ref = paper_ref.get(r["policy"], "N/A")
        print(f"  {r['policy']:<12} {r['success_rate']:>9.2f}%  {str(ref):>9}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["augment","evaluate","both"],
                        default="both")
    args = parser.parse_args()

    if args.mode in ("augment", "both"):
        run_augmentation()
    if args.mode in ("evaluate", "both"):
        run_evaluation()
