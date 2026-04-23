"""
Step 5b (FINAL FIX): Student Simulator Evaluation
===================================================
Fix applied based on diagnostic output:

  - cql_D_policy  → single file (1158 KB), no params.json inside
  - cql_Dp_policy → single file (1158 KB), no params.json inside
  - params.json   → lives in d3rlpy_logs/DiscreteCQL_XXXXXXXXX/

So the correct loading pattern is:
  1. Find params.json from d3rlpy_logs/
  2. Load architecture from params.json
  3. Load weights from models/cql_D_policy (single file)

Run:
  python step5b_final.py
"""

import os
import json
import time
import pickle
import random
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from mistralai import Mistral
from datasets import load_dataset

# ── Paths ──────────────────────────────────────────────────────────────────────
MISTRAL_API_KEY  = os.environ.get("MISTRAL_API_KEY", "B3lMQNsxQMCSe6BrmkbRy7ICigJavCwV")
MODEL_NAME       = "mistral-small-latest"
MODEL_DIR        = Path("models")
RESULTS_DIR      = Path("results")
D3RLPY_LOGS_DIR  = Path("d3rlpy_logs")          # where params.json lives
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH  = RESULTS_DIR / "eval_checkpoint.json"

N_EVAL_CONVERSATIONS = 50
MAX_TURNS            = 10
DISCOUNT_FACTOR      = 0.9
N_ACTIONS            = 4

ACTION_NAMES = {0: "instruct", 1: "encourage",
                2: "refocus",  3: "ask_question"}
ACTION_DESCRIPTIONS = {
    0: "teaching or instructing — explain or correct a concept",
    1: "encouraging — give positive, supportive remarks",
    2: "refocusing — bring the distracted student back to the lesson",
    3: "asking a question — probe the student's understanding",
}
COLORS = {"D": "#4a90d9", "D+": "#f5a623", "Prompt": "#5BAD72"}

client = Mistral(api_key=MISTRAL_API_KEY)


# ══════════════════════════════════════════════════════════════════════════════
# CQL LOADER — uses d3rlpy_logs for params.json + models/ for weights
# ══════════════════════════════════════════════════════════════════════════════

def find_params_json(log_index: int = 0) -> Path | None:
    """
    Find params.json files inside d3rlpy_logs/.
    log_index=0 → oldest log (cql_D),  log_index=1 → newest log (cql_Dp)
    Returns the path to params.json or None if not found.
    """
    if not D3RLPY_LOGS_DIR.exists():
        print(f"  d3rlpy_logs/ folder not found at: {D3RLPY_LOGS_DIR}")
        return None

    # Find all DiscreteCQL log folders, sorted by name (= by timestamp)
    log_folders = sorted([
        f for f in D3RLPY_LOGS_DIR.iterdir()
        if f.is_dir() and "DiscreteCQL" in f.name
    ])

    print(f"  Found {len(log_folders)} DiscreteCQL log folders:")
    for i, folder in enumerate(log_folders):
        params = folder / "params.json"
        print(f"    [{i}] {folder.name}  "
              f"{'(params.json exists)' if params.exists() else '(no params.json)'}")

    if not log_folders:
        return None

    # Clamp index to valid range
    idx = min(log_index, len(log_folders) - 1)
    chosen = log_folders[idx] / "params.json"

    if chosen.exists():
        print(f"  Using params.json from: {log_folders[idx].name}")
        return chosen

    return None


def load_cql_model(weights_path: Path, log_index: int = 0):
    """
    Load a CQL model using:
      weights_path : models/cql_D_policy  (single file with weights)
      log_index    : 0 = first/older log (cql_D), 1 = second/newer log (cql_Dp)
    """
    try:
        import d3rlpy
    except ImportError:
        print("  d3rlpy not installed.")
        return None

    if not weights_path.exists():
        print(f"  Weights file not found: {weights_path}")
        return None

    # Find matching params.json from d3rlpy_logs/
    params_json = find_params_json(log_index)

    if params_json is None:
        print(f"  No params.json found in {D3RLPY_LOGS_DIR}/")
        print("  Trying direct from_json on weights file as fallback...")
        try:
            algo = d3rlpy.algos.DiscreteCQL.from_json(str(weights_path))
            print(f"  Loaded (fallback) from: {weights_path}")
            return algo
        except Exception as e:
            print(f"  Fallback failed: {e}")
            return None

    # Load architecture from params.json, then load weights
    try:
        algo = d3rlpy.algos.DiscreteCQL.from_json(str(params_json))
        algo.load_model(str(weights_path))
        print(f"  CQL loaded successfully.")
        print(f"    params  : {params_json}")
        print(f"    weights : {weights_path}")
        return algo
    except Exception as e:
        print(f"  Loading failed: {e}")

        # Last resort: try from_json directly on the weights file
        try:
            algo = d3rlpy.algos.DiscreteCQL.from_json(str(weights_path))
            print(f"  Loaded via direct from_json: {weights_path}")
            return algo
        except Exception as e2:
            print(f"  All loading attempts failed: {e2}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
# MISTRAL HELPER
# ══════════════════════════════════════════════════════════════════════════════

def call_mistral(prompt: str, max_tokens: int = 200,
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
# STUDENT SIMULATOR
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
    clean   = re.sub(r"[^0-9.]", "",
                     correct_answer.split("####")[-1].strip())
    numbers = re.findall(r"-?\d+\.?\d*", student_text)
    confused = {"not sure", "confused", "don't know", "maybe"}
    return (bool(clean) and
            clean in numbers and
            not any(w in student_text.lower() for w in confused))


# ══════════════════════════════════════════════════════════════════════════════
# TUTOR POLICIES
# ══════════════════════════════════════════════════════════════════════════════

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
    words  = set(text.lower().split())
    math_w = {"equation","solve","calculate","formula","equals","answer","number"}
    frust  = {"frustrated","confused","hard","can't","stuck","lost"}
    pos    = {"yes","got it","understand","okay","right","thanks"}
    nums   = re.findall(r"-?\d+\.?\d*", text)
    return np.array([
        float(bool(words & math_w)),
        float(len(nums) > 0),
        float("again" in text.lower() or "repeat" in text.lower()),
        0., float(any(w in text.lower() for w in ["anyway","lunch","game"])),
        0., float("?" in text),
        float(bool(words & {"stuck","confused","which","where"})),
        0., float(bool(words & frust)),
        float("not sure" in text.lower() or "maybe" in text.lower()),
        float(bool(words & pos)),
        float(any(w in text.lower() for w in ["break","tired","later"])),
        float(bool(words & math_w)), 0., 0.,
        float(bool(re.search(r"[a-zA-Z]\s*=", text))),
        0., float(len(nums) > 0 and "=" in text), 0.,
        float(tq), float(sq), float(turn),
        float(len(words & math_w)) / max(len(words), 1),
        float(bool(words & {"because","so","therefore","since"})),
    ], dtype=np.float32)


def rl_tutor(history, problem, policy, policy_type,
             student_text, turn, tq, sq):
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


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION + EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

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
            tutor_text = tutor_fn(history, question)
            act        = None
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

        time.sleep(1.5)

    return {
        "solved":      solved,
        "turns":       turn,
        "disc_reward": (DISCOUNT_FACTOR ** turn) if solved else -1.0,
        "action_log":  act_log,
    }


def load_checkpoint() -> list[dict]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return []


def save_checkpoint(results: list[dict]):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=2)


def evaluate_policy(name, tutor_fn, problems,
                    n=N_EVAL_CONVERSATIONS,
                    policy=None,
                    policy_type="prompt") -> dict:
    print(f"\n  Evaluating {name} ({n} conversations)...")
    results = []
    sampled = random.choices(problems, k=n)

    for i, prob in enumerate(sampled):
        res = run_conversation(prob, tutor_fn, policy, policy_type)
        results.append(res)
        if (i + 1) % 50 == 0:
            sr = 100 * sum(r["solved"] for r in results) / len(results)
            print(f"    [{i+1}/{n}] running success: {sr:.1f}%")

    sr    = 100 * sum(r["solved"] for r in results) / len(results)
    avg_t = float(np.mean([r["turns"] for r in results]))
    all_a = [a for r in results for a in r["action_log"]]
    act_dist = {
        ACTION_NAMES[a]: round(100 * all_a.count(a) / max(len(all_a), 1), 1)
        for a in range(N_ACTIONS)
    } if all_a else {}

    print(f"    {name}: {sr:.2f}% success | avg turns={avg_t:.1f}")
    return {
        "policy":       name,
        "success_rate": sr,
        "avg_turns":    avg_t,
        "action_dist":  act_dist,
        "n_eval":       n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(results: list[dict]):
    names  = [r["policy"] for r in results]
    scores = [r["success_rate"] for r in results]
    colors = [
        COLORS["D+"] if "D+" in n else
        COLORS["Prompt"] if n == "Prompt" else
        COLORS["D"]
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars    = ax.bar(names, scores, color=colors, width=0.55, zorder=3)
    ax.axhline(36.0,  color=COLORS["Prompt"], linestyle="--",
               linewidth=1.2, label="Prompt baseline (36%)")
    ax.axhline(60.33, color=COLORS["D+"],     linestyle="--",
               linewidth=1.2, label="CQL(D+) paper target (60.33%)")

    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                s + 0.8, f"{s:.1f}%",
                ha="center", va="bottom",
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Step 5b (Final Fix): Student Simulator Evaluation")
    print("=" * 60)

    # Load GSM8K
    gsm8k    = load_dataset("gsm8k", "main")
    problems = list(gsm8k["test"])
    random.seed(42)

    # ── Load BC and FQI ────────────────────────────────────────────────────────
    print("\nLoading BC and FQI policies...")
    with open(MODEL_DIR / "bc_policy.pkl",  "rb") as f:
        bc_model = pickle.load(f)
    with open(MODEL_DIR / "fqi_policy.pkl", "rb") as f:
        fqi_model, _ = pickle.load(f)
    print("  bc_policy.pkl   loaded.")
    print("  fqi_policy.pkl  loaded.")

    # Load D+ variants if available
    bc_Dp = fqi_Dp = None
    if (MODEL_DIR / "bc_Dp_policy.pkl").exists():
        with open(MODEL_DIR / "bc_Dp_policy.pkl", "rb") as f:
            bc_Dp = pickle.load(f)
        print("  bc_Dp_policy.pkl  loaded.")
    if (MODEL_DIR / "fqi_Dp_policy.pkl").exists():
        with open(MODEL_DIR / "fqi_Dp_policy.pkl", "rb") as f:
            fqi_Dp, _ = pickle.load(f)
        print("  fqi_Dp_policy.pkl loaded.")

    # ── Load CQL models using correct method ───────────────────────────────────
    # Your diagnostic showed:
    #   models/cql_D_policy   → single weights file
    #   models/cql_Dp_policy  → single weights file
    #   params.json #0 (older) → d3rlpy_logs/DiscreteCQL_20260407075941/
    #   params.json #1 (newer) → d3rlpy_logs/DiscreteCQL_20260415111028/
    print("\nLoading CQL policies...")
    cql_D  = load_cql_model(MODEL_DIR / "cql_D_policy",  log_index=0)
    cql_Dp = load_cql_model(MODEL_DIR / "cql_Dp_policy", log_index=1)

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    all_results   = load_checkpoint()
    done_policies = {r["policy"] for r in all_results}
    if done_policies:
        print(f"\n  Resuming — already completed: {done_policies}")

    # ── Build evaluation queue ─────────────────────────────────────────────────
    eval_queue = [
        ("Prompt",   prompt_tutor, None,      "prompt"),
        ("BC(D)",    rl_tutor,     bc_model,  "bc"),
        ("FQI(D)",   rl_tutor,     fqi_model, "fqi"),
    ]
    if cql_D:
        eval_queue.append(("CQL(D)",  rl_tutor, cql_D,  "cql"))
    if bc_Dp:
        eval_queue.append(("BC(D+)",  rl_tutor, bc_Dp,  "bc"))
    if fqi_Dp:
        eval_queue.append(("FQI(D+)", rl_tutor, fqi_Dp, "fqi"))
    if cql_Dp:
        eval_queue.append(("CQL(D+)", rl_tutor, cql_Dp, "cql"))

    print(f"\n  Policies to evaluate: "
          f"{[p[0] for p in eval_queue if p[0] not in done_policies]}")

    # ── Run evaluations ────────────────────────────────────────────────────────
    for policy_name, tutor_fn, policy_obj, policy_type in eval_queue:
        if policy_name in done_policies:
            print(f"\n  Skipping {policy_name} — already done.")
            continue

        result = evaluate_policy(
            policy_name, tutor_fn, problems,
            policy=policy_obj, policy_type=policy_type,
        )
        all_results.append(result)
        save_checkpoint(all_results)
        print(f"  Checkpoint saved ({len(all_results)}/{len(eval_queue)} done).")

    # ── Save results + plot ────────────────────────────────────────────────────
    final_path = RESULTS_DIR / "evaluation_results.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {final_path}")

    plot_results(all_results)

    # ── Final summary table ────────────────────────────────────────────────────
    paper_ref = {
        "Prompt":  36.00, "BC(D)":  "<36",  "FQI(D)":  "~44",
        "CQL(D)":  48.67, "BC(D+)": ">36",  "FQI(D+)": "~46",
        "CQL(D+)": 60.33,
    }
    print("\n" + "=" * 60)
    print("FINAL RESULTS vs PAPER")
    print("=" * 60)
    print(f"  {'Policy':<12} {'Our result':>12} {'Paper':>10}")
    print(f"  {'-'*38}")
    for r in all_results:
        ref = paper_ref.get(r["policy"], "N/A")
        print(f"  {r['policy']:<12} {r['success_rate']:>10.2f}%   "
              f"{str(ref):>8}")

    print(f"\n  Next → python step6_analyze_results.py")


if __name__ == "__main__":
    main()