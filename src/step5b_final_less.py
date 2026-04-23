"""
Step 5b (FINAL STABLE VERSION): Student Simulator Evaluation
============================================================
Fixes applied:
1. Robust retry (503, 429) with exponential backoff
2. Fallback responses (no empty strings)
3. Reduced evaluation load
4. Global throttling
5. Response caching (reduces API calls significantly)
"""

import os
import json
import time
import pickle
import random
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mistralai import Mistral
from datasets import load_dataset

# ── Config ─────────────────────────────────────────────────────
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY not set.\n"
        "Run this in terminal before execution:\n"
        "export MISTRAL_API_KEY='your_api_key_here'"
    )
MODEL_NAME = "mistral-small-latest"

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
D3RLPY_LOGS_DIR = Path("d3rlpy_logs")

RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH = RESULTS_DIR / "api_cache.json"

# 🔥 Reduced load (Fix #3)
N_EVAL_CONVERSATIONS = 50
MAX_TURNS = 10
DISCOUNT_FACTOR = 0.9

client = Mistral(api_key=MISTRAL_API_KEY)

# ── Simple Cache (Fix #5) ──────────────────────────────────────
if CACHE_PATH.exists():
    with open(CACHE_PATH) as f:
        CACHE = json.load(f)
else:
    CACHE = {}

def save_cache():
    with open(CACHE_PATH, "w") as f:
        json.dump(CACHE, f)

# ── Robust Mistral Call (Fix #1, #2) ───────────────────────────
def call_mistral(prompt, max_tokens=200, temperature=0.7):
    if prompt in CACHE:
        return CACHE[prompt]

    for attempt in range(6):
        try:
            resp = client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = resp.choices[0].message.content.strip()

            if not text:
                text = "I'm not sure, can you explain again?"

            CACHE[prompt] = text
            return text

        except Exception as e:
            err = str(e).lower()

            if "503" in err or "unreachable_backend" in err:
                wait = 2 ** attempt
                print(f"503 error → retrying in {wait}s...")
                time.sleep(wait)

            elif "429" in err:
                wait = 20 * (attempt + 1)
                print(f"Rate limit → waiting {wait}s...")
                time.sleep(wait)

            else:
                print(f"Error: {e}")
                time.sleep(3)

    return "I'm confused, can you explain step by step?"

# ── Student Simulator ──────────────────────────────────────────
STUDENT_SYSTEM = "You are a weak 6th-grade math student. Keep answers short."

def sample_mistake(problem):
    return call_mistral(f"Problem: {problem}\nOne mistake student makes:")

def student_respond(history, mistake, problem):
    context = "\n".join([m["content"] for m in history[-6:]])
    return call_mistral(f"{STUDENT_SYSTEM}\n{context}")

# ── Tutor Policies ─────────────────────────────────────────────
def prompt_tutor(history, problem):
    context = "\n".join([m["content"] for m in history[-6:]])
    return call_mistral(f"Tutor helping solve:\n{context}")

# ── RL State ───────────────────────────────────────────────────
def extract_state(text):
    words = text.lower().split()
    return np.array([
        float("?" in text),
        float(len(words) > 5),
        float(any(w in text for w in ["confused", "stuck"])),
    ], dtype=np.float32)

def rl_tutor(history, problem, policy, policy_type, student_text):
    state = extract_state(student_text)
    state_2d = state.reshape(1, -1)

    if policy_type == "cql":
        action = int(policy.predict(state_2d)[0])
    else:
        action = int(policy.predict(state_2d)[0])

    return call_mistral(f"Tutor action {action}:\n{student_text}")

# ── Conversation Runner ────────────────────────────────────────
def run_conversation(problem, tutor_fn, policy=None, policy_type="prompt"):
    history = []
    solved = False

    history.append({"role": "assistant", "content": "Hi, let's solve this."})
    history.append({"role": "user", "content": problem["question"]})

    for turn in range(MAX_TURNS):
        last = history[-1]["content"]

        if policy_type == "prompt":
            tutor_text = tutor_fn(history, problem)
        else:
            tutor_text = rl_tutor(history, problem, policy, policy_type, last)

        history.append({"role": "assistant", "content": tutor_text})

        student_text = student_respond(history, "", problem)
        history.append({"role": "user", "content": student_text})

        if "####" in student_text:
            solved = True
            break

        # 🔥 Throttle (Fix #4)
        time.sleep(1.5)

    return {"solved": solved, "turns": turn}

# ── Evaluation ─────────────────────────────────────────────────
def evaluate_policy(name, tutor_fn, problems, policy=None, policy_type="prompt"):
    print(f"\nEvaluating {name}...")
    results = []

    for i in range(N_EVAL_CONVERSATIONS):
        res = run_conversation(random.choice(problems), tutor_fn, policy, policy_type)
        results.append(res)

        if (i + 1) % 10 == 0:
            print(f"{i+1}/{N_EVAL_CONVERSATIONS} done")

    success = sum(r["solved"] for r in results) / len(results)
    print(f"{name} success: {success*100:.2f}%")

    return {"policy": name, "success_rate": success * 100}

# ── Main ──────────────────────────────────────────────────────
def main():
    print("Starting evaluation...")

    gsm8k = load_dataset("gsm8k", "main")
    problems = list(gsm8k["test"])

    # Load models
    with open(MODEL_DIR / "bc_policy.pkl", "rb") as f:
        bc = pickle.load(f)

    with open(MODEL_DIR / "fqi_policy.pkl", "rb") as f:
        fqi, _ = pickle.load(f)

    results = []

    results.append(evaluate_policy("Prompt", prompt_tutor, problems))
    results.append(evaluate_policy("BC", rl_tutor, problems, bc, "bc"))
    results.append(evaluate_policy("FQI", rl_tutor, problems, fqi, "fqi"))

    # Save
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    save_cache()

    print("\nFinal Results:")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()