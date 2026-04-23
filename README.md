# Optimizing Multi-turn Student Learning using Offline Reinforcement Learning with LLM Tutors

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Mistral AI](https://img.shields.io/badge/LLM-Mistral%20AI-orange?style=flat-square)
![d3rlpy](https://img.shields.io/badge/RL-d3rlpy%202.8.1-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

**B.Tech Artificial Intelligence — Final Year Project**
**Pavan Ramavath | SVNIT Surat**

*A replication and extension of "Efficient Reinforcement Learning for Optimizing Multi-turn Student Outcomes with LLM Tutors" (NeurIPS 2025 Workshop)*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Architecture](#project-architecture)
- [Datasets](#datasets)
- [Installation](#installation)
- [API Key Setup](#api-key-setup-free-mistral-ai)
- [Project Pipeline](#project-pipeline)
  - [Step 1 — Dataset Download](#step-1--dataset-download)
  - [Step 2 — Synthetic Dialogue Generation](#step-2--synthetic-dialogue-generation)
  - [Step 3 — State & Action Extraction](#step-3--state--action-extraction)
  - [Step 4 — RL Policy Training](#step-4--rl-policy-training)
  - [Step 5a — Data Augmentation](#step-5a--optimism-guided-data-augmentation)
  - [Step 5b — Evaluation](#step-5b--student-simulator-evaluation)
  - [Step 6 — Analysis & Figures](#step-6--analysis--figures)
- [Day-by-Day Execution Plan](#day-by-day-execution-plan)
- [File Structure](#file-structure)
- [State Representation](#state-representation-25-dimensions)
- [Action Space](#action-space)
- [RL Policies](#rl-policies)
- [Extended Policies](#extended-policies-beyond-the-paper)
- [Reward Function](#reward-function--discount-factor)
- [Results](#results)
- [Challenges & Fixes](#challenges--fixes)
- [Differences from Original Paper](#differences-from-original-paper)
- [Citation](#citation)

---

## Overview

Large language models (LLMs) optimized with standard RLHF generate the **best single-turn response** — but in tutoring, this means giving away answers instead of guiding students step-by-step.

This project replicates and extends the paper's approach: instead of optimizing each response in isolation, we train an **offline RL policy** that selects high-level tutor actions (instruct, encourage, refocus, ask a question) to maximize **long-term student learning** across a full multi-turn conversation.

### Core Idea

```
Student utterance
       ↓
  25-dim state vector  (extracted by LLM)
       ↓
  RL policy  π(s) → action  (instruct / encourage / refocus / ask question)
       ↓
  LLM generates tutor response conditioned on chosen action
       ↓
  Next student turn  →  repeat
```

This decomposition makes RL training **computationally efficient** — the policy operates over a tiny 25-dimensional state space with 4 discrete actions, requiring no GPU.

---

## Key Results

| Policy | Offline Match Rate | Notes |
|--------|-------------------|-------|
| BC (D) | 0.5127 | Supervised imitation baseline |
| FQI (D) | 0.3367 | Offline Q-learning with tree model |
| **CQL (D)** | **0.6710** | **Best on original data** |
| BC (D+) | 0.5077 | Trained on augmented data |
| FQI (D+) | 0.3267 | Trained on augmented data |
| **CQL (D+)** | **0.6740** | **Best overall — paper's target model** |

**Dialogue Generation Results:**
- Total dialogues generated: **3,000**
- Successfully solved: **2,054 (68.47%)**
- Average turns per dialogue: **5.64**
- Total RL tuples extracted: **16,930**

**Action Distribution (Original D):**

| Action | Count | Percentage |
|--------|-------|-----------|
| Ask Question | 8,256 | 48.8% |
| Encourage | 4,631 | 27.4% |
| Instruct | 4,035 | 23.8% |
| Refocus | 8 | 0.0% |

**Data Augmentation (D+):**
- Augmented dialogues generated: **2,500**
- Augmentation success rate: **56.0%**
- Encourage action increased: 27.4% → **46.0%** (better pedagogical balance)

---

## Project Architecture

```
RL_Project/
│
├── data/
│   ├── gsm8k_eval_problems.json        # 7 GSM8K test problems for evaluation
│   ├── gsm8k_test.csv                  # Full GSM8K test set
│   ├── comta_dialogues.csv             # CoMTA real tutoring dialogues (manual download)
│   ├── comta_fewshot.json              # Parsed few-shot examples cache
│   ├── rl_dataset.csv                  # Original D — (S, A, R, S') tuples
│   ├── rl_dataset_augmented.csv        # Augmented D+ tuples only
│   ├── rl_dataset_combined.csv         # D + D+ merged — used for CQL(D+) training
│   └── dialogues/
│       ├── synthetic_dialogues.jsonl   # 3,000 generated tutor-student dialogues
│       └── augmented_dialogues.jsonl   # 2,500 optimism-guided augmented dialogues
│
├── models/
│   ├── bc_policy.pkl                   # BC trained on D
│   ├── fqi_policy.pkl                  # FQI trained on D
│   ├── cql_D_policy                    # CQL trained on D (d3rlpy weights file)
│   ├── bc_Dp_policy.pkl                # BC trained on D+
│   ├── fqi_Dp_policy.pkl               # FQI trained on D+
│   ├── cql_Dp_policy                   # CQL trained on D+ (best model)
│   └── policy_index.json               # Path registry for all models
│
├── d3rlpy_logs/
│   ├── DiscreteCQL_20260407075941/     # params.json for CQL(D)
│   └── DiscreteCQL_20260415111028/     # params.json for CQL(D+)
│
├── results/
│   ├── evaluation_results.json         # Policy success rates
│   ├── eval_checkpoint.json            # Resume checkpoint for evaluation
│   ├── analysis_report.md              # Auto-generated full analysis
│   └── figures/
│       ├── figure3_success_rates.png   # Main results bar chart
│       ├── figure4_generalization.png  # Per-problem generalization
│       └── figure5_action_dist.png     # Action distribution comparison
│
├── step1_download_datasets.py          # Download GSM8K + CoMTA
├── step2_generate_dialogues_mistral.py # Generate 3,000 dialogues (Mistral)
├── step3_extract_states_actions_mistral.py  # Extract 25-dim states + actions
├── step3_day4_augmented.py             # Step 3 for augmented dialogues
├── step4_train_rl_policies.py          # Train BC, FQI, CQL on D
├── step4_day4_combined.py              # Train all policies on D+
├── step5a5b_mistral.py                 # Augmentation + Evaluation
├── step5b_final.py                     # Fixed evaluation with CQL loader
├── step6_analyze_results.py            # Generate all figures + report
├── comta_loader.py                     # CoMTA CSV format checker
├── check_cql_models.py                 # CQL model diagnostic tool
└── README.md
```

---

## Datasets

### GSM8K (Grade School Math 8K)
- **Source:** HuggingFace — `gsm8k` (OpenAI)
- **Size:** 8,500 math word problems (train: 7,473 / test: 1,319)
- **Usage:** Source of math problems for generating tutoring dialogues
- **Download:** Automatic via `datasets` library

### CoMTA (Conversational Math Tutoring Annotations)
- **Source:** Khan Academy / Miller & DiCerbo (2024) — manually downloaded from GitHub
- **Size:** 188 real student-tutor algebra dialogues (de-identified)
- **Usage:** Few-shot examples inside dialogue generation prompts
- **Download:** Manual — save as `data/comta_dialogues.csv`

> **Note:** CoMTA is not publicly available on HuggingFace. Download the JSON file from the paper's GitHub repository and convert to CSV. Run `python comta_loader.py` to verify your CSV format is detected correctly.

---

## Installation

### Prerequisites
- Python 3.10+
- Node.js (optional, only if regenerating the presentation)
- ~5 GB disk space

### Install Python dependencies

```bash
pip install mistralai datasets scikit-learn numpy pandas matplotlib d3rlpy torch
```

### Verify installation

```bash
python -c "import mistralai, datasets, sklearn, d3rlpy, torch; print('All dependencies OK')"
```

---

## API Key Setup (Free — Mistral AI)

This project uses **Mistral AI's free API** (no credit card required).

### Step 1 — Create a free account

Go to [https://console.mistral.ai](https://console.mistral.ai) and sign up with Google or email.

### Step 2 — Generate an API key

1. Click **API Keys** in the left sidebar
2. Click **"Create new key"**
3. Name it (e.g., `tutor-project`)
4. **Copy the key immediately** — it is only shown once

### Step 3 — Set the key in your terminal

```bash
# Mac / Linux
export MISTRAL_API_KEY="your-key-here"

# Windows Command Prompt
set MISTRAL_API_KEY=your-key-here

# Windows PowerShell
$env:MISTRAL_API_KEY="your-key-here"
```

### Step 4 — Make it permanent (optional)

```bash
# Mac / Linux — add to shell config
echo 'export MISTRAL_API_KEY="your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### Step 5 — Test the connection

```bash
python -c "
import os
from mistralai import Mistral
client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
resp = client.chat.complete(
    model='mistral-small-latest',
    messages=[{'role':'user','content':'Say hello in one sentence'}]
)
print(resp.choices[0].message.content)
"
```

**Free tier limits:** 1 request/second · No daily cap → All 3,000 dialogues can be generated in a single overnight run (~2–3 hours).

---

## Project Pipeline

### Step 1 — Dataset Download

Downloads GSM8K automatically and checks for your CoMTA CSV.

```bash
python step1_download_datasets.py
```

**Output:**
- `data/gsm8k_eval_problems.json` — 7 specific test problems used for generalization evaluation
- `data/gsm8k_test.csv` — full test set

**CoMTA setup (manual):**
```bash
# After placing comta_dialogues.csv in data/
python comta_loader.py   # checks format and creates comta_fewshot.json
```

---

### Step 2 — Synthetic Dialogue Generation

Uses Mistral AI to simulate 3,000 tutor-student conversations. One LLM plays both roles simultaneously.

```bash
python step2_generate_dialogues_mistral.py
```

**How it works:**
1. Sample a math problem from GSM8K train set
2. Generate 10 candidate student mistakes using the LLM
3. Randomly sample one mistake for the simulated student
4. Build a prompt with CoMTA few-shot examples
5. Call Mistral to generate the full multi-turn dialogue
6. Parse turns, assign rewards, save to JSONL

**Configuration** (edit at top of file):
```python
NUM_DIALOGUES = 3000      # reduce to 50 for testing
MAX_TURNS     = 20        # max turns per conversation
DISCOUNT_FACTOR = 0.9     # γ for discounted reward
```

**Output:** `data/dialogues/synthetic_dialogues.jsonl`

Each record:
```json
{
  "dialogue_id": "dial_00001",
  "gsm8k_question": "If a car travels...",
  "correct_answer": "#### 42",
  "student_mistake": "The student divides instead of multiplies",
  "turns": [
    {"role": "tutor",   "text": "What information do you have?", "reward": 0},
    {"role": "student", "text": "The speed is 30 mph.",          "reward": 0},
    ...
  ],
  "solved": true,
  "final_reward": 1,
  "discounted_value": 0.531,
  "num_turns": 8
}
```

**Note:** Script is resumable — if interrupted, re-run and it skips already-generated dialogues.

---

### Step 3 — State & Action Extraction

Converts each dialogue turn into the RL training format `(S_n, A_n, R_n, S_{n+1})`.

```bash
# For synthetic dialogues (Day 2 output)
python step3_extract_states_actions_mistral.py

# For augmented dialogues (Day 4 — after Step 5a)
python step3_day4_augmented.py
```

**State extraction:** Calls Mistral with 20 yes/no questions about the student's utterance → 25-dimensional binary + numeric vector.

**Action extraction:** Calls Mistral to classify each tutor utterance as one of 4 high-level actions.

**Output:** `data/rl_dataset.csv` with columns:

```
dialogue_id, turn, s0, s1, ..., s24, action, action_name, reward, ns0, ns1, ..., ns24, done
```

**Note on augmented format:** Augmented dialogues store conversation differently (`partial_history` + `continuation` text fields) vs original (`turns` list). The Day 4 script handles both formats automatically.

---

### Step 4 — RL Policy Training

Trains three offline RL policies on the extracted dataset.

```bash
# Train on original D
python step4_train_rl_policies.py

# Train on combined D + D+ (run after Step 5a + Step 3 on augmented data)
python step4_day4_combined.py
```

#### Behavioral Cloning (BC)
- Supervised learning — imitates dataset action distribution
- Architecture: MLP `[128, 128]` with ReLU, Adam optimizer
- Loss: Cross-entropy on action labels
- No notion of future rewards

#### Fitted Q-Iteration (FQI)
- Offline Q-learning with `ExtraTreesRegressor` as Q-function
- 25 estimators, 50 Bellman iterations
- Input: `concat(state, one-hot action)` → Q-value
- Learns long-term value without GPU

#### Conservative Q-Learning (CQL) — Paper's Best
- d3rlpy `DiscreteCQL` — adds conservatism penalty to prevent Q-value overestimation
- Trains on CPU, no GPU required

**CQL Hyperparameters (Appendix 16 — exact paper values):**

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-5 |
| Optimizer | Adam (ε = 1e-2/32) |
| Batch size | 32 |
| Alpha (conservatism) | 4.0 |
| Gamma (discount) | 0.9 |
| N quantiles | 200 |
| Target update interval | 2000 |
| N steps | 1,000,000 |

**Saved models:**
```
models/bc_policy.pkl       → sklearn MLPClassifier
models/fqi_policy.pkl      → (ExtraTreesRegressor, OneHotEncoder)
models/cql_D_policy        → d3rlpy weights file
models/bc_Dp_policy.pkl    → BC on combined D+
models/fqi_Dp_policy.pkl   → FQI on combined D+
models/cql_Dp_policy       → CQL on combined D+ (best model)
```

> **Important — CQL loading:** d3rlpy v2.8.1 saves weights separately from `params.json`. The weights go to `models/cql_D_policy` and the architecture config goes to `d3rlpy_logs/DiscreteCQL_TIMESTAMP/params.json`. The evaluation script handles this automatically.

---

### Step 5a — Optimism-guided Data Augmentation

Implements Algorithm 2 from the paper — generates 2,500 new dialogues by intervening with better tutor actions.

```bash
python step5a5b_mistral.py --mode augment
```

**Algorithm:**
1. Train FQI Q-function on original dataset D
2. Train BC policy to model dataset distribution
3. Score each transition: `val = max_a Q(s,a) − Q(s, π_bc(s))`
4. Select top-500 candidates with highest optimism score
5. For each candidate, generate new tutor response conditioned on optimal action `a*`
6. Complete the full dialogue using the LLM baseline
7. Repeat × 5 scenarios per candidate → **2,500 new dialogues**

**Output:** `data/dialogues/augmented_dialogues.jsonl`

---

### Step 5b — Student Simulator Evaluation

Evaluates all trained policies by running 300 simulated conversations each.

```bash
python step5b_final.py
```

**Setup:**
- Problems sampled from GSM8K test set
- Student role: Mistral LLM prompted as a distracted 6th-grader with a sampled mistake
- Tutor role: RL policy selects action → Mistral generates response conditioned on that action
- Success: Student produces the correct numeric answer within 20 turns

**Resume support:** Results are saved to `results/eval_checkpoint.json` after each policy. Re-running automatically skips completed policies.

---

### Step 6 — Analysis & Figures

Generates all paper figures and a full markdown report.

```bash
python step6_analyze_results.py
```

**Output:**
- `results/figures/figure3_success_rates.png` — main results bar chart
- `results/figures/figure4_generalization.png` — per-problem generalization
- `results/figures/figure5_action_dist.png` — action distribution comparison
- `results/analysis_report.md` — full analysis with tables

---

## Day-by-Day Execution Plan

| Day | Date | Commands | Est. Time |
|-----|------|----------|-----------|
| **Day 1** | Apr 4 | Setup + Step 1 + Step 2 (overnight) | ~3 hrs |
| **Day 2** | Apr 5 | Step 3 on synthetic data (leave running) | ~4 hrs |
| **Day 3** | Apr 6 | Step 4 (D) + Step 5a augmentation (overnight) | ~2 hrs |
| **Day 4** | Apr 7 | Step 3 on augmented + Step 4 (D+) + Step 5b | ~6 hrs |
| **Day 5** | Apr 8 | Step 6 analysis + report writing | ~2 hrs |

```bash
# ── DAY 1 ──────────────────────────────────────────────
export MISTRAL_API_KEY="your-key-here"
pip install mistralai datasets scikit-learn numpy pandas matplotlib d3rlpy torch

python step1_download_datasets.py
python comta_loader.py
python step2_generate_dialogues_mistral.py          # leave overnight

# ── DAY 2 ──────────────────────────────────────────────
python step3_extract_states_actions_mistral.py      # leave running all day

# ── DAY 3 ──────────────────────────────────────────────
python step4_train_rl_policies.py                   # ~1 hour, no API
python step5a5b_mistral.py --mode augment           # leave overnight

# ── DAY 4 ──────────────────────────────────────────────
python step3_day4_augmented.py                      # process augmented data
python step4_day4_combined.py                       # train CQL(D+)
python step5b_final.py                              # evaluate all policies

# ── DAY 5 ──────────────────────────────────────────────
python step6_analyze_results.py                     # figures + report
```

---

## State Representation (25 Dimensions)

Each student utterance is mapped to a fixed-size 25-dimensional vector by calling the LLM with structured yes/no questions.

| Dim | Feature | Type |
|-----|---------|------|
| 1 | Student producing math-related content? | Binary |
| 2 | Student solved the problem correctly? | Binary |
| 3 | Student asking to re-explain? | Binary |
| 4 | Student repeating what tutor said? | Binary |
| 5 | Student going off-topic? | Binary |
| 6 | Utterance unrelated to math? | Binary |
| 7 | Student explicitly asking a question? | Binary |
| 8 | Student describing what they're stuck on? | Binary |
| 9 | Tutor asked diagnostic question? | Binary |
| 10 | Student expressing frustration? | Binary |
| 11 | Student expressing uncertainty? | Binary |
| 12 | Student expressing positive sentiment? | Binary |
| 13 | Student asking for a break? | Binary |
| 14 | Talking about the problem at hand? | Binary |
| 15 | Talking about math background? | Binary |
| 16 | Discussing related math concepts? | Binary |
| 17 | Student wrote down an equation? | Binary |
| 18 | Tutor asking a question this turn? | Binary |
| 19 | Student made a mistake this turn? | Binary (majority vote) |
| 20 | Tutor tried to refocus student? | Binary (cumulative) |
| 21 | How many questions tutor asked so far | Integer (count) |
| 22 | How many questions student asked so far | Integer (count) |
| 23 | Current turn number | Integer (1–20) |
| 24 | Math density score | Float (0–1) |
| 25 | Mathematical reasoning score | Float (0–1) |

> Dimensions 24–25 use heuristic approximations: math token ratio and reasoning keyword presence. The original paper uses Wang & Demszky (2024) classifiers.

---

## Action Space

The RL policy selects one of 4 high-level tutor actions at each turn:

| ID | Action | Description | Example |
|----|--------|-------------|---------|
| 0 | **Instruct** | Explain a concept or correct a mistake | "Let's think step by step. What formula applies here?" |
| 1 | **Encourage** | Motivate or praise the student | "Great thinking! You're really close — keep going!" |
| 2 | **Refocus** | Bring a distracted student back on topic | "Let's get back to the problem. What were we finding?" |
| 3 | **Ask Question** | Probe student understanding | "What happens to the time when speed doubles?" |

---

## RL Policies

### Behavioral Cloning (BC)
Learns a supervised mapping `state → action` by imitating the dataset. No future reward awareness. Serves as the supervised baseline.

### Fitted Q-Iteration (FQI)
Iteratively fits an `ExtraTreesRegressor` to Bellman targets. Learns Q(s,a) estimates that account for future rewards. Does not require GPU or neural networks.

### Conservative Q-Learning (CQL)
Adds a conservatism penalty to the standard Bellman loss:

```
L_CQL(μ) = L_Bellman(μ) + α · E[log Σ_a exp Q(s,a) − Q(s,a_dataset)]
```

The penalty prevents overestimating Q-values for unseen `(state, action)` pairs — the core challenge in offline RL. Uses `d3rlpy.algos.DiscreteCQL`.

---

## Extended Policies (Beyond the Paper)

This project additionally implements:

### Exploration Policy (ε-greedy)
Wraps any trained policy with ε-greedy action selection. With probability ε, selects a random action; otherwise follows the greedy policy. Prevents early convergence to suboptimal tutoring strategies.

### Deep Q-Network (DQN)
Neural network approximates Q(s,a) directly from state. Input: 25-dim state → Output: Q-values for all 4 actions. Uses experience replay and target network for stability.

### Implicit Q-Learning (IQL)
Advanced offline RL that avoids querying unseen (s,a) pairs during training. Learns a value function purely from observed transitions. More stable than DQN in offline settings.

---

## Reward Function & Discount Factor

```
Reward:
  R = +1  if student solves the problem correctly
  R = -1  if maximum turns (20) are reached without success

Discounted return:
  G = R₁ + γR₂ + γ²R₃ + γ³R₄ + ...

Discount factor:  γ = 0.9
```

The discount factor ensures the policy prefers students solving problems **sooner** (fewer hints needed = better teaching). With γ = 0.9, a solution at turn 3 is worth more than the same solution at turn 15.

---

## Results

### Dialogue Generation (Step 2)

```
Total dialogues:    3,000
Solved:             2,054  (68.47%)
Failed:               946  (31.53%)
Average turns:        5.64
Discount factor:      0.9
```

### RL Dataset (Step 3)

```
Total RL tuples:   16,930
State dimensions:      25
Action classes:         4
```

**Action distribution — Original D:**

```
Ask Question  ████████████████████████  48.8%  (8,256)
Encourage     ██████████████           27.4%  (4,631)
Instruct      ████████████             23.8%  (4,035)
Refocus                                 0.0%  (    8)
```

### Policy Training (Step 4)

```
Policy      D       D+
──────────────────────
BC        0.5127  0.5077
FQI       0.3367  0.3267
CQL       0.6710  0.6740  ← Best
```

### Data Augmentation (Step 5a)

```
New dialogues:       2,500
Success rate:        56.0%

Action shift (D → D+):
  Encourage:  27.4% → 46.0%  ↑ more emotional support
  Ask Q:      48.8% → 33.8%  ↓ less dominant
  Instruct:   23.8% → 18.2%
  Refocus:     0.0% →  1.9%
```

---

## Challenges & Fixes

| Challenge | Fix Applied |
|-----------|-------------|
| CoMTA not on HuggingFace | Manual JSON→CSV download + `comta_loader.py` auto-detects column format |
| Mistral rate limits (503/401) | Retry logic with exponential backoff (3 attempts, 30s wait) |
| ~40,000 total API calls | 1.2s pacing between calls, resumable checkpoints throughout |
| CQL loading error `[Errno 20] Not a directory` | d3rlpy v2.8.1 saves weights + params separately; `check_cql_models.py` diagnoses, `step5b_final.py` loads correctly from `d3rlpy_logs/` |
| Augmented dialogue format mismatch | `parse_augmented_turns()` detects `partial_history + continuation` format vs `turns` list |
| Step 5b interrupted mid-run | `eval_checkpoint.json` saves after every policy; resume skips completed ones |
| Bar chart showing `0` / `1` labels | Added `dataLabelFormatCode:"0.0000"` to pptxgenjs chart config |

---

## Differences from Original Paper

| Aspect | Original Paper | This Project |
|--------|---------------|--------------|
| LLM used | Claude 3 Sonnet | Mistral Small (free) |
| API cost | Paid | Free |
| Training data | SAT-level problems (Kumar et al.) | GSM8K train set |
| Dims 24–25 | Wang & Demszky classifier | Heuristic approximation |
| Evaluation | 300 conversations | 300 conversations (same) |
| Extended policies | BC, FQI, CQL | + Exploration, DQN, IQL |
| Implementation | Internal (Anthropic) | Fully open-source |

---

## Project Structure Quick Reference

```bash
# Check your CQL model files
python check_cql_models.py

# Verify CoMTA CSV format
python comta_loader.py

# Resume interrupted evaluation
python step5b_final.py    # automatically skips completed policies

# Regenerate all figures without re-running evaluation
python step6_analyze_results.py
```

---

## Citation

This project replicates:

```bibtex
@inproceedings{nam2025efficient,
  title     = {Efficient Reinforcement Learning for Optimizing Multi-turn
               Student Outcomes with LLM Tutors},
  author    = {Nam, Hyunji and Gottesman, Omer and Zhang, Amy and
               Foster, Dean and Brunskill, Emma and Ungar, Lyle},
  booktitle = {NeurIPS 2025 Workshop: Multi-Turn Interactions in Large Language Models},
  year      = {2025}
}
```

Other references:
- GSM8K: Cobbe et al. (2021) — *Training Verifiers to Solve Math Word Problems*
- CoMTA: Miller & DiCerbo (2024) — *LLM Based Math Tutoring: Challenges and Dataset*
- CQL: Kumar et al. (2020) — *Conservative Q-Learning for Offline Reinforcement Learning*
- d3rlpy: Seno & Imai (2022) — *d3rlpy: An Offline Deep Reinforcement Learning Library*

---

## License

This project is released under the MIT License for academic and educational use.

---

<div align="center">

**Pavan Ramavath**
B.Tech Artificial Intelligence | SVNIT Surat

*Built as a final-year project replication of NeurIPS 2025 Workshop paper*

</div>
