"""
Step 4 — DAY 4 VERSION (Train CQL on D+)
==========================================
What this does:
  1. Loads rl_dataset.csv          (original D — already exists from Day 2)
  2. Loads rl_dataset_augmented.csv (augmented D+ — just created by step3_day4)
  3. Merges them → rl_dataset_combined.csv
  4. Retrains CQL on the combined D+ dataset → saves as cql_Dp_policy/
  5. Also retrains BC(D+) and FQI(D+) for fair comparison

Models saved:
  models/bc_Dp_policy.pkl     ← BC trained on D+
  models/fqi_Dp_policy.pkl    ← FQI trained on D+
  models/cql_Dp_policy/       ← CQL trained on D+  (paper's best model)

Run:
  python step4_day4_combined.py

After this → run step5a5b_mistral.py --mode evaluate
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Try d3rlpy for CQL ────────────────────────────────────────────────────────
try:
    import d3rlpy
    from d3rlpy.dataset import MDPDataset
    HAS_D3RLPY = True
    print("d3rlpy found — CQL training enabled.")
except ImportError:
    HAS_D3RLPY = False
    print("d3rlpy not found — only BC and FQI will train.")
    print("Install: pip install d3rlpy torch")

# ── Paths ─────────────────────────────────────────────────────────────────────
ORIG_CSV     = Path("data/rl_dataset.csv")             # original D
AUG_CSV      = Path("data/rl_dataset_augmented.csv")   # augmented D+
COMBINED_CSV = Path("data/rl_dataset_combined.csv")    # merged D+
MODEL_DIR    = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ── Hyperparameters (Appendix 16 — exact paper values) ────────────────────────
GAMMA             = 0.9
ALPHA             = 4.0
LEARNING_RATE     = 5e-5
ADAM_EPSILON      = 1e-2 / 32
BATCH_SIZE        = 32
N_QUANTILES       = 200
TARGET_UPDATE     = 2000
N_STEPS           = 1_000_000   # reduce to 50_000 for quick test
N_STEPS_PER_EPOCH = 10_000
N_ACTIONS         = 4

ACTION_NAMES = {0: "instruct", 1: "encourage",
                2: "refocus",  3: "ask_question"}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — MERGE ORIGINAL + AUGMENTED INTO COMBINED CSV
# ══════════════════════════════════════════════════════════════════════════════

def merge_datasets() -> pd.DataFrame:
    """Merge rl_dataset.csv + rl_dataset_augmented.csv → rl_dataset_combined.csv"""
    print("\n[1] Merging datasets...")

    if not ORIG_CSV.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {ORIG_CSV}\n"
            "Make sure Day 2 Step 3 completed successfully."
        )
    if not AUG_CSV.exists():
        raise FileNotFoundError(
            f"Augmented dataset not found: {AUG_CSV}\n"
            "Make sure Day 4 Step 3 (step3_day4_augmented.py) completed."
        )

    df_orig = pd.read_csv(ORIG_CSV)
    df_aug  = pd.read_csv(AUG_CSV)

    print(f"  Original D  : {len(df_orig):,} tuples")
    print(f"  Augmented D+: {len(df_aug):,} tuples")

    df_combined = pd.concat([df_orig, df_aug], ignore_index=True)
    df_combined.to_csv(COMBINED_CSV, index=False)

    print(f"  Combined D+ : {len(df_combined):,} tuples → {COMBINED_CSV}")
    return df_combined


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD ARRAYS FROM CSV
# ══════════════════════════════════════════════════════════════════════════════

def load_arrays(df: pd.DataFrame):
    """Extract numpy arrays from the combined dataframe."""
    state_cols = [f"s{i}"  for i in range(25)]
    ns_cols    = [f"ns{i}" for i in range(25)]

    states      = df[state_cols].values.astype(np.float32)
    actions     = df["action"].values.astype(np.int32)
    rewards     = df["reward"].values.astype(np.float32)
    next_states = df[ns_cols].values.astype(np.float32)
    terminals   = df["done"].values.astype(np.float32)

    print(f"\n  State shape  : {states.shape}")
    print(f"  Reward mean  : {rewards.mean():.4f}")
    print(f"  Action dist  : "
          f"{ {ACTION_NAMES[i]: int((actions==i).sum()) for i in range(4)} }")

    return states, actions, rewards, next_states, terminals


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN BC on D+
# ══════════════════════════════════════════════════════════════════════════════

def train_bc_dp(states, actions):
    print("\n[2] Training BC(D+)...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        states, actions, test_size=0.1, random_state=42
    )
    bc = MLPClassifier(
        hidden_layer_sizes=(128, 128),
        activation="relu",
        solver="adam",
        alpha=0.1,
        learning_rate_init=1e-3,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    bc.fit(X_tr, y_tr)
    acc = accuracy_score(y_val, bc.predict(X_val))
    print(f"  BC(D+) validation accuracy: {acc:.4f}")

    save_path = MODEL_DIR / "bc_Dp_policy.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(bc, f)
    print(f"  Saved → {save_path}")
    return bc


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN FQI on D+
# ══════════════════════════════════════════════════════════════════════════════

def train_fqi_dp(states, actions, rewards, next_states, terminals):
    print("\n[3] Training FQI(D+)...")

    enc = OneHotEncoder(
        categories=[list(range(N_ACTIONS))],
        sparse_output=False,
    )
    enc.fit(np.arange(N_ACTIONS).reshape(-1, 1))

    def encode_sa(s_batch, a_batch):
        return np.hstack([s_batch, enc.transform(a_batch.reshape(-1, 1))])

    q_targets = rewards.copy()
    fqi_model = None

    for iteration in range(50):
        X = encode_sa(states, actions)
        fqi_model = ExtraTreesRegressor(
            n_estimators=25, min_samples_split=2, random_state=42
        )
        fqi_model.fit(X, q_targets)

        all_next_q = np.stack([
            fqi_model.predict(
                encode_sa(next_states, np.full(len(next_states), a, dtype=np.int32))
            )
            for a in range(N_ACTIONS)
        ], axis=1)

        q_targets = rewards + GAMMA * all_next_q.max(axis=1) * (1 - terminals)

        if (iteration + 1) % 10 == 0:
            print(f"  FQI iteration {iteration+1}/50 "
                  f"| mean Q: {q_targets.mean():.4f}")

    save_path = MODEL_DIR / "fqi_Dp_policy.pkl"
    with open(save_path, "wb") as f:
        pickle.dump((fqi_model, enc), f)
    print(f"  Saved → {save_path}")
    return fqi_model, enc


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN CQL on D+  (paper's best model)
# ══════════════════════════════════════════════════════════════════════════════

def train_cql_dp(states, actions, rewards, next_states, terminals):
    if not HAS_D3RLPY:
        print("\n[4] Skipping CQL(D+) — d3rlpy not installed.")
        print("    Install: pip install d3rlpy torch")
        return None

    print("\n[4] Training CQL(D+) — paper's best model...")
    print(f"    N_STEPS={N_STEPS:,} | This takes ~30–60 min on CPU.")
    print(f"    Tip: reduce N_STEPS to 50_000 at top of file for a quick test.\n")

    dataset = MDPDataset(
        observations = states,
        actions      = actions,
        rewards      = rewards,
        terminals    = terminals,
    )

    cql = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate          = LEARNING_RATE,
        optim_factory          = d3rlpy.optimizers.AdamFactory(eps=ADAM_EPSILON),
        batch_size             = BATCH_SIZE,
        alpha                  = ALPHA,
        gamma                  = GAMMA,
        target_update_interval = TARGET_UPDATE,
        reward_scaler          = d3rlpy.preprocessing.MinMaxRewardScaler(),
    ).create(device="cpu")

    save_path = str(MODEL_DIR / "cql_Dp_policy")
    cql.fit(
        dataset,
        n_steps           = N_STEPS,
        n_steps_per_epoch = N_STEPS_PER_EPOCH,
        evaluators        = {},
    )
    cql.save(save_path)
    print(f"  CQL(D+) saved → {save_path}/")
    return cql


# ══════════════════════════════════════════════════════════════════════════════
# OFFLINE SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def offline_check(states, actions, bc, fqi_model, enc, cql=None):
    print("\n[5] Offline sanity check (match rate vs dataset actions)...")
    _, X_te, _, y_te = train_test_split(
        states, actions, test_size=0.2, random_state=42
    )

    def fqi_predict(s):
        qs = np.stack([
            fqi_model.predict(
                np.hstack([s, enc.transform(
                    np.full(len(s), a, dtype=np.int32).reshape(-1, 1)
                )])
            )
            for a in range(N_ACTIONS)
        ], axis=1)
        return qs.argmax(axis=1)

    bc_acc  = accuracy_score(y_te, bc.predict(X_te))
    fqi_acc = accuracy_score(y_te, fqi_predict(X_te))
    print(f"  BC(D+)  match rate: {bc_acc:.4f}")
    print(f"  FQI(D+) match rate: {fqi_acc:.4f}")

    if cql:
        cql_preds = cql.predict(X_te)
        cql_acc   = accuracy_score(y_te, cql_preds)
        print(f"  CQL(D+) match rate: {cql_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Step 4 — DAY 4: Train all policies on D+ (combined data)")
    print("=" * 60)

    # 1. Merge datasets
    df_combined = merge_datasets()

    # 2. Extract arrays
    print("\n[Loading arrays from combined dataset...]")
    states, actions, rewards, next_states, terminals = load_arrays(df_combined)

    # 3. Train BC(D+)
    bc = train_bc_dp(states, actions)

    # 4. Train FQI(D+)
    fqi_model, enc = train_fqi_dp(
        states, actions, rewards, next_states, terminals
    )

    # 5. Train CQL(D+) — the paper's best model
    cql = train_cql_dp(states, actions, rewards, next_states, terminals)

    # 6. Sanity check
    offline_check(states, actions, bc, fqi_model, enc, cql)

    # 7. Save policy index
    policy_index = {
        "bc_D":    str(MODEL_DIR / "bc_policy.pkl"),        # from Day 2
        "fqi_D":   str(MODEL_DIR / "fqi_policy.pkl"),       # from Day 2
        "cql_D":   str(MODEL_DIR / "cql_D_policy"),         # from Day 2
        "bc_Dp":   str(MODEL_DIR / "bc_Dp_policy.pkl"),     # trained now
        "fqi_Dp":  str(MODEL_DIR / "fqi_Dp_policy.pkl"),    # trained now
        "cql_Dp":  str(MODEL_DIR / "cql_Dp_policy"),        # trained now ← BEST
    }
    with open(MODEL_DIR / "policy_index.json", "w") as f:
        json.dump(policy_index, f, indent=2)

    # Final summary
    print("\n" + "=" * 60)
    print("STEP 4 DAY 4 COMPLETE")
    print("=" * 60)
    print(f"  Combined D+ size  : {len(df_combined):,} tuples")
    print(f"  Models saved to   : {MODEL_DIR}/")
    print(f"    bc_Dp_policy.pkl")
    print(f"    fqi_Dp_policy.pkl")
    if cql:
        print(f"    cql_Dp_policy/    ← USE THIS for evaluation")
    print(f"\nNext → python step5a5b_mistral.py --mode evaluate")


if __name__ == "__main__":
    main()