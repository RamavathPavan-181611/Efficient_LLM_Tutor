"""
Step 4: Offline RL Training
============================
Paper: "Efficient RL for Optimizing Multi-turn Student Outcomes with LLM Tutors"

Trains four policies (matching paper's baselines):
  1. Behavioral Cloning (BC)     — supervised imitation of dataset actions
  2. Fitted Q-Iteration (FQI)    — offline Q-learning with tree-based Q-function
  3. Conservative Q-Learning (CQL) on D   — paper's main method, original data
  4. Conservative Q-Learning (CQL) on D+  — paper's best result, augmented data

Hyperparameters from Appendix 16 (exact):
  - learning rate   : 5e-5
  - optimizer       : Adam, epsilon = 1e-2 / 32
  - batch size      : 32
  - alpha (CQL)     : 4.0
  - gamma           : 0.9
  - n_quantiles     : 200
  - target_update   : 2000
  - n_steps         : 1,000,000
  - n_steps_per_epoch: 10,000

Install:
  pip install d3rlpy scikit-learn numpy pandas torch
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Try importing d3rlpy (for CQL) ────────────────────────────────────────────
try:
    import d3rlpy
    from d3rlpy.dataset import MDPDataset
    HAS_D3RLPY = True
    print("d3rlpy found — CQL training enabled.")
except ImportError:
    HAS_D3RLPY = False
    print("d3rlpy not installed. Only BC and FQI will run.")
    print("Install with: pip install d3rlpy torch")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_CSV    = Path("data/rl_dataset.csv")
MODEL_DIR   = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ── Hyperparameters (Appendix 16, exact) ──────────────────────────────────────
GAMMA             = 0.9
ALPHA             = 4.0          # CQL conservatism weight
LEARNING_RATE     = 5e-5
ADAM_EPSILON      = 1e-2 / 32
BATCH_SIZE        = 32
N_QUANTILES       = 200
TARGET_UPDATE     = 2000
N_STEPS           = 1_000_000    # reduce to 100_000 for quick test
N_STEPS_PER_EPOCH = 10_000
N_ACTIONS         = 4            # instruct, encourage, refocus, ask_question

ACTION_NAMES = {0: "instruct", 1: "encourage", 2: "refocus", 3: "ask_question"}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(csv_path: Path):
    """
    Load the RL dataset from Step 3 CSV.
    Returns arrays: states, actions, rewards, next_states, terminals
    """
    df = pd.read_csv(csv_path)

    state_cols      = [f"s{i}"  for i in range(25)]
    next_state_cols = [f"ns{i}" for i in range(25)]

    states      = df[state_cols].values.astype(np.float32)
    actions     = df["action"].values.astype(np.int32)
    rewards     = df["reward"].values.astype(np.float32)
    next_states = df[next_state_cols].values.astype(np.float32)
    terminals   = df["done"].values.astype(np.float32)

    print(f"  Loaded {len(states)} transitions")
    print(f"  State shape : {states.shape}")
    print(f"  Action dist : { {ACTION_NAMES[i]: int((actions==i).sum()) for i in range(4)} }")
    print(f"  Reward mean : {rewards.mean():.4f}")
    return states, actions, rewards, next_states, terminals


# ══════════════════════════════════════════════════════════════════════════════
# POLICY 1 — BEHAVIORAL CLONING
# ══════════════════════════════════════════════════════════════════════════════

class BehavioralCloningPolicy:
    """
    Supervised imitation: learn to predict the action taken in the dataset.
    Uses an MLP matching the paper's BC setup (Appendix 17):
      - hidden dims: [128, 128], ReLU
      - Adam, lr=1e-3, weight decay=1e-1
      - cross-entropy loss, max 1000 epochs
    """
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 128),
            activation="relu",
            solver="adam",
            alpha=0.1,           # weight decay = 1e-1
            learning_rate_init=1e-3,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )

    def train(self, states, actions):
        print("  Training Behavioral Cloning...")
        X_train, X_val, y_train, y_val = train_test_split(
            states, actions, test_size=0.1, random_state=42
        )
        self.model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, self.model.predict(X_val))
        print(f"  BC validation accuracy: {val_acc:.4f}")
        return val_acc

    def predict(self, state: np.ndarray) -> int:
        """Choose action for a given 25-dim state."""
        return int(self.model.predict(state.reshape(1, -1))[0])

    def predict_proba(self, state: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(state.reshape(1, -1))[0]

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"  Saved BC model → {path}")

    @classmethod
    def load(cls, path: Path):
        obj = cls()
        with open(path, "rb") as f:
            obj.model = pickle.load(f)
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# POLICY 2 — FITTED Q-ITERATION (FQI)
# ══════════════════════════════════════════════════════════════════════════════

class FittedQPolicy:
    """
    Fitted Q-Iteration with ExtraTreesRegressor (Appendix 17 Q-function setup):
      - sklearn ExtraTreesRegressor, 25 estimators
      - min_samples_split: 2
      - one-hot action encoding concatenated with 25-dim state
      - 50 iterations over the full dataset
    """
    def __init__(self, n_estimators=25, n_iterations=50):
        self.n_estimators  = n_estimators
        self.n_iterations  = n_iterations
        self.q_model       = None
        self.action_encoder = OneHotEncoder(
            categories=[list(range(N_ACTIONS))],
            sparse_output=False,
        )

    def _encode_sa(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Concatenate state with one-hot action."""
        a_onehot = self.action_encoder.transform(actions.reshape(-1, 1))
        return np.hstack([states, a_onehot])

    def train(self, states, actions, rewards, next_states, terminals):
        print("  Training Fitted Q-Iteration...")
        self.action_encoder.fit(np.arange(N_ACTIONS).reshape(-1, 1))

        # Initialise Q-targets to rewards
        q_targets = rewards.copy()

        for iteration in range(self.n_iterations):
            # Build features
            X = self._encode_sa(states, actions)

            # Fit regression tree
            model = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                min_samples_split=2,
                random_state=42,
            )
            model.fit(X, q_targets)

            # Compute max_a Q(s', a) for Bellman targets
            all_next_q = np.stack([
                model.predict(self._encode_sa(
                    next_states,
                    np.full(len(next_states), a, dtype=np.int32)
                ))
                for a in range(N_ACTIONS)
            ], axis=1)  # shape: (N, 4)

            max_next_q = all_next_q.max(axis=1)

            # Bellman update: r + γ * max_a Q(s', a) for non-terminal transitions
            q_targets = rewards + GAMMA * max_next_q * (1 - terminals)

            if (iteration + 1) % 10 == 0:
                print(f"    FQI iteration {iteration+1}/{self.n_iterations} "
                      f"| mean Q: {q_targets.mean():.4f}")

        self.q_model = model
        print("  FQI training complete.")

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q(s, a) for all 4 actions."""
        states_rep = np.tile(state, (N_ACTIONS, 1))
        actions_rep = np.arange(N_ACTIONS, dtype=np.int32)
        X = self._encode_sa(states_rep, actions_rep)
        return self.q_model.predict(X)

    def predict(self, state: np.ndarray) -> int:
        """Greedy action: argmax_a Q(s, a)."""
        return int(np.argmax(self.q_values(state)))

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump((self.q_model, self.action_encoder), f)
        print(f"  Saved FQI model → {path}")

    @classmethod
    def load(cls, path: Path):
        obj = cls()
        with open(path, "rb") as f:
            obj.q_model, obj.action_encoder = pickle.load(f)
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# POLICY 3 — CONSERVATIVE Q-LEARNING (CQL) via d3rlpy
# ══════════════════════════════════════════════════════════════════════════════

def train_cql(states, actions, rewards, next_states, terminals,
              tag: str = "cql") -> object | None:
    """
    Train CQL using d3rlpy with exact hyperparameters from Appendix 16.
    Returns the trained algo object, or None if d3rlpy not installed.
    """
    if not HAS_D3RLPY:
        print("  Skipping CQL (d3rlpy not installed).")
        return None

    print(f"  Training CQL ({tag})...")

    # Build d3rlpy MDPDataset
    dataset = MDPDataset(
        observations = states,
        actions      = actions,
        rewards      = rewards,
        terminals    = terminals,
    )

    # Configure DiscreteCQL with paper's exact hyperparameters (Appendix 16)
    cql = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate        = LEARNING_RATE,
        optim_factory        = d3rlpy.optimizers.AdamFactory(eps=ADAM_EPSILON),
        batch_size           = BATCH_SIZE,
        alpha                = ALPHA,
        gamma                = GAMMA,
        n_critics            = 1,
        target_update_interval = TARGET_UPDATE,
        reward_scaler        = d3rlpy.preprocessing.MinMaxRewardScaler(),
    ).create(device="cpu")   # No GPU needed (paper trains on CPU)

    # Train
    cql.fit(
        dataset,
        n_steps           = N_STEPS,
        n_steps_per_epoch = N_STEPS_PER_EPOCH,
        evaluators        = {},
    )

    # Save
    save_path = MODEL_DIR / f"{tag}_policy"
    cql.save(str(save_path))
    print(f"  Saved CQL ({tag}) → {save_path}")
    return cql


# ══════════════════════════════════════════════════════════════════════════════
# POLICY WRAPPER — unified interface for evaluation
# ══════════════════════════════════════════════════════════════════════════════

class PolicyWrapper:
    """Unified predict(state) → action interface for all policy types."""

    def __init__(self, policy_type: str, model):
        self.policy_type = policy_type
        self.model       = model

    def predict(self, state: np.ndarray) -> int:
        if self.policy_type in ("bc", "fqi"):
            return self.model.predict(state)
        elif self.policy_type == "cql":
            # d3rlpy predict expects (batch, obs_dim)
            action = self.model.predict(state.reshape(1, -1))
            return int(action[0])
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")

    def action_name(self, state: np.ndarray) -> str:
        return ACTION_NAMES[self.predict(state)]


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK — compare policies on held-out transitions
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_policies_offline(policies: dict, states, actions):
    """
    Offline evaluation: compare each policy's chosen action vs dataset action.
    This is NOT the real evaluation (which needs student simulation in Step 5),
    but it's a quick sanity check that training worked.
    """
    print("\n" + "=" * 50)
    print("Offline policy comparison (vs dataset actions)")
    print("=" * 50)

    _, X_test, _, y_test = train_test_split(
        states, actions, test_size=0.2, random_state=42
    )

    for name, policy in policies.items():
        preds = [policy.predict(s) for s in X_test]
        acc   = accuracy_score(y_test, preds)
        print(f"  {name:20s}: match rate = {acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Step 4: Offline RL Training")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading RL dataset...")
    states, actions, rewards, next_states, terminals = load_dataset(DATA_CSV)

    trained_policies = {}

    # ── Train BC ─────────────────────────────────────────────────────────────
    print("\n[1/3] Behavioral Cloning")
    bc = BehavioralCloningPolicy()
    bc.train(states, actions)
    bc.save(MODEL_DIR / "bc_policy.pkl")
    trained_policies["BC"] = PolicyWrapper("bc", bc)

    # ── Train FQI ────────────────────────────────────────────────────────────
    print("\n[2/3] Fitted Q-Iteration")
    fqi = FittedQPolicy()
    fqi.train(states, actions, rewards, next_states, terminals)
    fqi.save(MODEL_DIR / "fqi_policy.pkl")
    trained_policies["FQI"] = PolicyWrapper("fqi", fqi)

    # ── Train CQL (original data D) ──────────────────────────────────────────
    print("\n[3/3] Conservative Q-Learning (CQL on D)")
    cql_model = train_cql(states, actions, rewards, next_states, terminals, tag="cql_D")
    if cql_model:
        trained_policies["CQL(D)"] = PolicyWrapper("cql", cql_model)

    # ── Offline sanity check ─────────────────────────────────────────────────
    evaluate_policies_offline(trained_policies, states, actions)

    # ── Save policy index ────────────────────────────────────────────────────
    policy_index = {
        "bc":    str(MODEL_DIR / "bc_policy.pkl"),
        "fqi":   str(MODEL_DIR / "fqi_policy.pkl"),
        "cql_D": str(MODEL_DIR / "cql_D_policy"),
    }
    with open(MODEL_DIR / "policy_index.json", "w") as f:
        json.dump(policy_index, f, indent=2)

    print("\n" + "=" * 60)
    print("STEP 4 COMPLETE — Trained Policies Summary")
    print("=" * 60)
    print(f"  Models saved to : {MODEL_DIR}/")
    print(f"  BC              : bc_policy.pkl")
    print(f"  FQI             : fqi_policy.pkl")
    if cql_model:
        print(f"  CQL(D)          : cql_D_policy/")
    print("\nNote: CQL on augmented data D+ comes after Step 5 (data augmentation)")
    print("Next step -> Step 5: Optimism-guided Data Augmentation + Final Evaluation")


if __name__ == "__main__":
    main()
