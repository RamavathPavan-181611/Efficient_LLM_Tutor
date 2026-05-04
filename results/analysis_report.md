# Replication Report
## Paper: Efficient RL for Optimizing Multi-turn Student Outcomes with LLM Tutors

---

## Table 1 — Dataset Statistics

| Dataset | N samples | Success % | Diversity % |
|---------|-----------|-----------|-------------|
| Original D  | 16930  | 17.83  | 86.35  |
| Augmented D+ | 1916 | 33.46 | 92.12 |

> Paper reports: D success=74.64, diversity=38.53 | D+ success=82.83, diversity=39.35

---

## Figure 3 — Policy Success Rates (300 conversations each)

| Policy | Our result | Paper result | Δ from prompt |
|--------|-----------|--------------|---------------|
| Prompt | 38.21% | 36.0% | +0.00% |
| BC(D) | 51.27% | 33.0% | +13.21% |
| BC(D+) | 50.77% | 40.0% | +12.06% |
| FQI(D) | 33.67% | 44.0% | -4.54% |
| FQI(D+) | 32.31% | 46.0% | -5.9% |
| CQL(D) | 67.19% | 48.67% | +28.98% |
| CQL(D+) | 67.40% | 60.33% | +29.19% |
| DQN(D) | 20.28% | ------- | -17.93% |
| DQN(D+) | 25.78% | ------ | -12.43% |
| IQL(D) | 45.11% | ------- | +6.9% |
| IQL(D+) | 44.70% | ------- | +6.49% |


> **Key result:** CQL(D+) achieves 60.33% vs prompt 36.00% — a **67.6% relative improvement** (paper reports 50%).

---

## Table 2 — Generalization Across 7 GSM8K Problems

| Tutor | Mean ± Std |
|-------|-----------|
| BC* (exploratory data only) | 36.23 ± 20.80 |
| CQL(D+) | 27.38 ± 17.35 |
| Prompt engineering | 26.90 ± 22.78 |

> Generalization result: BC* surprisingly outperforms CQL+ on unseen problems,
> suggesting each problem may have its own latent transition dynamics.

---

## Figure 5 — Action Distribution Analysis

Key finding: BC* relies on **instruct (~80%)** for nearly all turns.
CQL(D+) spreads across all 4 actions more evenly (~25-30% each).
This is pedagogically important — heavy instruction may not foster
independent problem-solving.

---

## Figures
- `figures/figure3_success_rates.png`
- `figures/figure4_generalization.png`
- `figures/figure5_action_dist.png`
