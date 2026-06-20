# Perception Risk for Path Planning in Autonomous Rover Navigation

Reproduction package for the paper *"Perception Risk for Path Planning in Autonomous Rover Navigation"*.

This pipeline implements a closed-loop **perception–planning–adaptation** architecture that compares Q-Learning (off-policy) and SARSA (on-policy) under a fog-of-war navigation scenario. The agent learns to navigate a real RGB-D rover sequence (Intel RealSense D435i) while actively triggering directional scans whenever it approaches unknown terrain. A directional **perception-hazard tensor** $\xi$ shapes the reward function so that the policy learns to balance progress toward the goal against the risk of entering poorly-observed or hazardous regions.

> **Repository name:** The GitHub repository should be renamed to match the paper title, e.g. `perception-hazard-for-path-planning-in-autonomous-rover-navigation`. Renaming is done in GitHub → Settings → Repository name. The local clone URL will need to be updated accordingly after renaming.

---

## Table of Contents

1. [Research Questions](#research-questions)
2. [Repository Structure](#repository-structure)
3. [File Reference](#file-reference)
4. [Model Description](#model-description)
5. [Algorithm](#algorithm)
6. [Installation](#installation)
7. [Dataset Setup](#dataset-setup)
8. [Running the Pipeline](#running-the-pipeline)
9. [Running Experiments](#running-experiments)
10. [Outputs](#outputs)
11. [Results](#results)
12. [Configuration Reference](#configuration-reference)

---

## Research Questions

The code is structured to reproduce evidence for the three research questions investigated in the paper:

- **RQ1** — Is there a trade-off between perception quality (scan depth $\tau_{\text{scan}}$) and path optimality (cumulative reward)?
  → Swept via `run_experiments.py` over $\tau_{\text{scan}} \in \{5, 10, 15, 20\}$ cells.

- **RQ2** — Is there a trade-off between diversity of state visitations (Perception Entropy $H_{\text{perc}}$) and path optimality?
  → Measured per episode in `*_History.csv` and plotted by `generate_paper_figures.py`.

- **RQ3** — Is safety sensitive to perception uncertainty?
  → Safety margin (average distance to obstacles along the greedy path) is recorded per episode and tested with Mann-Whitney U in `hypothesis_tests.py`.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                                        # All source code
│   ├── main.py                                 # Entry point — mission orchestration
│   │
│   ├── rl_environment.py                       # Safety-aware MDP (GridEnv)
│   ├── rl_agent.py                             # Tabular RL agent (Q-Learning & SARSA)
│   ├── rl_trainer.py                           # Online training loop with in-episode scanning
│   │
│   ├── grid_mapper.py                          # Depth image → 2-D occupancy grid projection
│   ├── confidence_mapper.py                    # Per-cell confidence accumulation
│   ├── unbounded_metrics.py                    # Perception-hazard tensor ξ and coverage V_φ
│   ├── view_cones.py                           # 8-directional view cone utilities
│   ├── adaptive_perception.py                  # Adaptive perception model (lookup table)
│   │
│   ├── image_loader.py                         # RGB-D pair discovery and loading
│   ├── pose_loader.py                          # Ground-truth pose loading (TUM format)
│   │
│   ├── run_experiments.py                      # Automated τ_scan sweep (RQ1, RQ3)
│   ├── hyperparameter_sweep.py                 # Grid search over α, γ, θ
│   ├── compare_safety.py                       # Q-Learning safety margin comparison
│   ├── compare_safety_sarsa.py                 # SARSA safety margin comparison
│   ├── hypothesis_tests.py                     # Mann-Whitney U tests (RQ3)
│   ├── plot_results.py                         # Per-episode metric visualisations
│   ├── plot_safety_comparison.py               # Safety margin box plots
│   ├── plot_comparison_qlearning_vs_sarsa.py   # Algorithm comparison figures
│   ├── generate_paper_figures.py               # Publication-quality figures (RQ1, RQ2)
│   │
│   └── hypothesis_test_results.md              # Statistical test results and interpretation
│
└── results/                                    # Generated outputs (created at runtime)
    ├── mission_summary.csv                     # Aggregate run metrics (all algorithms × scan depths)
    ├── Q-Learning/Scan_{5,10,15,20}/           # Per-run history + Q-table + GIF
    ├── SARSA/Scan_{5,10,15,20}/                # Per-run history + Q-table + GIF
    ├── pdfs/                                   # Pre-built result PDFs
    └── figures/                                # Cross-algorithm comparison plots
```

---

## File Reference

### Data ingestion

| File | Role |
|---|---|
| `image_loader.py` | Discovers paired RGB + depth frames in `ROVER_data/realsense_D435i/`. Matches frames by TUM-format timestamps (`rgb.txt`, `depth.txt`) with a 30 ms tolerance; falls back to sorted-filename pairing if timestamps are absent. Provides `load_rgb()` and `load_depth()` returning float32 arrays. |
| `pose_loader.py` | Loads ground-truth camera poses from `ROVER_data/groundtruth.txt`. Estimates heading from consecutive positions via `atan2(Δy, Δx)`. `transform_points()` converts camera-frame 3-D points into world-frame coordinates using the estimated yaw. |

### Mapping layer

| File | Role |
|---|---|
| `grid_mapper.py` | Projects a depth image into a top-down 2-D occupancy grid. Uses a hit-ratio threshold (0.2) against the relevant obstacle-height band (0.8 m – 2.5 m) to avoid floor clutter. `accumulate_with_confidence()` merges new frames into the running global map using a per-cell confidence score that decides which observation wins. |
| `confidence_mapper.py` | `ConfidenceAccumulator` maintains per-cell running statistics (observation count, depth mean / variance) across all frames. The final confidence combines a **consistency** score (low variance = high confidence) and an **observation** score (more views = higher confidence), both in [0, 1]. |
| `view_cones.py` | Defines the 8-directional coordinate system (N = +row, E = +col) and the `STEP_TO_DIR_IDX` mapping from grid steps `(dr, dc)` to direction indices 0–7. Provides `get_view_cones()` to return left / head / right cone names. |
| `unbounded_metrics.py` | The core hazard engine.<br>• `compute_directional_distances()` — vectorised sweep returning, for every free cell, the straight-line distance to the nearest obstacle / unknown in each of 8 directions (`dist[r, c, dir]`) and the type of cell that stopped the ray (`stop_type[r, c, dir]`).<br>• `compute_cone_exploration()` — convolution-based computation of how many *known* cells lie within a θ-degree cone (the visibility fraction).<br>• `compute_unbounded_risk()` — applies the formula $\xi = \zeta \cdot \varphi(d) \cdot V_\varphi$ across the whole grid in one vectorised call. Result: a `(rows, cols, 8)` hazard tensor saved in memory. |

### RL core

| File | Role |
|---|---|
| `rl_environment.py` | `GridEnv` — the safety-aware MDP. State is a 6-tuple $(\rho, c, \varphi, b_h, b_l, b_r)$ that is **goal-blind**: the agent must discover the +1000 reward through exploration rather than by reading the goal coordinates. Reward cases: +1000 for goal, −20 for collision (obstacle / unknown / out-of-bounds), and $\Pi_t - C_t - \Xi_t$ for a normal step. `extract_greedy_path()` traces the learned policy deterministically with Euclidean tie-breaking toward the goal. |
| `rl_agent.py` | `RLAgent` — tabular RL backed by a `defaultdict` Q-table keyed by `(state, action)`, every entry starting at 0.0. Implements ε-greedy `select_action()`, `update_sarsa()` (on-policy: uses the next action that will actually be taken), `update_qlearning()` (off-policy: uses the greedy max over next actions), Shannon entropy of state visitations, and per-(state, action) bookkeeping for CSV export. |
| `rl_trainer.py` | `train_online()` — drives the per-episode loop. Persists the Q-table across all calls so knowledge accumulates as the map grows. Crucially, the in-episode scan callback is wired in here: *before* every move it checks whether the agent is about to step into unknown terrain closer than $\tau_{\text{scan}}$ cells, and if so triggers a scan that rebuilds the entire hazard tensor *before* the step executes. Returns the greedy path, best-episode path, the updated ε value, convergence stats, and a full per-episode `history` dict. |

### Active perception

| File | Role |
|---|---|
| `adaptive_perception.py` | `PerceptionModel` — a stateful lookup table that learns to predict `(distance, frame_count)` measurements per `(source, target, direction)` triplet. Implements measure → predict → execute → update → adapt with error-adaptive learning rate λ and variance-adaptive planning horizon ψ. Reserved for the unbounded-risk variant of the pipeline. |

### Main pipeline

| File | Role |
|---|---|
| `main.py` | Top-level entry point. `run_rl_pipeline()` loads pairs and poses, computes world bounds, pre-builds the **oracle map** (full ground-truth map from all dataset frames, used as the source of truth that scans reveal from), pre-computes the perception coverage mask, then for each mode in `RUN_MODE` calls `run_experiment()`. The latter initialises an empty agent-side map, performs an initial 360° scan, invokes `train_online()` for all `RL_EPISODES` episodes (with a `_scan_callback` closure that rebuilds the hazard tensor on every triggered scan), then physically follows the greedy path and records mission metrics. Exports per-run CSVs, mission GIF, and triggers the comparison plot generators. |

### Experiment automation

| File | Role |
|---|---|
| `run_experiments.py` | Loops over `THRESHOLDS = [5, 10, 15, 20, 25]`, rewrites `SCAN_DEPTH_THRESHOLD` in `main.py` via regex, and runs `main.py` as a subprocess with `BATCH_MODE=1`. After all runs pass, triggers safety comparison, algorithm comparison plots, and paper figure generation. |
| `hyperparameter_sweep.py` | Grid search over $\alpha \times \gamma \times \theta$ (3 × 3 × 3 = 27 combinations) for both algorithms. Saves a results table PNG and a 27-panel map figure. |

### Analysis and plotting

| File | Role |
|---|---|
| `plot_results.py` | Reads per-scan History CSVs and plots 7 metrics (reward, path length, TD error, goal success, safety margin, tension, entropy) side-by-side Q-Learning vs SARSA. Auto-called by `main.py` after each run. |
| `plot_safety_comparison.py`, `compare_safety.py`, `compare_safety_sarsa.py` | Cross-scan safety comparison: aggregate safety margins across all scan depths into figures and CSVs. |
| `plot_comparison_qlearning_vs_sarsa.py` | Head-to-head Q-Learning vs SARSA plots across scan depths. |
| `generate_paper_figures.py` | Generates final publication figures (RQ1, RQ2) from the results data. |
| `hypothesis_tests.py` | Mann-Whitney U and Spearman correlation tests with Bonferroni correction. Outputs are written to `src/hypothesis_test_results.md`. |

---

## Model Description

### Notation

| Paper symbol | Code parameter | Meaning |
|---|---|---|
| $V$ | `view_cones.py` | View cone — one of 8 × 60° directional sectors (N, NE, E, SE, S, SW, W, NW) |
| $\rho_{\text{perc}}$ | `PERCEPTION_RADIUS_M` | Maximum sensing range of the rover (metres) |
| $\tau_{\text{scan}}$ | `SCAN_DEPTH_THRESHOLD` | Scan trigger distance (cells); also extends the effective sensing depth |
| $\theta$ | `THETA_DEGREE` | Effective field-of-view of the vision cone (degrees) |
| $V_\varphi$ | `1 − vframes_current` | Rate of **invisible** (unknown) cells inside a vision cone. Higher $V_\varphi$ = more unmapped area = higher risk. `vframes_current` stores the complementary known fraction. |
| $\zeta_{\text{obs}}$ | `HAZARD_OBSTACLE` | Hazard multiplier for confirmed obstacle cells (occ = 1); default **10** |
| $\zeta_{\text{unk}}$ | `HAZARD_UNKNOWN` | Hazard multiplier for unknown cells (occ = −1); default **1** |
| $d$ | `directional_distances` | Per-cell, per-direction distance to the nearest hazard, shape `(rows, cols, 8)` |
| $d_{\text{var}}$ | `VAR_DIST` | Distance softening constant in $\varphi(d) = d_{\text{var}} / (d + d_{\text{var}})$; default **1.0** cells |
| $\xi$ | `directional_risk` | Per-cell, per-direction perception-hazard tensor, shape `(rows, cols, 8)` |
| $\Xi_t$ | `risk[nr, nc, action]` | Hazard at the **destination** cell looking in the action direction — the value subtracted from the reward |
| $\varepsilon$ | `epsilon` | ε-greedy exploration rate (decays from 0.99 → 0.1 over 500 episodes) |
| $H_{\text{perc}}$ | `hist['entropy']` | Perception Entropy — Shannon entropy of state visitations |

---

### Occupancy Grid

The environment is a 2-D occupancy grid with resolution $r = 0.05$ m / cell:

- **0** — free (traversable)
- **1** — occupied (obstacle)
- **−1** — unknown (not yet revealed)

The agent-side grid starts entirely unknown and is revealed incrementally through directional scans against a hidden **oracle map** pre-computed from all dataset frames at startup. This produces a fog-of-war simulation where the agent must trade exploration of unknown terrain against the cost of additional scans.

---

### Perception-Hazard Tensor $\xi$

For each cell $(\rho, c)$ and direction $\varphi$, the directional hazard in the reward function is:

$$\xi(\rho, c, \varphi) = \zeta(\rho, c, \varphi) \cdot \underbrace{\frac{d_{\text{var}}}{d(\rho, c, \varphi) + d_{\text{var}}}}_{\varphi(d)} \cdot V_\varphi(\rho, c, \varphi)$$

with

$$\zeta = \begin{cases} \zeta_{\text{obs}} = 10 & \text{if the ray hits an obstacle} \\ \zeta_{\text{unk}} = 1 & \text{if the ray hits unknown terrain} \\ 1 & \text{if the ray reaches the boundary unobstructed} \end{cases}$$

and $d_{\text{var}} = 1.0$ cell is a distance softening constant that prevents division by zero and tunes the falloff: hazard halves roughly every cell of separation, very close blockers saturate near 1.0, and distant blockers fade smoothly to zero.

$V_\varphi$ is the **rate of invisible (unknown) cells** within a $\theta$-degree cone centred on $\varphi$, computed by 2-D convolution: known-cell mask × cone kernel → known fraction → $V_\varphi = 1 -$ known fraction. Obstacle cells count as observed; only unknown (−1) cells inflate $V_\varphi$.

---

### Reward Function

For action $a_t$ in state $s_t$ leading to $s_{t+1}$:

$$R_t = \begin{cases} \Omega_t = +1000 & \text{if } s_{t+1} = G_{\text{cell}} \\ -20 & \text{if } s_{t+1} \in \{\text{obstacle}, \text{unknown}, \text{out-of-bounds}\} \\ \Pi_t - C_t - \Xi_t & \text{otherwise} \end{cases}$$

| Term | Value | Description |
|---|---|---|
| $\Omega_t$ | +1000 | Goal bonus |
| $\Pi_t$ | $\omega \cdot (d_{\text{old}} - d_{\text{new}})$ | Progress carrot; $\omega = 4$ |
| $C_t$ | 1.0 | Movement cost (cardinal and diagonal) |
| $\Xi_t$ | $\xi(\rho_{t+1}, c_{t+1}, a_t)$ | Hazard at the destination cell looking in the chosen direction |

---

### State Representation (Goal-Blind)

$$s = (\rho,\; c,\; \varphi,\; b_h,\; b_l,\; b_r)$$

with $(\rho, c)$ the current cell, $\varphi \in \{0, \ldots, 7\}$ the heading, and $b_h, b_l, b_r$ the binned distances to the nearest hazard in the ahead / left / right cones. Distance bins: [0, 1), [1, 2), [2, 4), [4, 8), ≥ 8 cells. The goal coordinates are deliberately **not** included — the agent learns a general safety policy rather than a GPS trajectory.

---

### Perception Entropy $H_{\text{perc}}$

$$H_{\text{perc}} = -\sum_s P(s) \log_2 P(s), \quad P(s) = \frac{\text{visit\_count}(s)}{\text{total\_visits}}$$

Higher entropy = broader state exploration. The paper reports statistically significant positive Spearman correlations between $H_{\text{perc}}$ and cumulative reward across all scan depths (Bonferroni-corrected $p_{\text{adj}} < 0.001$).

---

### RL Update Rules

**Q-Learning** (off-policy, risk-seeking):

$$Q(s, a) \leftarrow Q(s, a) + \alpha \bigl[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \bigr]$$

**SARSA** (on-policy, conservative):

$$Q(s, a) \leftarrow Q(s, a) + \alpha \bigl[ r + \gamma\, Q(s', a') - Q(s, a) \bigr]$$

$a'$ in SARSA is the action actually selected by ε-greedy at $s'$, which makes SARSA sensitive to exploration penalties and therefore more cautious near hazards.

---

## Algorithm

This section explains the full closed-loop pipeline, including subtleties that are easy to misread from a quick look at the code.

### 1 — Scan triggers in a single direction

Before every move the trainer checks the cell the agent is about to enter:

$$\text{trigger scan} \iff \bigl(d(s_t, a_t) < \tau_{\text{scan}}\bigr) \;\land\; \bigl(\mathcal{M}(c_{\text{stop}}) = -1\bigr)$$

That is: the agent only scans when (i) the chosen direction's ray ends close (within $\tau_{\text{scan}}$ cells), AND (ii) the ray was stopped by **unknown** terrain (not by an obstacle or boundary). A scan reveals a **single 60° cone** in that one direction — not all 8 directions — with effective radius $\rho_{\text{perc}} + \tau_{\text{scan}} \cdot r$. The newly revealed cells (free, obstacle, or remaining unknown beyond the cone) are written into the agent's accumulated occupancy map.

### 2 — Hazard tensor is rebuilt over the whole map

Even though only one cone of cells changed, the hazard tensor $\xi$ is recomputed **from scratch for every free cell in the map and all 8 directions**. This is necessary because directional distances cascade: a single newly revealed obstacle can change the ray-stop result for many neighbouring cells looking in many directions.

```
Before scan:        After scan reveals wall:
. . . . ?           . . . . #
. . A . ?     →     . . A . #
. . . . ?           . . . . #

cell A looking East:  d ≈ 4, stop = unknown   →   d ≈ 4, stop = obstacle  (ζ jumps 1 → 10)
cell B (left of A) looking East: also re-evaluated, also affected
```

The same cascade works in the **other direction**: if a scan reveals open space instead of a wall, formerly elevated hazard values drop accordingly. This is the active-perception incentive — scanning can *open up* previously avoided paths once they turn out to be safe.

### 3 — Q-table updates only for the cell that was actually visited

The hazard rebuild touches the **entire map** (potentially hundreds of cells × 8 directions). The Q-table update, in contrast, touches a **single (state, action) entry** — the one corresponding to the move just executed:

```python
# rl_environment.py — reward computed using the freshly rebuilt ξ
reward = -(d_step + risk[next_cell, action]) + progress_to_goal

# rl_agent.py — single Q-table entry updated
Q[(s, action)] += α · (r + γ · max_a' Q[(s', a')] − Q[(s, action)])
```

So after a scan, the rest of the map carries up-to-date hazard values in the *reward signal*, but most cells' Q-values are still **stale** until the agent physically visits them again. This is the key dynamic of the system.

### 4 — Why ε matters: rediscovering stale Q-values

Suppose before a scan the agent had concluded that going East is bad ($Q(s, \text{East}) = 10$, low). Then a scan reveals open space to the East and the hazard drops sharply. The reward signal would now reward going East — but the agent will only see this if it actually *tries* East again. With a purely greedy policy ($\varepsilon = 0$) it would never go East, and the stale low Q-value would remain forever.

This is exactly why ε-greedy never reaches zero:

```python
RL_EPS_START = 0.99    # start: ~all-random
RL_EPS_END   = 0.10    # floor:  always at least 10% random
```

The 10% random exploration floor guarantees that, on average, every direction is occasionally retried. When a previously-bad direction has had its hazard dropped by a scan, that occasional random attempt now returns a higher-than-expected reward and the Q-value starts rising. After enough random visits, it can overtake the greedy choice and the agent switches paths.

**Timing matters.** Early scans (when $\varepsilon$ is still close to 0.99) get re-explored heavily and their corrected hazard values propagate fast through the Q-table. Late scans (when $\varepsilon$ is at the 0.10 floor) propagate slowly — the agent may not have enough remaining random attempts to fully exploit the new information before the 500 episodes run out.

### 5 — One step inside an episode, end-to-end

```
Step k of episode n
│
├─ 1. Agent picks an action a_t via ε-greedy on Q(s_t, ·)
│       — with probability ε: random valid action
│       — otherwise:           argmax_a Q(s_t, a)
│
├─ 2. Scan check (before moving)
│       if d(s_t, a_t) < τ_scan AND stop_type(s_t, a_t) = -1:
│            reveal 60° cone in direction a_t  → updates accumulated_occ
│            recompute directional distances    (whole map, 8 dirs)
│            recompute hazard tensor ξ          (whole map, 8 dirs)
│            env.update_map(new occ, new ξ, new distances)
│
├─ 3. Execute move:  s_{t+1} = s_t + a_t
│       — collision (obstacle/unknown/out-of-bounds): r = -20, stay in place
│       — goal:                                       r = +1000, episode ends
│       — normal:                                     r = progress − cost − ξ[s_{t+1}, a_t]
│
├─ 4. Q-table update for the single (s_t, a_t) entry
│       Q-Learning: Q(s,a) += α·(r + γ·max Q(s',·) − Q(s,a))
│       SARSA:      Q(s,a) += α·(r + γ·Q(s', a') − Q(s,a))   with a' = next ε-greedy choice
│
└─ Loop to step k+1 (unless done or step ≥ MAX_RL_STEPS = 400)
```

`MAX_RL_STEPS = 400` caps each episode; reaching the goal ends it early. Hitting an obstacle / unknown applies the −20 penalty but does **not** end the episode — the agent stays put and keeps trying.

### 6 — ε decays once across the whole mission

ε is **never reset** by a scan or by an episode boundary. It is a single value passed through `train_online()` and updated multiplicatively:

```python
ε ← max(ε_min, ε · ε_decay)        # after every episode
ε_decay  =  (ε_min / ε_0)^(1/N)    # so ε reaches ε_min at episode N=500
```

Approximate trajectory:

| Episode | ε     |
|---------|-------|
|   1     | 0.99  |
| 100     | 0.64  |
| 250     | 0.32  |
| 400     | 0.16  |
| 500     | 0.10  |

The Q-table also persists across all episodes and scans — knowledge is accumulated, not reset.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset Setup

The pipeline uses RGB-D sequences from the **ROVER dataset** (Intel RealSense D435i, outdoor ground rover).
Dataset homepage: https://iis-esslingen.github.io/rover/
No registration required — direct public download.

### 1. Download a sequence

Pick any sequence from the download page:
https://iis-esslingen.github.io/rover/pages/download/

Any sequence works. As a concrete example (~15–19 GB):

```
https://fdm.hs-esslingen.de/schmidt2025rover/garden_small_2023-08-18.zip
```

All available sequences are mirrored on HuggingFace:
https://huggingface.co/datasets/iis-esslingen/ROVER

### 2. Extract and place under `ROVER_data/`

Each sequence contains an `intelrealsense_D435i/` folder. Rename it to `realsense_D435i` and place everything at the repository root:

```
ROVER_data/
├── realsense_D435i/          # renamed from intelrealsense_D435i/
│   ├── rgb/<timestamp>.png
│   ├── depth/<timestamp>.png # 16-bit PNG, depth = pixel_value × 10^{-3} m
│   ├── rgb.txt               # timestamp list for RGB frames
│   └── depth.txt             # timestamp list for depth frames
└── groundtruth.txt           # TUM format: timestamp tx ty tz qx qy qz qw
```

`ROVER_data/` is gitignored — the dataset is never committed.

### 3. Paths in `src/main.py`

```python
DATASET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "ROVER_data", "realsense_D435i")
GT_FILE     = os.path.join(os.path.dirname(SCRIPT_DIR), "ROVER_data", "groundtruth.txt")
```

### Camera and pipeline parameters

| Parameter | Value |
|---|---|
| Camera | Intel RealSense D435i |
| fx = fy | 610 px |
| cx, cy | 320, 240 px |
| Depth encoding | 16-bit PNG, scale 10^{-3} m/unit |
| Frames used | 40–230 (1-indexed; frames 1–39 discarded for pose convergence) |
| Obstacle height band | 0.8 m – 2.5 m |
| Hit-ratio threshold | 0.2 |
| Timestamp match tolerance | 30 ms |

---

## Running the Pipeline

```bash
cd src
python main.py
```

Batch mode (no interactive wait, suitable for scripting):

```bash
BATCH_MODE=1 python main.py          # Linux / macOS
set BATCH_MODE=1 && python main.py   # Windows CMD
```

---

## Running Experiments

### RQ1 & RQ3 — Scan-depth sweep

Sweeps $\tau_{\text{scan}} \in \{5, 10, 15, 20\}$ cells, then auto-generates all comparison figures and runs statistical tests:

```bash
cd src
python run_experiments.py
```

### RQ2 — Perception Entropy analysis

Generated automatically at the end of `run_experiments.py`. To regenerate figures only:

```bash
cd src
python generate_paper_figures.py
```

### RQ3 — Mann-Whitney U tests

```bash
cd src
python hypothesis_tests.py
```

Results and interpretation: [`src/hypothesis_test_results.md`](src/hypothesis_test_results.md).

### Hyperparameter grid search

```bash
cd src
python hyperparameter_sweep.py
```

---

## Outputs

Outputs are written to `results/` at the repository root, organised by algorithm and scan depth.

**Per-run outputs** → `results/<Algorithm>/Scan_<δ>/`

| File | Description |
|---|---|
| `<ALG>_Th<θ>_Scan<δ>_History.csv` | Per-episode: cumulative reward, path length, mean TD error, goal-reached flag, average safety margin, average tension, $H_{\text{perc}}$ |
| `<ALG>_Qtable_Scan<δ>.csv` | Full Q-table: $(\rho, c, \varphi, b_h, b_l, b_r,$ action, Q-value, visit count, tension$)$ |
| `mission_<ALG>_Th<θ>_Scan<δ>.gif` | Animated occupancy map showing fog-of-war revelation during the mission |

**Aggregate outputs** → `results/`

| File | Description |
|---|---|
| `mission_summary.csv` | One row per run: convergence episode, final greedy reward, path length, safety margin, total scans, $H_{\text{perc}}$ |

**Cross-algorithm figures** → `results/figures/`

| File | Description |
|---|---|
| `map_comparison_Th<θ>_Scan<δ>.png` | Side-by-side final maps for Q-Learning vs SARSA |
| `path_overlay_Th<θ>_Scan<δ>.png` | Both agents' greedy paths overlaid on the accumulated occupancy map |
| `convergence_diagnostics_Th<θ>_Scan<δ>.png` | Episode rewards, path lengths, and TD error magnitude over training |
| `safety_comparison_qlearning.csv`, `safety_comparison_sarsa.csv` | Safety margins by $\tau_{\text{scan}}$ (RQ3 input) |
| Paper figures | Publication-quality plots from `generate_paper_figures.py` |

---

## Results

Pre-built result PDFs are stored in [`results/pdfs/`](results/pdfs/) and committed to the repository — no re-run required to inspect them.

| File | Contents |
|---|---|
| [`results/pdfs/results.pdf`](results/pdfs/results.pdf) | Full report: statistical tables, CSV previews around each convergence point, all 18 figures |
| [`results/pdfs/results_v2.pdf`](results/pdfs/results_v2.pdf) | Compact report: statistical tables + 6 key publication figures |
| [`results/pdfs/Box_Plot.pdf`](results/pdfs/Box_Plot.pdf) | Safety margin distributions per scan depth (Q-Learning vs SARSA) |
| [`results/pdfs/Safety_Margins.pdf`](results/pdfs/Safety_Margins.pdf) | Safety margin evolution across 500 training episodes |
| [`results/pdfs/Fig_rq1_1_QLEARNING.pdf`](results/pdfs/Fig_rq1_1_QLEARNING.pdf) | RQ1 — Q-Learning cumulative reward convergence across scan depths |
| [`results/pdfs/Fig_rq1_1_SARSA.pdf`](results/pdfs/Fig_rq1_1_SARSA.pdf) | RQ1 — SARSA cumulative reward convergence across scan depths |
| [`results/pdfs/Fig_rq1_2_QLEARNING.pdf`](results/pdfs/Fig_rq1_2_QLEARNING.pdf) | RQ2 — Q-Learning reward vs Perception Entropy (Spearman ρ) |
| [`results/pdfs/Fig_rq1_2_SARSA.pdf`](results/pdfs/Fig_rq1_2_SARSA.pdf) | RQ2 — SARSA reward vs Perception Entropy (Spearman ρ) |

Statistical test outputs (Mann-Whitney U, Spearman correlation with Bonferroni correction) are in [`src/hypothesis_test_results.md`](src/hypothesis_test_results.md).

---

## Configuration Reference

All parameters are set at the top of `src/main.py`.

| Code parameter | Paper symbol | Default | Description |
|---|---|---|---|
| `START_FRAME` | — | 40 | First dataset frame (1-indexed; frames 1–39 discarded for pose convergence) |
| `END_FRAME` | — | 230 | Last dataset frame |
| `GOAL_FRAME` | — | 60 | Frame whose pose defines the goal cell |
| `RL_EPISODES` | $N$ | 500 | Training episodes per mission |
| `RL_ALPHA` | $\alpha$ | 0.4 | Learning rate |
| `RL_GAMMA` | $\gamma$ | 0.95 | Discount factor |
| `MAX_RL_STEPS` | $T_{\max}$ | 400 | Maximum steps per episode |
| `RL_EPS_START` | $\varepsilon_0$ | 0.99 | Initial exploration rate |
| `RL_EPS_END` | $\varepsilon_{\min}$ | 0.10 | Terminal exploration rate |
| `RL_EPS_DECAY` | — | `"auto"` | Per-episode decay; `"auto"` computes decay so ε reaches $\varepsilon_{\min}$ at episode $N$ |
| `REWARD_GOAL` | $\Omega_t$ | 1000.0 | Terminal goal reward |
| `COST_CARDINAL` | $C_t$ | 1.0 | Movement cost (cardinal) |
| `COST_DIAGONAL` | $C_t$ | 1.0 | Movement cost (diagonal) |
| `PROGRESS_REWARD_SCALE` | $\omega$ | 4 | Scale factor for the progress carrot $\Pi_t$ |
| `VAR_DIST` | $d_{\text{var}}$ | 1.0 | Distance softening constant in the hazard formula |
| `THETA_DEGREE` | $\theta$ | 60.0 | FOV cone angle for $V_\varphi$ (degrees) |
| `HAZARD_OBSTACLE` | $\zeta_{\text{obs}}$ | 10.0 | Hazard multiplier for confirmed obstacles |
| `HAZARD_UNKNOWN` | $\zeta_{\text{unk}}$ | 1.0 | Hazard multiplier for unknown cells |
| `SCAN_DEPTH_THRESHOLD` | $\tau_{\text{scan}}$ | 5 | Scan trigger distance (cells); swept over {5, 10, 15, 20} for RQ1/RQ3 |
| `PERCEPTION_RADIUS_M` | $\rho_{\text{perc}}$ | 1.0 | Base vision-cone radius (metres) |
| `PERCEPTION_FOV_DEG` | $\varphi_{\text{fov}}$ | 60.0 | Scan FOV (degrees) |
| `REWARD_BLIND` | — | 0.0 | Penalty for stepping outside dataset coverage (set 0 to disable) |
| `MAX_MISSION_STEPS` | — | 300 | Maximum physical steps before mission failure |
| `RUN_MODE` | — | `"both"` | `"both"`, `"qlearning"`, or `"sarsa"` |
| `OUTPUT_DIR` | — | `../results` | Output directory (relative to `src/`) |
