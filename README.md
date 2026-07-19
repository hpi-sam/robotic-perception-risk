# Perception Risk for Path Planning in Autonomous Rover Navigation

This repository contains the **reproduction package** for the paper:

> **Perception Risk for Path Planning in Autonomous Rover Navigation**

It implements a closed-loop **perception–planning–adaptation** architecture for autonomous rover navigation under perception uncertainty, using model-free reinforcement learning (SARSA and Q-Learning) on real RGB-D rover data from an Intel RealSense D435i camera. [file:1]

The code exposes a **perception-risk score** as an explicit runtime decision variable, combining geometric risk (distance to obstacles/unknowns) and perception confidence (coverage of unknown cells) into a unified scalar that shapes the reward during policy training. [file:1]

> **Repository name recommendation**  
> For consistency with the paper, rename the GitHub repo to something like `robotic-perception-risk` under **Settings → Repository name**, then update your local remote URL.

---

## Table of Contents

1. [Research Questions](#research-questions)  
2. [Conceptual Overview](#conceptual-overview)  
3. [Repository Structure](#repository-structure)  
4. [Notation and Perception Model](#notation-and-perception-model)  
5. [Reinforcement Learning Setup](#reinforcement-learning-setup)  
6. [Algorithmic Pipeline](#algorithmic-pipeline)  
7. [Installation](#installation)  
8. [Dataset Setup](#dataset-setup)  
9. [Running the Pipeline](#running-the-pipeline)  
10. [Running Experiments](#running-experiments)  
11. [Outputs](#outputs)  
12. [Results and Appendix](#results-and-appendix)  
13. [Configuration Reference](#configuration-reference)  
14. [Reproducibility Notes](#reproducibility-notes)

---

## Research Questions

The reproduction package is organised to provide evidence for the three research questions studied in the paper. [file:1]

- **RQ1 — Perception quality vs cumulative reward**  
  *Does increasing the scan-depth threshold $\tau_{\text{scan}}$ improve cumulative reward?*  
  This asks under which conditions perception quality (scan depth) improves the agent’s ability to find effective trajectories under uncertainty. [file:1]

- **RQ2 — Perception Entropy vs path optimality**  
  *Does Perception Entropy, as a measure of visitation diversity, affect navigation performance during training?*  
  This investigates whether broader state coverage during learning reliably improves path optimality under perception uncertainty. [file:1]

- **RQ3 — Safety saturation under increased sensing**  
  *Does increasing the scan-depth threshold $\tau_{\text{scan}}$ improve the average obstacle-clearance distance $\bar{d}_{\text{clear}}$ during training, and is there a saturation point beyond which additional scan depth yields diminishing returns?* [file:1]

---

## Conceptual Overview

Autonomous rovers in partially observable environments must jointly answer:

- “Where should I go next?” (planning)  
- “How much do I trust what I see?” (perception) [file:1]

Classic motion planners optimise geometric objectives (path length, clearance) over a static environment model, often assuming perception provides a reliable map. In reality, sensor noise, occlusions, and unknown regions introduce perception uncertainty that can render formally feasible trajectories unsafe or inefficient. [file:1]

This work:

- Introduces a **perception-aware decision layer** that ranks candidate trajectories using a unified scalar score combining:
  - Obstacle proximity.  
  - Perception coverage (fraction of unknown cells in a view cone).  
  - Hazard multipliers for obstacles vs unknown regions. [file:1]
- Trains **tabular RL agents** (SARSA and Q-Learning) with a reward function shaped by this perception-risk score, enabling policies that trade off:
  - Navigation progress toward the goal.  
  - Safety margins and perception certainty. [file:1]
- Studies **Perception Entropy** as a measure of visitation diversity, analysing how exploration behaviour affects navigation performance before and after policy convergence. [file:1]

---

## Repository Structure

All source code is under `src/`, and all generated artefacts are under `results/`. [file:1]

```text
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                                        # All source code
│   ├── main.py                                 # Entry point — mission orchestration
│   │
│   ├── rl_environment.py                       # Safety-aware grid MDP (GridEnv)
│   ├── rl_agent.py                             # Tabular RL agent (Q-Learning & SARSA)
│   ├── rl_trainer.py                           # Online training loop with in-episode scanning
│   │
│   ├── grid_mapper.py                          # Depth image → occupancy grid projection
│   ├── confidence_mapper.py                    # Per-cell confidence accumulation
│   ├── unbounded_metrics.py                    # Perception-risk score ξ_perc and coverage V_φ
│   ├── view_cones.py                           # 8-directional view cone utilities
│   ├── adaptive_perception.py                  # Adaptive perception model (unbounded-risk variant)
│   │
│   ├── image_loader.py                         # RGB-D frame discovery and loading
│   ├── pose_loader.py                          # Ground-truth pose loading (TUM format)
│   │
│   ├── run_experiments.py                      # Automated τ_scan sweep (RQ1, RQ3)
│   ├── hyperparameter_sweep.py                 # Grid search over α, γ, θ
│   ├── compare_safety.py                       # Q-Learning safety margin comparison
│   ├── compare_safety_sarsa.py                 # SARSA safety margin comparison
│   ├── hypothesis_tests.py                     # Mann–Whitney U tests on safety margins (prints to stdout)
│   │
│   ├── plot_results.py                         # Per-episode metric visualisations
│   ├── plot_safety_comparison.py               # Safety margin box plots
│   ├── plot_comparison_qlearning_vs_sarsa.py   # Algorithm comparison figures
│   ├── generate_paper_figures.py               # Publication-quality figures (RQ1, RQ2)
│   │
│   └── hypothesis_test_results.md              # Statistical test results and interpretation
│
└── results/                                    # Generated outputs (created at runtime)
    ├── mission_summary.csv                     # Aggregate run metrics (algorithms × scan depths)
    ├── Q-Learning/Scan_{5,10,15,20}/           # Per-run history + Q-table + GIF
    ├── SARSA/Scan_{5,10,15,20}/                # Per-run history + Q-table + GIF
    ├── pdfs/                                   # Pre-built result PDFs (reports, appendix, figures)
    └── figures/                                # Cross-algorithm comparison plots
```

---

## Notation and Perception Model

### Occupancy grid and geometry

- Occupancy grid $\mathcal{M}$ with cell size $r = 0.05\,\mathrm{m}$ per cell. [file:1]  
- Cells are labelled as:
  - `0` — free (traversable).  
  - `1` — obstacle (occupied).  
  - `-1` — unknown (not yet observed, fog-of-war). [file:1]

The rover moves in one of **eight discrete directions**, each associated with a $60^\circ$ view cone $V$:

- Directions: N, NE, E, SE, S, SW, W, NW. [file:1]  
- Cones and direction indices are defined in `view_cones.py`. [file:1]

### Perception parameters

Key perception quantities: [file:1]

- $\rho_{\mathrm{perc}}$ — base perception radius, `PERCEPTION_RADIUS_M`.  
- $\tau_{\text{scan}}$ — scan-depth threshold (cells), `SCAN_DEPTH_THRESHOLD`.  
- $r_{\text{scan}} = \rho_{\mathrm{perc}} + \tau_{\text{scan}} r$ — effective scan radius when active perception is triggered.  
- $\theta$ — cone angle for coverage (`THETA_DEGREE`).  
- $\varphi_{\text{fov}}$ — scan field-of-view (`PERCEPTION_FOV_DEG`). [file:1]

Within a view cone:

- $V_\phi$ — **invisible-cell ratio**: fraction of unknown cells within the scan radius for that cone. [file:1]  
- $d$ — shortest straight-line distance from the rover to the nearest obstacle or unknown cell along that cone. [file:1]  
- $\zeta$ — hazard multiplier:
  - $\zeta_{\mathrm{obs}} = 10$ for confirmed obstacles.  
  - $\zeta_{\mathrm{unk}} = 1$ for unknown cells.  
  - $1$ if the ray reaches boundary/free space unobstructed. [file:1]

**Perception Entropy**:

$$
H_{\mathrm{perc}} = -\sum_s P(s)\,\log_2 P(s),
$$

with $P(s)$ the fraction of visits to state $s$ during a mission. [file:1]

### Perception-risk score

The **perception-risk score** $\xi_{\mathrm{perc}}$ is the runtime decision variable used to rank motion directions under perception uncertainty. [file:1]

For a given direction:

$$
\xi_{\mathrm{perc}} = \zeta \cdot \frac{1}{d + 1} \cdot V_\phi.
$$

- $\frac{1}{d+1}$ emphasises nearby hazards.  
- $V_\phi$ penalises directions with greater perception uncertainty.  
- $\zeta$ distinguishes obstacles, unknown regions, and free space. [file:1]

Implementation (`unbounded_metrics.py`) computes these quantities over the entire grid and all 8 directions, yielding a tensor used inside the reward. [file:1]

---

## Reinforcement Learning Setup

### State representation (goal-blind)

States are 6-tuples: [file:1]

$$
s = (x, y, \phi, b_h, b_l, b_r),
$$

where:

- $(x, y)$ — current grid cell.  
- $\phi \in \{0,\ldots,7\}$ — rover orientation index.  
- $b_h$, $b_l$, $b_r$ — discretised straight-line distances to the nearest obstacle/unknown in forward, left, and right cones. [file:1]

Distance bins:

- $[0,1)$, $[1,2)$, $[2,4)$, $[4,8)$, $[8,\infty)$ cells. [file:1]

The goal coordinates are **not** part of the state (goal-blind), so the policy encodes general safety-aware navigation rather than a fixed GPS trajectory. [file:1]

### Action space

Actions:

$$
A = \{\text{N}, \text{NE}, \text{E}, \text{SE}, \text{S}, \text{SW}, \text{W}, \text{NW}\}.
$$

Each action corresponds to a one-step move; mappings are defined in `view_cones.py`. [file:1]

### Reward function

The planner optimises a perception-aware reward combining progress, movement cost, collision penalty, and perception-risk. [file:1]

We use a **bullet-form description** to avoid any LaTeX environment issues:

Per-step reward $R_t$:

- If $s_{t+1} = G_{\text{cell}}$ (goal reached):  
  $R_t = +1000$.
- If $s_{t+1}$ is an obstacle or unknown cell:  
  $R_t = R_{\mathrm{coll}} = -20$.
- Otherwise:  
  $R_t = \Pi_t - C_t - \xi_{\mathrm{perc}}$,

where:

- Progress term:
  $$
  \Pi_t = \omega\bigl(\|s_t - G\|_2 - \|s_{t+1} - G\|_2\bigr),
  $$
  with $\omega = 4$.  
- Movement cost: $C_t = 1$ per step (cardinal and diagonal).  
- Perception penalty: $\xi_{\mathrm{perc}}$ as defined above. [file:1]

This is exactly the reward formulation described in the paper’s perception-risk section. [file:1]

### RL algorithms

Two tabular RL algorithms are used: [file:1]

- **Q-Learning** (off-policy):

  $$
  Q(s, a) \leftarrow Q(s, a) + \alpha\bigl[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\bigr].
  $$

- **SARSA** (on-policy):

  $$
  Q(s, a) \leftarrow Q(s, a) + \alpha\bigl[r + \gamma\, Q(s', a') - Q(s, a)\bigr],
  $$

  where $a'$ is the next action chosen by $\epsilon$-greedy at $s'$. [file:1]

Training configuration:

- Episodes per mission: $N = 500$.  
- Learning rate: $\alpha = 0.4$.  
- Discount factor: $\gamma = 0.95$.  
- $\epsilon$-greedy decays from $0.99$ to $0.10$. [file:1]

---

## Algorithmic Pipeline

High-level pipeline is implemented in `main.py`, `rl_trainer.py`, `rl_environment.py`, and `unbounded_metrics.py`. [file:1]

### 1. Data ingestion and mapping

- `image_loader.py`  
  Discovers paired RGB and depth frames in `ROVER_data/realsense_D435i/`, matching timestamps in `rgb.txt` and `depth.txt` with a 30 ms tolerance, or using sorted filenames if necessary. [file:1]

- `pose_loader.py`  
  Loads ground-truth poses from `groundtruth.txt` (TUM format) and estimates rover heading via $\mathrm{atan2}(\Delta y, \Delta x)$. [file:1]

- `grid_mapper.py`  
  Projects depth into a top-down occupancy grid using:
  - Obstacle height band (e.g. 0.8–2.5 m).  
  - Hit-ratio threshold (e.g. 0.2) to classify obstacles. [file:1]

- `confidence_mapper.py`  
  Maintains per-cell statistics (observation count, depth mean/variance) and computes confidence by combining consistency and observation count. [file:1]

### 2. Active perception and coverage

Active perception decides when to trigger scans and calculates perception quantities. [file:1]

- Scan triggered in direction $a_t$ if:
  - Nearest obstacle/unknown along that direction lies within $\tau_{\text{scan}}$.  
  - Ray terminates at an unknown cell. [file:1]

- Scan reveals a $60^\circ$ cone of radius $r_{\text{scan}} = \rho_{\mathrm{perc}} + \tau_{\text{scan}}r$ from the oracle map into the agent map. [file:1]

- For each cone, $V_\phi$ (invisible-cell ratio) is computed over the scan radius via 2D convolution. [file:1]

- `unbounded_metrics.py` recomputes:
  - Directional distances $d$ to the nearest obstacle/unknown.  
  - Invisible-cell ratios $V_\phi$.  
  - Hazard multipliers $\zeta$.  
  - Directional perception-risk tensor $\xi_{\mathrm{perc}}$ for all free cells and directions. [file:1]

### 3. RL training loop

`rl_trainer.py` implements `train_online()`:

1. Agent selects $a_t$ via $\epsilon$-greedy on the current Q-table. [file:1]  
2. Scan condition checked; if triggered, map and perception-risk tensor are updated. [file:1]  
3. Environment executes the move, computes $R_t$, handles collisions and goal termination. [file:1]  
4. Q-table updated for $(s_t, a_t)$ using Q-Learning or SARSA. [file:1]  
5. Episode continues until goal reached or `MAX_RL_STEPS` exceeded. [file:1]

---

## Installation

Set up a virtual environment and install dependencies: [file:1]

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset Setup

The experiments use RGB-D sequences from the **ROVER dataset** (Intel RealSense D435i). [file:1]

- Homepage: <https://iis-esslingen.github.io/rover/>  
- Downloads: <https://iis-esslingen.github.io/rover/pages/download/>  
- HuggingFace: <https://huggingface.co/datasets/iis-esslingen/ROVER> [file:1]

### 1. Download a sequence

Example sequence (~15–19 GB): [file:1]

```text
https://fdm.hs-esslingen.de/schmidt2025rover/garden_small_2023-08-18.zip
```

### 2. Extract and place under `ROVER_data/`

```text
ROVER_data/
├── realsense_D435i/            # renamed from intelrealsense_D435i/
│   ├── rgb/<timestamp>.png
│   ├── depth/<timestamp>.png   # 16-bit PNG, depth = pixel_value × 10^{-3} m
│   ├── rgb.txt                 # timestamp list for RGB frames
│   └── depth.txt               # timestamp list for depth frames
└── groundtruth.txt             # TUM: timestamp tx ty tz qx qy qz qw
```

`ROVER_data/` is gitignored. [file:1]

### 3. Paths in `src/main.py`

```python
DATASET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "ROVER_data", "realsense_D435i")
GT_FILE     = os.path.join(os.path.dirname(SCRIPT_DIR), "ROVER_data", "groundtruth.txt")
```

### Camera and pipeline parameters

| Parameter               | Value                                   |
|-------------------------|-----------------------------------------|
| Camera                  | Intel RealSense D435i                   |
| fx = fy                 | 610 px                                  |
| cx, cy                  | 320, 240 px                             |
| Depth encoding          | 16-bit PNG, scale $10^{-3}$ m/unit      |
| Frames used             | 40–230 (1-indexed; 1–39 discarded)      |
| Obstacle height band    | 0.8 m – 2.5 m                           |
| Hit-ratio threshold     | 0.2                                     |
| Timestamp tolerance     | 30 ms                                   |

---

## Running the Pipeline

Single mission:

```bash
cd src
python main.py
```

Batch mode (non-interactive): [file:1]

```bash
# Linux / macOS
BATCH_MODE=1 python main.py

# Windows (CMD)
set BATCH_MODE=1 && python main.py
```

---

## Running Experiments

### RQ1 & RQ3 — Scan-depth sweep

```bash
cd src
python run_experiments.py
```

Sweeps $\tau_{\text{scan}} \in \{5, 10, 15, 20\}$, runs missions for both algorithms, and generates RQ1/RQ3 figures and CSVs. [file:1]

### RQ2 — Perception Entropy analysis

```bash
cd src
python generate_paper_figures.py
```

Regenerates the RQ1/RQ2 publication figures from the History CSVs. [file:1]

### RQ3 — Hypothesis tests

```bash
cd src
python hypothesis_tests.py
```

Prints Mann–Whitney U results (consecutive scan-depth pairs, both algorithms) to stdout. A committed summary and interpretation are in:

```text
src/hypothesis_test_results.md
```

> Note: `hypothesis_tests.py` does **not** write this file — it is a hand-maintained summary. Spearman correlations are produced by `generate_paper_figures.py`; Welch's $t$-tests by `post_convergence_ttest.py`.

### Hyperparameter grid search

```bash
cd src
python hyperparameter_sweep.py
```

Runs $\alpha \times \gamma \times \theta$ grid (27 combos) for both algorithms and produces summary figures. [file:1]

---

## Outputs

### Per-run outputs (`results/<Algorithm>/Scan_<δ>/`)

- `<ALG>_Th<θ>_Scan<δ>_History.csv` — per-episode metrics (reward, path length, TD error, goal flag, safety margin, tension, $H_{\mathrm{perc}}$). [file:1]  
- `<ALG>_Qtable_Scan<δ>.csv` — full Q-table entries. [file:1]  
- `mission_<ALG>_Th<θ>_Scan<δ>.gif` — animated mission (fog-of-war + path). [file:1]

### Aggregate outputs (`results/`)

- `mission_summary.csv` — convergence episode, final reward, path length, safety margin, total scans, $H_{\mathrm{perc}}$. [file:1]

### Cross-algorithm figures (`results/figures/`)

Includes:

- Final maps (Q-Learning vs SARSA).  
- Path overlays.  
- Convergence diagnostics.  
- Safety comparison CSVs and plots.  
- Paper figures generated by `generate_paper_figures.py`. [file:1]

---

## Results and Appendix

Pre-built PDFs under `results/pdfs/`: [file:1]

- `Appendix.pdf` — extended derivations, statistical tables, additional figures.  
- `results.pdf` — full report of all runs and figures.  
- `results_v2.pdf` — compact report with key tables and six central figures.  
- `Box_Plot.pdf` — safety margin distributions per scan depth.  
- `Safety_Margins.pdf` — safety margin evolution over episodes.  
- `Fig_rq1_1_QLEARNING.pdf`, `Fig_rq1_1_SARSA.pdf` — RQ1 reward convergence.  
- `Fig_rq1_2_QLEARNING.pdf`, `Fig_rq1_2_SARSA.pdf` — RQ2 reward vs Perception Entropy. [file:1]

---

## Configuration Reference

Parameters in `src/main.py`: [file:1]

| Parameter               | Symbol                  | Default | Description |
|-------------------------|-------------------------|---------|-------------|
| `START_FRAME`           | —                       | 40      | First dataset frame |
| `END_FRAME`             | —                       | 230     | Last dataset frame |
| `GOAL_FRAME`            | —                       | 60      | Frame defining goal cell |
| `RL_EPISODES`           | $N$                     | 500     | Episodes per mission |
| `RL_ALPHA`              | $\alpha$               | 0.4     | Learning rate |
| `RL_GAMMA`              | $\gamma$               | 0.95    | Discount factor |
| `MAX_RL_STEPS`          | $T_{\max}$             | 400     | Max steps per episode |
| `RL_EPS_START`          | $\varepsilon_0$       | 0.99    | Initial exploration rate |
| `RL_EPS_END`            | $\varepsilon_{\min}$  | 0.10    | Terminal exploration rate |
| `RL_EPS_DECAY`          | —                       | `"auto"` | Decay so $\varepsilon$ reaches $\varepsilon_{\min}$ at episode $N$ |
| `REWARD_GOAL`           | —                       | 1000.0  | Goal reward ($+1000$; terminates the episode) |
| `COST_CARDINAL`         | $C_t$                  | 1.0     | Movement cost (cardinal) |
| `COST_DIAGONAL`         | $C_t$                  | 1.0     | Movement cost (diagonal) |
| `PROGRESS_REWARD_SCALE` | $\omega$               | 4       | Progress reward scale |
| `VAR_DIST`              | —                       | 1.0     | Distance-softening constant; implements $\frac{1}{d+1}$ as `VAR_DIST/(d+VAR_DIST)` (code-only) |
| `THETA_DEGREE`          | $\theta$               | 60.0    | Cone angle for $V_\phi$ (degrees) |
| `HAZARD_OBSTACLE`       | $\zeta_{\mathrm{obs}}$ | 10.0    | Hazard multiplier for obstacles |
| `HAZARD_UNKNOWN`        | $\zeta_{\mathrm{unk}}$ | 1.0     | Hazard multiplier for unknown cells |
| `SCAN_DEPTH_THRESHOLD`  | $\tau_{\text{scan}}$   | 5       | Scan trigger distance (cells) |
| `PERCEPTION_RADIUS_M`   | $\rho_{\mathrm{perc}}$ | 1.0     | Base perception radius (m) |
| `PERCEPTION_FOV_DEG`    | $\varphi_{\text{fov}}$ | 60.0    | Scan FOV (degrees) |
| `REWARD_BLIND`          | —                       | 0.0     | Penalty outside dataset coverage |
| `MAX_MISSION_STEPS`     | —                       | 300     | Max physical steps per mission |
| `RUN_MODE`              | —                       | `"both"`| `"both"`, `"qlearning"`, or `"sarsa"` |
| `OUTPUT_DIR`            | —                       | `../results` | Output directory |

---

## Reproducibility Notes

- Experiments share hyperparameters and evaluation protocols across scan depths and algorithms for fair comparison. [file:1]  
- Convergence and statistical tests follow the manuscript (10%/5% reward stability, Spearman’s $\rho$, Mann–Whitney U, Welch’s $t$ with Bonferroni where applicable). [file:1]  
- Code, configurations, and seeds are part of this reproduction package as referenced in the paper. [file:1]