# Perception Risk for Path Planning in Autonomous Rover Navigation

This repository contains the **reproduction package** for the paper  
**“Perception Risk for Path Planning in Autonomous Rover Navigation”**,  
including all code, configuration files, and result artifacts needed to re-run the experiments, reproduce the figures, and inspect the statistical analyses.

The pipeline implements a closed-loop **perception–planning–adaptation** architecture that evaluates perception-aware navigation under uncertainty using tabular Q-Learning and SARSA in a fog-of-war rover scenario with RGB-D data from the Intel RealSense D435i camera. [file:1]

> **Repository name:**  
> It is recommended to rename the GitHub repository to match the paper title, e.g. `robotic-perception-risk`.  
> You can change this under **GitHub → Settings → Repository name** and update your local clone URL accordingly. [file:1]

---

## Table of Contents

1. [Research Questions](#research-questions)  
2. [Conceptual Overview](#conceptual-overview)  
3. [Repository Structure](#repository-structure)  
4. [Perception and Risk Model](#perception-and-risk-model)  
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

- **RQ1 — Perception quality vs. reward**  
  *Does increasing the scan-depth threshold \(\tau_{\text{scan}}\) improve cumulative reward?*  
  This assesses how perception quality (scan depth) affects the ability to find effective trajectories under perception uncertainty. [file:1]

- **RQ2 — Perception Entropy vs. path optimality**  
  *Does Perception Entropy, as a measure of visitation diversity, affect navigation performance during training?*  
  Here we study whether broader state visitation during learning is associated with more optimal trajectories under uncertainty. [file:1]

- **RQ3 — Safety saturation under increased sensing**  
  *Does increasing the scan-depth threshold \(\tau_{\text{scan}}\) improve the average obstacle-clearance distance \(\bar{d}_{\text{clear}}\) during training? Is there a saturation point beyond which additional scan depth yields diminishing returns?*  
  This investigates whether safety margins, measured as average obstacle distance, saturate as sensing range is extended. [file:1]

The code and analysis scripts are structured such that each RQ can be reproduced end-to-end using the commands listed in the sections below. [file:1]

---

## Conceptual Overview

The paper introduces a **perception-risk score** that exposes perception uncertainty as an explicit runtime decision variable for trajectory evaluation, rather than embedding uncertainty only inside perception or planning components. [file:1]

- A **grid-based perception model** combines:
  - Obstacle proximity (distance to nearest obstacle/unknown). [file:1]
  - Perception coverage (fraction of unknown cells in a view cone). [file:1]
  - Hazard multipliers distinguishing obstacles, unknown regions, and free space. [file:1]

- The **perception-risk score** \(\xi_{\text{perc}}\) is used as a reward penalty within a tabular RL agent, enabling the policy to:
  - Prefer safer trajectories with better perception confidence. [file:1]
  - Trade off navigation efficiency (progress toward goal) with risk and uncertainty. [file:1]

- The **closed-loop architecture** couples:
  - Mapping and confidence accumulation from RGB-D data. [file:1]
  - Active perception (directional scans triggered by unknown hazards). [file:1]
  - Reinforcement learning (Q-Learning and SARSA) trained with a perception-aware reward function. [file:1]

Perception Entropy \(H_{\text{perc}} = -\sum_s P(s) \log_2 P(s)\) is used to quantify visitation diversity during training and to analyse how exploration behaviour relates to navigation performance under uncertainty. [file:1]

---

## Repository Structure

All code lives under `src/`, and all generated artefacts are written to `results/`. [file:1]

```text
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                                        # Source code (reproduction package)
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
│   ├── adaptive_perception.py                  # Adaptive perception model (unbounded-risk variant)
│   │
│   ├── image_loader.py                         # RGB-D frame discovery and loading
│   ├── pose_loader.py                          # Ground-truth pose loading (TUM format)
│   │
│   ├── run_experiments.py                      # Automated τ_scan sweep (RQ1, RQ3)
│   ├── hyperparameter_sweep.py                 # Grid search over α, γ, θ
│   ├── compare_safety.py                       # Q-Learning safety margin comparison
│   ├── compare_safety_sarsa.py                 # SARSA safety margin comparison
│   ├── hypothesis_tests.py                     # Hypothesis tests (RQ3; Mann–Whitney U, Spearman)
│   │
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
    ├── pdfs/                                   # Pre-built result PDFs, paper appendix and reports
    └── figures/                                # Cross-algorithm comparison plots
```

This structure mirrors the paper’s separation between data ingestion, perception modelling, RL training, and analysis/plotting. [file:1]

---

## Perception and Risk Model

### Notation and Geometry

The environment is a 2-D occupancy grid \(\mathcal{M}\) with cell size \(r = 0.05\ \text{m}\), built from RealSense D435i RGB-D data and ground-truth poses. [file:1]

- Each grid cell corresponds to a state \(s\) with integer coordinates \((x, y)\). [file:1]
- The rover moves in one of eight discrete directions (N, NE, E, SE, S, SW, W, NW), each associated with a \(60^\circ\) view cone \(V\). [file:1]
- The occupancy values are:
  - `0` — free (traversable). [file:1]
  - `1` — occupied (obstacle). [file:1]
  - `-1` — unknown (unobserved / fog-of-war). [file:1]

The **perception radius** \(\rho_{\text{perc}} = 1\ \text{m}\) defines a base sensing range, while the **scan-depth threshold** \(\tau_{\text{scan}}\) (in grid cells) extends the effective scan radius to:
\[
r_{\text{scan}} = \rho_{\text{perc}} + \tau_{\text{scan}} \cdot r
\]
which is used when active perception is triggered. [file:1]

### Perception Quantities

For each view cone \(V\) centred at a rover pose and heading: [file:1]

- \(d\) is the shortest straight-line distance from the rover to the nearest obstacle or unknown cell along that direction. [file:1]
- \(V_{\phi}\) is the **invisible-cell ratio**: the fraction of unknown cells within the view cone up to radius \(r_{\text{scan}}\). [file:1]
- \(\zeta\) is a **hazard multiplier**:
  - \(\zeta_{\text{obs}} = 10\) for confirmed obstacles. [file:1]
  - \(\zeta_{\text{unk}} = 1\) for unknown cells. [file:1]
  - \(1\) if the ray reaches free space/boundary unobstructed. [file:1]

Perception Entropy \(H_{\text{perc}} = -\sum_s P(s)\log_2 P(s)\) measures visitation diversity, with \(P(s)\) the fraction of visits to state \(s\) over a mission. [file:1]

### Perception Risk Score

The perception-risk score \(\xi_{\text{perc}}\) serves as the runtime decision variable for evaluating motion directions and ranking planner-generated trajectories. [file:1]

For a given direction:

\[
\xi_{\text{perc}} = \zeta \cdot \frac{1}{d + 1} \cdot V_{\phi}
\]

- The inverse-distance term emphasises nearby hazards. [file:1]
- The invisible-cell ratio penalises directions with greater perception uncertainty. [file:1]
- The hazard multiplier distinguishes obstacles, unknown regions, and free space. [file:1]

In the vectorised implementation (`unbounded_metrics.py`), the components are computed across the entire grid and 8 directions to obtain a **directional hazard tensor** \(\xi\) that is used in the reward function. [file:1]

---

## Reinforcement Learning Setup

### State Representation (Goal-Blind)

States are 6-tuples:
\[
s = (x, y, \phi, b_h, b_l, b_r)
\]

- \((x, y)\): current grid cell. [file:1]
- \(\phi \in \{0, \ldots, 7\}\): rover orientation, aligned with the 8 discrete actions. [file:1]
- \(b_h, b_l, b_r\): discretised distances to the nearest obstacle/unknown in the forward, left, and right view cones. [file:1]

Distances are binned into five ranges:

- \([0, 1)\), \([1, 2)\), \([2, 4)\), \([4, 8)\), and \([8, \infty)\) cells. [file:1]

The goal coordinates are **not** part of the state (goal-blind), so the learned policy is a general safety-aware navigation strategy rather than a fixed GPS trajectory. [file:1]

### Action Space

The action set is:
\[
A = \{\text{N}, \text{NE}, \text{E}, \text{SE}, \text{S}, \text{SW}, \text{W}, \text{NW}\}
\]

Each action corresponds to a 1-step move in the grid with consistent mapping to direction indices in `view_cones.py`. [file:1]

### Reward Function

For action \(a_t\) in state \(s_t\) leading to \(s_{t+1}\), the perception-aware reward is: [file:1]

\[
R_t =
\begin{cases}
+1000 & \text{if } s_{t+1} = G_{\text{cell}} \ (\text{goal reached}), \\
R_{\text{coll}} & \text{if } s_{t+1} \in \{\text{obstacle},\ \text{unknown}\}, \\
\Pi_t - C_t - \xi_{\text{perc}} & \text{otherwise},
\end{cases}
\]

where:

- Goal bonus: \(+1000\). [file:1]
- Collision penalty: \(R_{\text{coll}} = -20\). [file:1]
- Progress reward: \(\Pi_t = \omega(\|s_t - G\|_2 - \|s_{t+1} - G\|_2)\) with \(\omega = 4\). [file:1]
- Movement cost: \(C_t = 1\) per step (cardinal and diagonal). [file:1]
- Perception penalty: \(\xi_{\text{perc}}\) for the chosen direction. [file:1]

### RL Algorithms

The perception-risk scoring is evaluated with two tabular RL methods:

- **Q-Learning** (off-policy, risk-seeking tendency): [file:1]  
  \[
  Q(s, a) \leftarrow Q(s, a) + \alpha \big[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \big]
  \]

- **SARSA** (on-policy, more conservative): [file:1]  
  \[
  Q(s, a) \leftarrow Q(s, a) + \alpha \big[ r + \gamma Q(s', a') - Q(s, a) \big]
  \]
  where \(a'\) is the next action selected by \(\epsilon\)-greedy at \(s'\).

The reproduction package uses model-free tabular RL with \(500\) episodes per configuration, learning rate \(\alpha = 0.4\), discount factor \(\gamma = 0.95\), and \(\epsilon\)-greedy exploration decaying from \(0.99\) to \(0.10\). [file:1]

---

## Algorithmic Pipeline

The high-level pipeline follows the perception–planning–adaptation loop described in the paper, implemented primarily in `main.py`, `rl_environment.py`, `rl_trainer.py`, and `unbounded_metrics.py`. [file:1]

### 1. Data ingestion and mapping

- `image_loader.py` discovers paired RGB and depth frames in `ROVER_data/realsense_D435i/`, matching timestamps with a tolerance of 30 ms (or falling back to sorted filenames where necessary). [file:1]
- `pose_loader.py` loads ground-truth camera poses from `groundtruth.txt` (TUM format) and estimates rover heading via successive positions. [file:1]
- `grid_mapper.py` fuses depth and pose into a top-down occupancy grid, applying:
  - An obstacle-height band (e.g. \(0.8\ \text{m}–2.5\ \text{m}\)) to avoid floor clutter. [file:1]
  - A hit-ratio threshold to classify cells as obstacles or free. [file:1]
- `confidence_mapper.py` maintains per-cell statistics (observation count, depth mean/variance) and computes confidence scores combining consistency and observation frequency. [file:1]

### 2. Active perception and coverage

Active perception logic determines when scans are triggered and updates the map: [file:1]

- Before each move, the trainer checks the nearest obstacle/unknown distance in the direction of the chosen action. [file:1]
- If \(d(s_t, a_t) < \tau_{\text{scan}}\) *and* the ray terminates in an unknown cell, a new scan is triggered. [file:1]
- A \(60^\circ\) cone up to radius \(r_{\text{scan}}\) is revealed from the oracle map into the agent-side occupancy grid. [file:1]
- `unbounded_metrics.py` recomputes, over the entire map:
  - Directional distances to the nearest obstacle/unknown. [file:1]
  - Invisible-cell ratios \(V_{\phi}\) via 2D convolution over the occupancy grid. [file:1]
  - The hazard tensor \(\xi\) for all free cells and 8 directions. [file:1]

### 3. RL episode loop

`rl_trainer.py` implements `train_online()`:

1. **Action selection**  
   The agent selects an action via \(\epsilon\)-greedy over the current Q-table. [file:1]

2. **Scan check and hazard recomputation**  
   The scan trigger condition is evaluated; on a scan, the map and hazard tensor are rebuilt. [file:1]

3. **Transition and reward**  
   The environment executes the move, applies collision handling, and computes the perception-aware reward. [file:1]

4. **Q-update**  
   A single \((s, a)\) entry in the Q-table is updated according to either Q-Learning or SARSA. [file:1]

5. **Loop**  
   The process repeats until the goal is reached or `MAX_RL_STEPS` (default 400) is exceeded. [file:1]

Across episodes, \(\epsilon\) decays from \(0.99\) to \(0.10\), and the Q-table is persisted, enabling knowledge accumulation as perception improves. [file:1]

---

## Installation

Create a Python virtual environment and install the dependencies: [file:1]

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The code targets standard Python and scientific libraries (NumPy, SciPy, pandas, Matplotlib, etc.) as specified in `requirements.txt`. [file:1]

---

## Dataset Setup

The experiments use RGB-D rover sequences from the **ROVER dataset** (Intel RealSense D435i, outdoor ground rover). [file:1]

- Dataset homepage: <https://iis-esslingen.github.io/rover/>  
- Direct downloads: <https://iis-esslingen.github.io/rover/pages/download/>  
- HuggingFace mirror: <https://huggingface.co/datasets/iis-esslingen/ROVER> [file:1]

### 1. Download a sequence

Any sequence compatible with the RealSense D435i is supported. For example (≈15–19 GB): [file:1]

```text
https://fdm.hs-esslingen.de/schmidt2025rover/garden_small_2023-08-18.zip
```

### 2. Extract and place under `ROVER_data/`

After extraction, ensure the following layout at the repository root: [file:1]

```text
ROVER_data/
├── realsense_D435i/            # renamed from intelrealsense_D435i/
│   ├── rgb/<timestamp>.png
│   ├── depth/<timestamp>.png   # 16-bit PNG, depth = pixel_value × 10^{-3} m
│   ├── rgb.txt                 # timestamp list for RGB frames
│   └── depth.txt               # timestamp list for depth frames
└── groundtruth.txt             # TUM format: timestamp tx ty tz qx qy qz qw
```

The `ROVER_data/` directory is gitignored and not committed to the repository. [file:1]

### 3. Paths in `src/main.py`

Paths are configured at the top of `src/main.py`: [file:1]

```python
DATASET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "ROVER_data", "realsense_D435i")
GT_FILE     = os.path.join(os.path.dirname(SCRIPT_DIR), "ROVER_data", "groundtruth.txt")
```

### Camera and pipeline parameters

| Parameter          | Value                                  |
|--------------------|----------------------------------------|
| Camera             | Intel RealSense D435i                  |
| fx = fy            | 610 px                                 |
| cx, cy             | 320, 240 px                            |
| Depth encoding     | 16-bit PNG, scale \(10^{-3}\) m/unit   |
| Frames used        | 40–230 (1-indexed; 1–39 discarded)     |
| Obstacle band      | 0.8 m – 2.5 m                          |
| Hit-ratio threshold| 0.2                                    |
| Timestamp tolerance| 30 ms                                  |

These values match the setup used in the paper’s evaluation section. [file:1]

---

## Running the Pipeline

To run a single mission (one algorithm configuration, one scan depth):

```bash
cd src
python main.py
```

Batch mode (non-interactive, suitable for scripts and experiment sweeps): [file:1]

```bash
# Linux / macOS
BATCH_MODE=1 python main.py

# Windows (CMD)
set BATCH_MODE=1 && python main.py
```

This will:

- Build the oracle map from the selected sequence. [file:1]
- Initialise the agent-side occupancy map. [file:1]
- Train the RL agent for `RL_EPISODES` episodes. [file:1]
- Follow the greedy path physically and record mission metrics. [file:1]
- Generate per-run plots and GIFs under `results/`. [file:1]

---

## Running Experiments

### RQ1 & RQ3 — Scan-depth sweep

To sweep \(\tau_{\text{scan}} \in \{5, 10, 15, 20\}\) and generate all comparison figures and safety statistics: [file:1]

```bash
cd src
python run_experiments.py
```

`run_experiments.py`:

- Iterates over scan depths, updating `SCAN_DEPTH_THRESHOLD` in `main.py`. [file:1]
- Runs `main.py` in batch mode for each configuration. [file:1]
- Calls `plot_results.py`, `compare_safety.py`, `compare_safety_sarsa.py`, and `plot_safety_comparison.py`. [file:1]
- Produces the data and figures used for RQ1 and RQ3 in the paper. [file:1]

### RQ2 — Perception Entropy analysis

Perception Entropy vs. reward plots are automatically generated at the end of `run_experiments.py`. To regenerate only the publication figures: [file:1]

```bash
cd src
python generate_paper_figures.py
```

This script uses episode-level History CSVs to compute the relationship between \(H_{\text{perc}}\) and cumulative reward, split into pre- and post-convergence phases. [file:1]

### RQ3 — Hypothesis tests

To reproduce the Mann–Whitney U and Spearman correlation analyses:

```bash
cd src
python hypothesis_tests.py
```

Outputs (test statistics, p-values, effect sizes, interpretation) are written to: [file:1]

```text
src/hypothesis_test_results.md
```

### Hyperparameter grid search

To run the grid search over \(\alpha \times \gamma \times \theta\) for both RL algorithms (27 combinations):

```bash
cd src
python hyperparameter_sweep.py
```

The script creates a results table and a 27-panel map figure summarising how different hyperparameter settings affect navigation outcomes. [file:1]

---

## Outputs

All outputs are written to `results/`, organised by algorithm and scan depth. [file:1]

### Per-run outputs

Under `results/<Algorithm>/Scan_<δ>/`:

- `<ALG>_Th<θ>_Scan<δ>_History.csv`  
  Per-episode metrics: cumulative reward, path length, TD error, goal success flag, average safety margin, average tension, Perception Entropy \(H_{\text{perc}}\). [file:1]

- `<ALG>_Qtable_Scan<δ>.csv`  
  Full Q-table entries: \((x, y, \phi, b_h, b_l, b_r, \text{action}, \text{Q-value}, \text{visit count}, \text{tension})\). [file:1]

- `mission_<ALG>_Th<θ>_Scan<δ>.gif`  
  Animated occupancy map showing fog-of-war revelation and the learned greedy path during the mission. [file:1]

### Aggregate outputs

At the root of `results/`:

- `mission_summary.csv`  
  One row per run, including convergence episode, final greedy reward (episode 500), path length, safety margin, total scans, and \(H_{\text{perc}}\). [file:1]

### Cross-algorithm figures

Under `results/figures/`:

- `map_comparison_Th<θ>_Scan<δ>.png` — final maps side-by-side for Q-Learning vs SARSA. [file:1]
- `path_overlay_Th<θ>_Scan<δ>.png` — greedy paths overlaid on the accumulated occupancy grid. [file:1]
- `convergence_diagnostics_Th<θ>_Scan<δ>.png` — episode rewards, path lengths, TD error magnitudes over training. [file:1]
- `safety_comparison_qlearning.csv`, `safety_comparison_sarsa.csv` — safety margins vs. scan depth (RQ3 input). [file:1]
- Paper figures generated by `generate_paper_figures.py`. [file:1]

---

## Results and Appendix

Pre-built PDF reports are committed under `results/pdfs/`, so you can inspect the main findings without re-running the pipeline. [file:1]

- `Appendix.pdf` — supplementary material for *Perception Entropy for Path Planning in Autonomous Rover Navigation* (extended derivations, full statistical tables, extra figures). [file:1]
- `results.pdf` — full report: statistical tables, CSV excerpts around convergence points, and all main figures. [file:1]
- `results_v2.pdf` — compact report with key tables and six core publication figures. [file:1]
- `Box_Plot.pdf` — safety-margin distributions per scan depth, Q-Learning vs SARSA. [file:1]
- `Safety_Margins.pdf` — safety margin evolution across 500 episodes. [file:1]
- `Fig_rq1_1_QLEARNING.pdf`, `Fig_rq1_1_SARSA.pdf` — cumulative reward convergence across scan depths for both algorithms (RQ1). [file:1]
- `Fig_rq1_2_QLEARNING.pdf`, `Fig_rq1_2_SARSA.pdf` — reward vs Perception Entropy with Spearman \(\rho\) (RQ2). [file:1]

Statistical test outputs and commentary are available in `src/hypothesis_test_results.md`. [file:1]

---

## Configuration Reference

All configuration parameters are declared at the top of `src/main.py`. [file:1]

| Parameter               | Symbol             | Default | Description                                              |
|-------------------------|--------------------|---------|----------------------------------------------------------|
| `START_FRAME`           | —                  | 40      | First dataset frame (1-indexed)                         |
| `END_FRAME`             | —                  | 230     | Last dataset frame                                      |
| `GOAL_FRAME`            | —                  | 60      | Frame whose pose defines the goal cell                  |
| `RL_EPISODES`           | \(N\)              | 500     | Training episodes per mission                           |
| `RL_ALPHA`              | \(\alpha\)         | 0.4     | Learning rate                                           |
| `RL_GAMMA`              | \(\gamma\)         | 0.95    | Discount factor                                         |
| `MAX_RL_STEPS`          | \(T_{\max}\)       | 400     | Maximum steps per episode                               |
| `RL_EPS_START`          | \(\varepsilon_0\)  | 0.99    | Initial exploration rate                                |
| `RL_EPS_END`            | \(\varepsilon_{\min}\) | 0.10 | Terminal exploration rate                               |
| `RL_EPS_DECAY`          | —                  | `"auto"`| Auto-computed decay; reaches \(\varepsilon_{\min}\) at episode \(N\) |
| `REWARD_GOAL`           | \(\Omega_t\)       | 1000.0  | Goal reward                                             |
| `COST_CARDINAL`         | \(C_t\)            | 1.0     | Movement cost (cardinal)                                |
| `COST_DIAGONAL`         | \(C_t\)            | 1.0     | Movement cost (diagonal)                                |
| `PROGRESS_REWARD_SCALE` | \(\omega\)         | 4       | Progress reward scaling factor                          |
| `VAR_DIST`              | \(d_{\text{var}}\) | 1.0     | Distance softening constant in hazard formula           |
| `THETA_DEGREE`          | \(\theta\)         | 60.0    | FOV cone angle for \(V_{\phi}\) (degrees)               |
| `HAZARD_OBSTACLE`       | \(\zeta_{\text{obs}}\) | 10.0 | Hazard multiplier for obstacles                         |
| `HAZARD_UNKNOWN`        | \(\zeta_{\text{unk}}\) | 1.0  | Hazard multiplier for unknown cells                     |
| `SCAN_DEPTH_THRESHOLD`  | \(\tau_{\text{scan}}\) | 5   | Scan trigger distance (cells); swept for RQ1/RQ3        |
| `PERCEPTION_RADIUS_M`   | \(\rho_{\text{perc}}\) | 1.0 | Base vision-cone radius (metres)                        |
| `PERCEPTION_FOV_DEG`    | \(\varphi_{\text{fov}}\) | 60.0 | Scan FOV (degrees)                                     |
| `REWARD_BLIND`          | —                  | 0.0     | Penalty for stepping outside dataset coverage           |
| `MAX_MISSION_STEPS`     | —                  | 300     | Maximum physical steps before mission failure           |
| `RUN_MODE`              | —                  | `"both"`| `"both"`, `"qlearning"`, or `"sarsa"`                   |
| `OUTPUT_DIR`            | —                  | `../results` | Output directory relative to `src/`                 |

Adjusting these parameters allows you to reproduce variants of the experiment design or explore alternative sensing and learning configurations. [file:1]

---

## Reproducibility Notes

- All experiments in the paper use fixed hyperparameters and shared evaluation protocols across scan depths to ensure fair comparisons between Q-Learning and SARSA. [file:1]
- Episode-level statistics, convergence criteria, and hypothesis tests follow the definitions in the manuscript (e.g. 10%/5% reward stability thresholds, Spearman correlations, Welch’s t-tests with Bonferroni correction where applicable). [file:1]
- Code, configurations, and random seeds are included in this reproduction package as referenced in the paper. [file:1]

If you plan to extend the framework (e.g. continuous action spaces, multi-agent coordination, causal perception transfer), it is recommended to keep this `README.md` as the top-level documentation and add dedicated sections for new modules and experimental protocols. [file:1]

---

If you tell me your preferred structure (e.g., shorter README for GitHub vs. a more tutorial-style one), I can adapt this file to be more concise or add extra sections like “Quick Start” and “Paper–Code Mapping”.