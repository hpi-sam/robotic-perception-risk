# Perception Risk for Path Planning in Autonomous Rover Navigation

This repository contains the **reproduction package** for the paper:

> **Perception Risk for Path Planning in Autonomous Rover Navigation**

It implements a closed-loop **perception–planning–adaptation** architecture for autonomous rover navigation under perception uncertainty, using model-free reinforcement learning (SARSA and Q-Learning) on real RGB-D rover data from an Intel RealSense D435i sensor. [file:1]

The code exposes a **perception-risk score** as an explicit runtime decision variable, combining geometric risk (obstacle distance) and perception confidence (coverage of unknown cells) in a unified scalar that shapes the reward function during policy training. [file:1]

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

The code and analysis scripts are structured such that each RQ can be reproduced from the commands listed below.

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
│   ├── hypothesis_tests.py                     # Hypothesis tests (Mann–Whitney U, Spearman)
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
- Cones are defined and indexed in `view_cones.py` and used throughout the perception pipeline. [file:1]

### Perception parameters

The main perception-related quantities are: [file:1]

- $\rho_{\mathrm{perc}}$ — base perception radius, implemented as `PERCEPTION_RADIUS_M`.  
- $\tau_{\text{scan}}$ — scan-depth threshold (cells), implemented as `SCAN_DEPTH_THRESHOLD`.  
- $r_{\text{scan}} = \rho_{\mathrm{perc}} + \tau_{\text{scan}} r$ — effective scan radius used when triggering active perception.  
- $\theta$ — cone angle used for coverage calculations (`THETA_DEGREE`).  
- $\varphi_{\text{fov}}$ — scan field-of-view parameter, implemented as `PERCEPTION_FOV_DEG`. [file:1]

Within a view cone:

- $V_\phi$ — **invisible-cell ratio**: fraction of unknown cells within the scan radius for that cone. [file:1]  
- $d$ — shortest straight-line distance from the rover to the nearest obstacle or unknown cell in that cone. [file:1]  
- $\zeta$ — hazard multiplier:
  - $\zeta_{\mathrm{obs}} = 10$ for confirmed obstacles.  
  - $\zeta_{\mathrm{unk}} = 1$ for unknown cells.  
  - $1$ for free/boundary cells. [file:1]

**Perception Entropy** is defined as:

\[
H_{\mathrm{perc}} = -\sum_s P(s)\,\log_2 P(s),
\]

with $P(s)$ the fraction of visits to state $s$ during a mission, and is tracked per episode. [file:1]

### Perception-risk score

The **perception-risk score** $\xi_{\mathrm{perc}}$ serves as the runtime decision variable for ranking candidate trajectories under perception uncertainty. [file:1]

For a given direction:

\[
\xi_{\mathrm{perc}} = \zeta \cdot \frac{1}{d + 1} \cdot V_\phi.
\]

- The **inverse-distance** term $\frac{1}{d+1}$ emphasises nearby hazards.  
- The **invisible-cell ratio** $V_\phi$ penalises directions with greater perception uncertainty.  
- The **hazard multiplier** $\zeta$ distinguishes obstacles, unknown regions, and free space. [file:1]

In the implementation (`unbounded_metrics.py`), these quantities are computed over the entire grid for all 8 directions, yielding a directional hazard tensor that is used inside the reward function. [file:1]

---

## Reinforcement Learning Setup

### State representation (goal-blind)

Each state corresponds to a 6-tuple: [file:1]

\[
s = (x, y, \phi, b_h, b_l, b_r),
\]

where:

- $(x, y)$ — current grid cell.  
- $\phi \in \{0,\ldots,7\}$ — rover orientation (index into the 8 directions).  
- $b_h$, $b_l$, $b_r$ — discretised straight-line distances to the nearest obstacle or unknown cell in the forward, left, and right view cones, respectively. [file:1]

Distances are binned into five ranges:

- $[0,1)$, $[1,2)$, $[2,4)$, $[4,8)$, and $[8,\infty)$ cells. [file:1]

The goal coordinates are **not** included in the state (goal-blind design), so the learned policy is a general safety-aware navigation strategy rather than a fixed GPS trajectory. [file:1]

### Action space

The action set is: [file:1]

\[
A = \{\text{N}, \text{NE}, \text{E}, \text{SE}, \text{S}, \text{SW}, \text{W}, \text{NW}\},
\]

with each action representing a single grid step; `view_cones.py` provides the mapping from actions to direction indices. [file:1]

### Reward function

The planner optimises a perception-aware reward function that combines progress, movement cost, collision penalty, and the perception-risk score. [file:1]

For action $a_t$ in state $s_t$ leading to $s_{t+1}$:

\[
R_t =
\begin{cases}
+1000 & \text{if } s_{t+1} = G_{\text{cell}} \ (\text{goal reached}),\\[2pt]
R_{\mathrm{coll}} & \text{if } s_{t+1} \in \{\text{obstacle},\ \text{unknown}\},\\[2pt]
\Pi_t - C_t - \xi_{\mathrm{perc}} & \text{otherwise,}
\end{cases}
\]

where:

- Goal bonus: $+1000$.  
- Collision penalty: $R_{\mathrm{coll}} = -20$ when attempting to enter an obstacle or unknown cell.  
- Progress reward:
  \[
  \Pi_t = \omega\bigl(\|s_t - G\|_2 - \|s_{t+1} - G\|_2\bigr),
  \]
  with $\omega = 4$.  
- Movement cost: $C_t = 1$ per step (cardinal and diagonal).  
- Perception penalty: $\xi_{\mathrm{perc}}$ for the chosen direction from the current state. [file:1]

### RL algorithms

The perception-risk formulation is evaluated with two tabular RL algorithms: [file:1]

- **Q-Learning** (off-policy):

  \[
  Q(s, a) \leftarrow Q(s, a) + \alpha\bigl[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\bigr],
  \]

- **SARSA** (on-policy):

  \[
  Q(s, a) \leftarrow Q(s, a) + \alpha\bigl[r + \gamma\, Q(s', a') - Q(s, a)\bigr],
  \]

  where $a'$ is the next action selected by $\epsilon$-greedy at $s'$. [file:1]

Both methods are trained for 500 episodes per configuration, with:

- Learning rate: $\alpha = 0.4$.  
- Discount factor: $\gamma = 0.95$.  
- $\epsilon$-greedy exploration decaying from $0.99$ to $0.10$. [file:1]

---

## Algorithmic Pipeline

The closed-loop pipeline is implemented in `main.py`, `rl_trainer.py`, `rl_environment.py`, and `unbounded_metrics.py`. [file:1]

### 1. Data ingestion and mapping

- `image_loader.py`  
  Discovers paired RGB and depth frames in `ROVER_data/realsense_D435i/`, matching timestamps from `rgb.txt` and `depth.txt` with a 30 ms tolerance; falls back to sorted filename pairing if timestamps are missing. [file:1]

- `pose_loader.py`  
  Loads ground-truth poses from `ROVER_data/groundtruth.txt` (TUM format) and estimates rover heading from consecutive positions via $\mathrm{atan2}(\Delta y, \Delta x)$. [file:1]

- `grid_mapper.py`  
  Projects depth images into a top-down 2-D occupancy grid, using:
  - An obstacle-height band (e.g. 0.8–2.5 m) to avoid floor clutter.  
  - A hit-ratio threshold (e.g. 0.2) to classify obstacles. [file:1]

- `confidence_mapper.py`  
  Maintains per-cell statistics (observation count, depth mean/variance) and computes confidence by combining consistency (low variance) and observation count. [file:1]

### 2. Active perception and coverage estimation

Active perception decides when to trigger additional scans and computes the perception quantities used in the risk score. [file:1]

- **Scan trigger condition**:  

  A scan in direction $a_t$ is triggered whenever:

  - The nearest obstacle/unknown along that direction lies within the scan-depth threshold $\tau_{\text{scan}}$.  
  - The ray terminates at an unknown cell (not at an obstacle or boundary). [file:1]

- **Scan effect**:  

  A $60^\circ$ cone of radius $r_{\text{scan}} = \rho_{\mathrm{perc}} + \tau_{\text{scan}}r$ is revealed from the oracle map into the agent-side occupancy grid. [file:1]

- **Coverage computation**:  

  For each view cone $V$, the invisible-cell ratio $V_\phi$ is computed as the fraction of unknown cells within the scan radius. Rather than stopping at the nearest obstacle, the computation considers the entire scan radius, using 2D convolution over the occupancy grid for efficiency. [file:1]

- **Hazard tensor** (`unbounded_metrics.py`):  

  After each scan, the module recomputes for all free cells and all 8 directions:
  - Directional distances $d$ to the nearest obstacle/unknown.  
  - Invisible-cell ratios $V_\phi$.  
  - Hazard multipliers $\zeta$.  
  - The perception-hazard tensor $\xi$ used in the reward. [file:1]

### 3. RL training loop

`rl_trainer.py` implements `train_online()`:

1. **Action selection**  
   The agent chooses $a_t$ via $\epsilon$-greedy on the current Q-table. [file:1]

2. **Scan check**  
   The scan trigger condition is evaluated; if satisfied, a new cone is revealed and the hazard tensor is recomputed. [file:1]

3. **Transition and reward**  
   The environment executes the move, handling collisions (stay in place with penalty) and goal termination, and computes the perception-aware reward $R_t$. [file:1]

4. **Q-update**  
   A single $(s_t, a_t)$ entry in the Q-table is updated using either Q-Learning or SARSA. [file:1]

5. **Episode progression**  
   Steps repeat until the goal is reached or `MAX_RL_STEPS` (default 400) is exceeded. `RL_EPISODES` (default 500) control the total number of episodes per mission; $\epsilon$ decays once across the whole mission. [file:1]

---

## Installation

Create a virtual environment and install dependencies: [file:1]

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies include NumPy, SciPy, pandas, Matplotlib, and other standard libraries used throughout the reproduction package. [file:1]

---

## Dataset Setup

The experiments use RGB-D sequences from the **ROVER dataset** (Intel RealSense D435i outdoor rover). [file:1]

- Dataset homepage: <https://iis-esslingen.github.io/rover/>  
- Downloads: <https://iis-esslingen.github.io/rover/pages/download/>  
- HuggingFace mirror: <https://huggingface.co/datasets/iis-esslingen/ROVER> [file:1]

### 1. Download a sequence

Any RealSense D435i sequence can be used. For example (~15–19 GB): [file:1]

```text
https://fdm.hs-esslingen.de/schmidt2025rover/garden_small_2023-08-18.zip
```

### 2. Extract and place under `ROVER_data/`

Place the extracted data at the repository root in the following structure: [file:1]

```text
ROVER_data/
├── realsense_D435i/            # renamed from intelrealsense_D435i/
│   ├── rgb/<timestamp>.png
│   ├── depth/<timestamp>.png   # 16-bit PNG, depth = pixel_value × 10^{-3} m
│   ├── rgb.txt                 # timestamp list for RGB frames
│   └── depth.txt               # timestamp list for depth frames
└── groundtruth.txt             # TUM format: timestamp tx ty tz qx qy qz qw
```

`ROVER_data/` is gitignored and not committed. [file:1]

### 3. Paths in `src/main.py`

Paths are configured at the top of `src/main.py`: [file:1]

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

These values match the evaluation setup described in the manuscript. [file:1]

---

## Running the Pipeline

To run a single mission (one algorithm, one scan depth):

```bash
cd src
python main.py
```

Batch mode (non-interactive, suitable for sweeps): [file:1]

```bash
# Linux / macOS
BATCH_MODE=1 python main.py

# Windows (CMD)
set BATCH_MODE=1 && python main.py
```

This will:

- Build the oracle map from the selected sequence.  
- Initialise the agent-side occupancy map.  
- Train the RL agent for `RL_EPISODES` episodes.  
- Execute the greedy path and record mission-level metrics.  
- Generate per-run plots and GIFs under `results/`. [file:1]

---

## Running Experiments

### RQ1 & RQ3 — Scan-depth sweep

To sweep $\tau_{\text{scan}} \in \{5, 10, 15, 20\}$ and generate all comparison figures and safety statistics: [file:1]

```bash
cd src
python run_experiments.py
```

`run_experiments.py`:

- Iterates over scan depths, updating `SCAN_DEPTH_THRESHOLD` in `main.py`.  
- Runs `main.py` in batch mode for each configuration.  
- Calls plotting and comparison scripts (`plot_results.py`, `compare_safety.py`, `compare_safety_sarsa.py`, `plot_safety_comparison.py`).  
- Produces the data and figures used for RQ1 and RQ3. [file:1]

### RQ2 — Perception Entropy analysis

Perception Entropy vs reward plots are generated automatically at the end of `run_experiments.py`. To regenerate only the RQ1/RQ2 publication figures: [file:1]

```bash
cd src
python generate_paper_figures.py
```

This script uses the per-episode `History.csv` files to analyse the relationship between $H_{\mathrm{perc}}$ and reward before and after convergence. [file:1]

### RQ3 — Hypothesis tests

To reproduce the Mann–Whitney U and Spearman correlation analyses: [file:1]

```bash
cd src
python hypothesis_tests.py
```

Results (test statistics, $p$-values, effect sizes, interpretation) are written to:

```text
src/hypothesis_test_results.md
```

### Hyperparameter grid search

To run the grid search over $\alpha \times \gamma \times \theta$ (27 combinations) for both RL algorithms: [file:1]

```bash
cd src
python hyperparameter_sweep.py
```

This produces:

- A PNG table summarising results across hyperparameters.  
- A 27-panel figure visualising map outcomes. [file:1]

---

## Outputs

All outputs are written to `results/`, organised by algorithm and scan depth. [file:1]

### Per-run outputs (`results/<Algorithm>/Scan_<δ>/`)

- `<ALG>_Th<θ>_Scan<δ>_History.csv`  
  Per-episode metrics: cumulative reward, path length, mean TD error, goal-reached flag, average safety margin, average tension, Perception Entropy $H_{\mathrm{perc}}$. [file:1]

- `<ALG>_Qtable_Scan<δ>.csv`  
  Full Q-table entries: $(x, y, \phi, b_h, b_l, b_r, \text{action}, \text{Q-value}, \text{visit count}, \text{tension})$. [file:1]

- `mission_<ALG>_Th<θ>_Scan<δ>.gif`  
  Animated occupancy map showing fog-of-war revelation and the learned greedy path. [file:1]

### Aggregate outputs (`results/`)

- `mission_summary.csv`  
  One row per run: convergence episode, final greedy reward (episode 500), greedy path length, safety margin, total scan count, $H_{\mathrm{perc}}$. [file:1]

### Cross-algorithm figures (`results/figures/`)

- `map_comparison_Th<θ>_Scan<δ>.png` — final maps side-by-side for Q-Learning vs SARSA.  
- `path_overlay_Th<θ>_Scan<δ>.png` — greedy paths overlaid on the accumulated occupancy grid.  
- `convergence_diagnostics_Th<θ>_Scan<δ>.png` — episode rewards, path lengths, TD error magnitudes over training.  
- `safety_comparison_qlearning.csv`, `safety_comparison_sarsa.csv` — safety margins vs scan depth (input for RQ3).  
- Publication figures generated by `generate_paper_figures.py`. [file:1]

---

## Results and Appendix

Pre-built PDFs are committed under `results/pdfs/`, so you can inspect the key findings without re-running experiments. [file:1]

- `Appendix.pdf` — paper appendix: extended derivations, full statistical tables, additional figures for *Perception Entropy for Path Planning in Autonomous Rover Navigation*.  
- `results.pdf` — full report: statistical tables, selected CSV previews, all 18 figures.  
- `results_v2.pdf` — compact report: core tables + 6 key publication figures.  
- `Box_Plot.pdf` — safety margin distributions per scan depth (Q-Learning vs SARSA).  
- `Safety_Margins.pdf` — safety margin evolution across 500 episodes.  
- `Fig_rq1_1_QLEARNING.pdf`, `Fig_rq1_1_SARSA.pdf` — cumulative reward convergence across scan depths (RQ1).  
- `Fig_rq1_2_QLEARNING.pdf`, `Fig_rq1_2_SARSA.pdf` — reward vs Perception Entropy, with Spearman $\rho$ (RQ2). [file:1]

Detailed statistical outputs and commentary are in `src/hypothesis_test_results.md`. [file:1]

---

## Configuration Reference

All parameters are declared at the top of `src/main.py`. [file:1]

| Code parameter         | Paper symbol            | Default | Description |
|------------------------|-------------------------|---------|-------------|
| `START_FRAME`          | —                       | 40      | First dataset frame (1-indexed; early frames dropped for pose convergence) |
| `END_FRAME`            | —                       | 230     | Last dataset frame |
| `GOAL_FRAME`           | —                       | 60      | Frame whose pose defines the goal cell |
| `RL_EPISODES`          | $N$                     | 500     | Training episodes per mission |
| `RL_ALPHA`             | $\alpha$               | 0.4     | Learning rate |
| `RL_GAMMA`             | $\gamma$               | 0.95    | Discount factor |
| `MAX_RL_STEPS`         | $T_{\max}$             | 400     | Maximum steps per episode |
| `RL_EPS_START`         | $\varepsilon_0$       | 0.99    | Initial exploration rate |
| `RL_EPS_END`           | $\varepsilon_{\min}$  | 0.10    | Terminal exploration rate |
| `RL_EPS_DECAY`         | —                       | `"auto"` | Per-episode decay so $\varepsilon$ reaches $\varepsilon_{\min}$ at episode $N$ |
| `REWARD_GOAL`          | $\Omega_t$             | 1000.0  | Goal reward |
| `COST_CARDINAL`        | $C_t$                  | 1.0     | Movement cost (cardinal) |
| `COST_DIAGONAL`        | $C_t$                  | 1.0     | Movement cost (diagonal) |
| `PROGRESS_REWARD_SCALE`| $\omega$               | 4       | Progress reward scale factor |
| `VAR_DIST`             | $d_{\text{var}}$       | 1.0     | Distance softening constant (used in tensor implementation) |
| `THETA_DEGREE`         | $\theta$               | 60.0    | FOV cone angle for $V_\varphi$ (degrees) |
| `HAZARD_OBSTACLE`      | $\zeta_{\mathrm{obs}}$ | 10.0    | Hazard multiplier for obstacles |
| `HAZARD_UNKNOWN`       | $\zeta_{\mathrm{unk}}$ | 1.0     | Hazard multiplier for unknown cells |
| `SCAN_DEPTH_THRESHOLD` | $\tau_{\text{scan}}$   | 5       | Scan trigger distance (cells); swept over $\{5,10,15,20\}$ for RQ1/RQ3 |
| `PERCEPTION_RADIUS_M`  | $\rho_{\mathrm{perc}}$ | 1.0     | Base perception radius (metres) |
| `PERCEPTION_FOV_DEG`   | $\varphi_{\text{fov}}$ | 60.0    | Scan FOV (degrees) |
| `REWARD_BLIND`         | —                       | 0.0     | Penalty for leaving dataset coverage (0 to disable) |
| `MAX_MISSION_STEPS`    | —                       | 300     | Maximum physical steps before mission failure |
| `RUN_MODE`             | —                       | `"both"`| `"both"`, `"qlearning"`, or `"sarsa"` |
| `OUTPUT_DIR`           | —                       | `../results` | Output directory (relative to `src/`) |

---

## Reproducibility Notes

- All experiments use shared hyperparameters and evaluation protocols across scan depths and algorithms to ensure fair comparisons. [file:1]  
- Episode-level metrics, convergence criteria (10% / 5% reward stability), and hypothesis tests follow the definitions in the manuscript (Spearman’s $\rho$, Mann–Whitney U, Welch’s $t$-tests with Bonferroni correction where applicable). [file:1]  
- Code, configurations, and random seeds are included in this reproduction package as referenced in the paper. [file:1]

If you later extend the framework (e.g. continuous action spaces, multi-agent coordination, causal perception transfer), consider adding new sections to this `README.md` for those modules while keeping this core description intact.
