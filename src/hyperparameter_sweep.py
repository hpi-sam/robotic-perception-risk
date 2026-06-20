"""
hyperparameter_sweep.py
-----------------------
Grid search over (alpha, gamma, theta) for Q-Learning and SARSA.
Runs sequentially and saves a results table + 27-panel map figure to
results/figures/.

Usage:
    python src/hyperparameter_sweep.py
"""

import itertools
import math
import os
import numpy as np
import matplotlib.pyplot as plt

import main
from main import run_experiment, WORLD_CFG

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "results", "figures")


def run_sweep():
    print("=" * 60)
    print("Starting Hyperparameter Sweep (Grid Search)")
    print("=" * 60)

    alphas = [0.1, 0.5, 0.9]
    gammas = [0.8, 0.95, 0.99]
    thetas = [30.0, 45.0, 60.0]

    combinations = list(itertools.product(alphas, gammas, thetas))
    print(f"Total configurations to test: {len(combinations)}")
    print("Algorithms: Q-Learning, SARSA\n")

    # ── Load dataset ─────────────────────────────────────────────────────────
    pairs = main.discover_pairs(main.DATASET_DIR)
    if not pairs:
        print("No dataset pairs found. Aborting.")
        return

    gt_poses = main.load_groundtruth(main.GT_FILE)
    if gt_poses is None or len(gt_poses) == 0:
        print("No ground truth poses found. Aborting.")
        return

    frame_poses = [main.get_pose(gt_poses, i) for i in range(len(pairs))]

    # ── World dimensions (mirror main.py) ────────────────────────────────────
    start_idx = max(0, main.START_FRAME - 1)
    end_idx   = min(len(pairs), main.END_FRAME)
    valid_positions = [p for p in frame_poses[start_idx:end_idx] if p is not None]
    pos_arr = np.array(valid_positions)

    GRID_EXPANSION_M = main.GRID_EXPANSION_M
    world_x_min = pos_arr[:, 0].min() - GRID_EXPANSION_M
    world_x_max = pos_arr[:, 0].max() + GRID_EXPANSION_M
    world_y_min = pos_arr[:, 1].min() - GRID_EXPANSION_M
    world_y_max = pos_arr[:, 1].max() + GRID_EXPANSION_M

    WORLD_CFG.x_range = (world_x_min, world_x_max)
    WORLD_CFG.z_range = (world_y_min, world_y_max)
    res  = WORLD_CFG.resolution
    cols = int(math.ceil((world_x_max - world_x_min) / res))
    rows = int(math.ceil((world_y_max - world_y_min) / res))
    world_dims = (rows, cols, res, world_x_min, world_y_min)

    start_pos = frame_poses[main.START_FRAME - 1]
    goal_pos  = frame_poses[main.GOAL_FRAME  - 1]

    def to_grid(pose):
        r = int((pose[0] - world_x_min) / res)
        c = int((pose[1] - world_y_min) / res)
        return (r, c)

    global_start_cell = to_grid(start_pos)
    global_goal_cell  = to_grid(goal_pos)

    coverage_mask = None
    if main.USE_ACTIVE_PERCEPTION:
        coverage_mask = main.compute_coverage_mask(
            rows, cols, res, world_x_min, world_y_min,
            gt_poses, main.PERCEPTION_RADIUS_M, main.PERCEPTION_FOV_DEG,
        )

    global_occ, global_conf = None, None
    if main.USE_ACTIVE_PERCEPTION:
        print("\n[PRE-COMPUTING GLOBAL MAP FOR FOG-OF-WAR SIMULATION...]")
        global_occ, global_conf = main.precompute_global_map(
            pairs, frame_poses, start_idx, end_idx, world_dims)
        print("[GLOBAL MAP COMPLETE.]\n")

    calculated_eps_decay = main.RL_EPS_DECAY
    if isinstance(main.RL_EPS_DECAY, str) and main.RL_EPS_DECAY.lower() == "auto":
        calculated_eps_decay = (main.RL_EPS_END / main.RL_EPS_START) ** (1.0 / main.RL_EPISODES)

    # ── Sequential sweep ─────────────────────────────────────────────────────
    all_results   = []
    total_combos  = len(combinations)

    print(f"\n[STARTING] Running {total_combos} combinations SEQUENTIALLY "
          "(Q-Learning + SARSA each)...\n")

    for combo_idx, (alpha, gamma, theta) in enumerate(combinations, 1):
        print(f"\n{'='*60}")
        print(f"  Combination {combo_idx}/{total_combos}: "
              f"Alpha={alpha}, Gamma={gamma}, Theta={theta}")
        print(f"{'='*60}")

        for mode, label in (('qlearning', 'Q-Learning'), ('sarsa', 'SARSA')):
            print(f"\n  [{label.upper()}] Running...")
            res = run_experiment(
                mode=mode, pairs=pairs, gt_poses=gt_poses,
                frame_poses=frame_poses, world_dims=world_dims,
                global_start_cell=global_start_cell,
                global_goal_cell=global_goal_cell,
                coverage_mask=coverage_mask,
                calculated_eps_decay=calculated_eps_decay,
                fig=None, ax=None,
                alpha=alpha, gamma=gamma, theta=theta,
                global_occ=global_occ, global_conf=global_conf,
            )
            res['alpha'] = alpha
            res['gamma'] = gamma
            res['theta'] = theta
            res['algo']  = label
            all_results.append(res)

        q_res = all_results[-2]
        s_res = all_results[-1]
        print(f"\n  [PAIR RESULT] A={alpha} G={gamma} T={theta}")
        print(f"    Q-Learning : {'Reached Exit' if q_res['found'] else 'Did NOT Reach Exit'} "
              f"| Steps: {q_res['agent_steps']} | Reward: {q_res['actual_path_reward']:.2f}")
        print(f"    SARSA      : {'Reached Exit' if s_res['found'] else 'Did NOT Reach Exit'} "
              f"| Steps: {s_res['agent_steps']} | Reward: {s_res['actual_path_reward']:.2f}")

    results_q    = [r for r in all_results if r['algo'] == 'Q-Learning']
    results_sarsa = [r for r in all_results if r['algo'] == 'SARSA']

    # ── Console table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SWEEP COMPLETE. FULL RESULTS TABLE")
    print("=" * 80)
    header = (f"{'Algorithm':<12} | {'Alpha':<6} | {'Gamma':<6} | {'Theta':<6} | "
              f"{'Success':<8} | {'Steps':<6} | {'Frames':<8} | Path Reward")
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['algo']:<12} | {r['alpha']:<6} | {r['gamma']:<6} | {r['theta']:<6} | "
              f"{'Yes' if r['found'] else 'No':<8} | {r['agent_steps']:<6} | "
              f"{r['frames_req']:<8} | {r['actual_path_reward']:.2f}")

    def sort_key(r):
        return (r['found'], r['actual_path_reward'])

    results_q.sort(key=sort_key, reverse=True)
    results_sarsa.sort(key=sort_key, reverse=True)

    for label, results in (("Q-LEARNING", results_q), ("SARSA", results_sarsa)):
        print(f"\n--- TOP 3 {label} CONFIGURATIONS ---")
        for i in range(min(3, len(results))):
            r = results[i]
            print(f"{i+1}. [alpha={r['alpha']}, gamma={r['gamma']}, theta={r['theta']}]")
            print(f"   Reward: {r['actual_path_reward']:.2f} | "
                  f"Status: {'Success' if r['found'] else 'Failed'} | "
                  f"Steps: {r['agent_steps']} | Frames: {r['frames_req']}")

    best_q_config    = (results_q[0]['alpha'],    results_q[0]['gamma'],    results_q[0]['theta'])    if results_q    else None
    best_sarsa_config = (results_sarsa[0]['alpha'], results_sarsa[0]['gamma'], results_sarsa[0]['theta']) if results_sarsa else None

    # ── Visual table ──────────────────────────────────────────────────────────
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('tight')
    ax.axis('off')

    columns   = ('Algorithm', 'Alpha', 'Gamma', 'Theta', 'Success', 'Steps', 'Frames', 'Path Reward', 'Is Best')
    cell_text = []
    for r in all_results:
        is_best = ""
        cfg = (r['alpha'], r['gamma'], r['theta'])
        if r['algo'] == 'Q-Learning' and cfg == best_q_config:
            is_best = "[BEST Q]"
        elif r['algo'] == 'SARSA' and cfg == best_sarsa_config:
            is_best = "[BEST SARSA]"
        cell_text.append([r['algo'], f"{r['alpha']}", f"{r['gamma']}", f"{r['theta']}",
                          'Yes' if r['found'] else 'No',
                          f"{r['agent_steps']}", f"{r['frames_req']}",
                          f"{r['actual_path_reward']:.2f}", is_best])

    table = ax.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    for i, row in enumerate(cell_text):
        if "[BEST" in row[8]:
            for j in range(len(columns)):
                table[(i + 1, j)].set_facecolor('#ffffcc')

    plt.title('Hyperparameter Grid Search Results', fontsize=16, pad=20)
    plt.tight_layout()
    table_path = os.path.join(_OUTPUT_DIR, 'sweep_results_table.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[!] Visual table saved -> {table_path}")

    # ── 27-panel map grid ─────────────────────────────────────────────────────
    print("Generating 27-panel visual comparison plot...")
    fig_maps, axes_maps = plt.subplots(5, 6, figsize=(35, 30))
    axes_flat = axes_maps.flatten()

    for idx, (alpha, gamma, theta) in enumerate(combinations):
        ax_map = axes_flat[idx]
        q_res  = next((r for r in results_q    if r['alpha'] == alpha and r['gamma'] == gamma and r['theta'] == theta), None)
        s_res  = next((r for r in results_sarsa if r['alpha'] == alpha and r['gamma'] == gamma and r['theta'] == theta), None)
        if not q_res or not s_res:
            continue

        occ          = q_res['occ']
        display_grid = np.zeros((*occ.shape, 3))
        display_grid[occ == -1] = [0.7, 0.7, 0.7]
        display_grid[occ ==  0] = [1.0, 1.0, 1.0]
        display_grid[occ ==  1] = [0.0, 0.0, 0.0]

        ax_map.imshow(display_grid, origin='lower')
        ax_map.plot(global_goal_cell[1], global_goal_cell[0],
                    'rx', markersize=8, markeredgewidth=2)

        if q_res['path']:
            qy, qx = zip(*q_res['path'])
            ax_map.plot(qx, qy, 'b-', linewidth=2, alpha=0.7,
                        label=f"Q-Learn ({q_res['actual_path_reward']:.1f})")
        if s_res['path']:
            sy, sx = zip(*s_res['path'])
            ax_map.plot(sx, sy, 'r-', linewidth=2, alpha=0.7,
                        label=f"SARSA ({s_res['actual_path_reward']:.1f})")

        cfg = (alpha, gamma, theta)
        title_str = f"A:{alpha} G:{gamma} T:{theta}\nQ:{q_res['actual_path_reward']:.1f} | S:{s_res['actual_path_reward']:.1f}"
        if cfg == best_q_config and cfg == best_sarsa_config:
            ax_map.set_facecolor('#ffebcc')
            title_str = "[BEST BOTH]\n" + title_str
        elif cfg == best_q_config:
            ax_map.set_facecolor('#e6f2ff')
            title_str = "[BEST Q]\n" + title_str
        elif cfg == best_sarsa_config:
            ax_map.set_facecolor('#ffe6e6')
            title_str = "[BEST SARSA]\n" + title_str

        ax_map.set_title(title_str, fontsize=10)
        ax_map.legend(fontsize=8, loc='upper left')
        ax_map.axis('off')

    for idx in range(len(combinations), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    maps_path = os.path.join(_OUTPUT_DIR, 'sweep_results_maps.png')
    fig_maps.savefig(maps_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[!] 27-panel map grid saved -> {maps_path}")


if __name__ == "__main__":
    run_sweep()
