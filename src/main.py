"""
main.py
-------
Active Perception RL Navigation Pipeline.

Features:
- Processes depth frames via in-episode vision-cone ray-casting.
- RL agents (Q-Learning and SARSA) train on the incrementally revealed map.
- Normalised risk calculation [0-1] using directional distance sweeps.
- Easily editable global configuration.
"""

import os
import math
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image

# Local imports
from image_loader import discover_pairs, load_rgb, load_depth
from pose_loader import load_groundtruth, get_pose, estimate_heading, transform_points
from grid_mapper import (
    CameraIntrinsics,
    GridConfig,
    back_project_clipped,
    depth_to_occupancy_grid_world,
    accumulate_with_confidence
)
from confidence_mapper import ConfidenceAccumulator
from unbounded_metrics import compute_directional_distances, compute_unbounded_risk
from rl_environment import GridEnv
from rl_agent import RLAgent
from rl_trainer import train_online, compute_epsilon_decay

# =================================================================
# ====== CONFIGURATION ======
# =================================================================

# --- Dataset ---
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "ROVER_data", "realsense_D435i")
GT_FILE     = os.path.join(os.path.dirname(SCRIPT_DIR), "ROVER_data", "groundtruth.txt")
OUTPUT_DIR  = os.path.join(os.path.dirname(SCRIPT_DIR), "results")

# --- Iteration Configuration ---
START_FRAME = 40          # 1-indexed first frame to load (starts at 40 for pose convergence)
END_FRAME   = 230         # Max frame index to load
GOAL_FRAME  = 60          # 1-indexed frame whose pose defines the goal location

# --- RL Hyperparameters ---
RL_EPISODES  = 500        # Training episodes per mission
RL_ALPHA     = 0.4        # Learning rate alpha
RL_GAMMA     = 0.95       # Discount factor gamma
MAX_RL_STEPS = 400        # Hard step cutoff per episode

# --- Exploration Strategy ---
RL_EPS_START = 0.99       # Initial exploration rate
RL_EPS_END   = 0.1        # Terminal exploration rate
RL_EPS_DECAY = "auto"     # "auto" = compute mathematically, or provide a float (e.g. 0.999)

# --- Goal & Progress Rewards ---
REWARD_GOAL           = 1000.0   # Reward for reaching the goal
COST_CARDINAL         = 1.0      # Movement cost for cardinal steps
COST_DIAGONAL         = 1.0      # Movement cost for diagonal steps
CARROT_REWARD_ENABLED = True     # Enable dense progress reward toward goal
PROGRESS_REWARD_SCALE = 4        # Scale factor for progress reward

# --- Normalised Risk ---
USE_NORMALIZED_RISK = True   # True = [0,1] bounded; False = raw unbounded
VAR_DIST            = 1.0    # Distance scaling for risk formula
THETA_DEGREE        = 60.0   # FOV cone angle θ (degrees)
HAZARD_OBSTACLE     = 10.0   # Hazard weight when d_dir stops at an obstacle
HAZARD_UNKNOWN      = 1.0    # Hazard weight when d_dir stops at unknown terrain

# --- Active Perception ---
USE_ACTIVE_PERCEPTION = True    # Use vision-cone frame requests during navigation
PERCEPTION_RADIUS_M   = 1.0     # Scan radius around the agent (metres)
PERCEPTION_FOV_DEG    = 60.0    # Field-of-view for frame selection (degrees)
REWARD_BLIND          = 0.0     # Penalty for stepping outside dataset coverage
MAX_MISSION_STEPS     = 300     # Max physical steps before mission failure
SCAN_DEPTH_THRESHOLD  = 5       # d_dir threshold (cells): 5, 10, 15, or 20
RUN_MODE              = "both"  # "both", "qlearning", or "sarsa"

# =================================================================

INTRINSICS = CameraIntrinsics(fx=610.0, fy=610.0, cx=320.0, cy=240.0)

GRID_EXPANSION_M = 1.0

WORLD_CFG = GridConfig(
    resolution=0.05,
    near_clip=0.5,
    obstacle_hit_ratio=0.2
)


# =================================================================
# ====== HELPER FUNCTIONS ======
# =================================================================

def compute_coverage_mask(rows, cols, res, x_min, y_min, gt_poses, radius_m, fov_deg):
    """Pre-calculates an 8-layer mask of where the dataset has valid perception coverage."""
    mask = np.zeros((rows, cols, 8), dtype=bool)
    half_fov  = fov_deg / 2.0
    rad_cells = int(radius_m / res)

    from rl_environment import ACTIONS
    dir_angles = [math.atan2(dr, dc) for dr, dc in ACTIONS]

    print(f"  [Pre-computing Perception Mask] Radius: {radius_m}m ({rad_cells} cells)")

    for i in range(len(gt_poses) - 1):
        p1, p2 = gt_poses[i], gt_poses[i + 1]
        px, py = p1[0], p1[1]
        p_h    = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

        pc = int((px - x_min) / res)
        pr = int((py - y_min) / res)
        if pr < 0 or pr >= rows or pc < 0 or pc >= cols:
            continue

        for d_idx, d_angle in enumerate(dir_angles):
            angle_diff = abs((p_h - d_angle + math.pi) % (2 * math.pi) - math.pi)
            if math.degrees(angle_diff) <= half_fov:
                r_start = max(0, pr - rad_cells)
                r_end   = min(rows, pr + rad_cells + 1)
                c_start = max(0, pc - rad_cells)
                c_end   = min(cols, pc + rad_cells + 1)
                for rr in range(r_start, r_end):
                    for cc in range(c_start, c_end):
                        if (rr - pr) ** 2 + (cc - pc) ** 2 <= rad_cells ** 2:
                            mask[rr, cc, d_idx] = True
    return mask


def precompute_global_map(pairs, frame_poses, start_idx, end_idx, world_dims):
    """Builds the complete ground-truth map from all dataset frames (Fog of War simulation)."""
    rows, cols, res, x_min, y_min = world_dims

    global_occ  = np.full((rows, cols), -1, dtype=np.int8)
    global_conf = np.zeros((rows, cols), dtype=np.float32)
    conf_acc    = ConfidenceAccumulator(rows=rows, cols=cols)

    prev_pos     = None
    last_heading = 0.0

    for i in range(start_idx, end_idx):
        if i >= len(pairs):
            break
        rgb_path, depth_path = pairs[i]
        pos = frame_poses[i]
        if pos is None:
            continue

        if prev_pos is not None:
            if math.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1]) > 0.01:
                last_heading = estimate_heading(prev_pos, pos)
        heading = last_heading

        depth     = load_depth(depth_path)
        cam_pts   = back_project_clipped(depth, INTRINSICS, near_clip=WORLD_CFG.near_clip, far_clip=10.0)
        world_pts = transform_points(cam_pts, pos, heading)
        occ_grid_frame, _, _, frame_conf, _ = depth_to_occupancy_grid_world(world_pts, WORLD_CFG)

        conf_acc.update_world(world_pts, WORLD_CFG)
        final_conf_frame = frame_conf * conf_acc.confidence_grid()

        global_occ, global_conf = accumulate_with_confidence(
            global_occ, global_conf, occ_grid_frame, final_conf_frame,
            free_clear_threshold=0.3
        )
        prev_pos = pos

    return global_occ, global_conf


def apply_vision_cone_raycast(agent_pos, planner_dir, radius_m, fov_deg,
                               global_occ, global_conf, agent_occ, agent_conf, res):
    """Vectorised ray-casting: reveals global map cells within the agent's FOV cone."""
    from rl_environment import ACTIONS
    if agent_pos is None:
        return agent_occ, agent_conf

    r_a, c_a   = agent_pos
    radius_c   = int(radius_m / res)
    rows, cols = global_occ.shape

    num_rays = max(8, int(math.ceil((fov_deg / 360.0) * (2 * math.pi * radius_c * 2))))

    if fov_deg >= 360.0:
        start_angle, end_angle = 0.0, 2 * math.pi
    else:
        heading_angle = math.atan2(ACTIONS[planner_dir][0], ACTIONS[planner_dir][1])
        half_fov      = math.radians(fov_deg / 2.0)
        start_angle   = heading_angle - half_fov
        end_angle     = heading_angle + half_fov

    angles   = np.linspace(start_angle, end_angle, num_rays)
    dr_steps = np.sin(angles)
    dc_steps = np.cos(angles)

    active_mask = np.ones(num_rays, dtype=bool)

    for step in range(1, radius_c + 1):
        if not np.any(active_mask):
            break

        rs = np.round(r_a + step * dr_steps).astype(int)
        cs = np.round(c_a + step * dc_steps).astype(int)

        in_bounds    = (rs >= 0) & (rs < rows) & (cs >= 0) & (cs < cols)
        active_mask &= in_bounds

        if not np.any(active_mask):
            break

        valid_indices    = np.where(active_mask)[0]
        curr_rs, curr_cs = rs[valid_indices], cs[valid_indices]

        agent_occ[curr_rs, curr_cs]  = global_occ[curr_rs, curr_cs]
        agent_conf[curr_rs, curr_cs] = global_conf[curr_rs, curr_cs]

        hit_obstacle = (global_occ[curr_rs, curr_cs] == 1)
        active_mask[valid_indices[hit_obstacle]] = False

    return agent_occ, agent_conf


def _ensure_passable(occ: np.ndarray, r: int, c: int, max_radius: int = 15):
    """Spirals outward from (r, c) to find the nearest free (occ == 0) cell."""
    rows, cols = occ.shape
    if 0 <= r < rows and 0 <= c < cols and occ[r, c] == 0:
        return (r, c)

    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) != radius and abs(dc) != radius:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and occ[nr, nc] == 0:
                    return (nr, nc)
    return (r, c)


def _smooth(data, window=50):
    """Simple moving average."""
    if len(data) < window:
        window = max(1, len(data))
    cumsum = np.cumsum(data)
    cumsum = np.insert(cumsum, 0, 0)
    return (cumsum[window:] - cumsum[:-window]) / window


def _update_plot_live(ax, occ, goal, path_sarsa, path_ql, frame_idx):
    ax.clear()
    display_grid = np.zeros((*occ.shape, 3))
    display_grid[occ == -1] = [0.7, 0.7, 0.7]
    display_grid[occ == 0]  = [1.0, 1.0, 1.0]
    display_grid[occ == 1]  = [0.0, 0.0, 0.0]

    ax.imshow(display_grid, origin='lower')
    ax.set_title(f"Active Perception RL Navigation (Scan #{frame_idx})", fontweight="bold")
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")
    ax.plot(goal[1], goal[0], marker='*', color='gold', markersize=15, label='Goal')

    if path_sarsa:
        ys, xs = zip(*path_sarsa)
        if len(path_sarsa) > 1:
            ax.plot(xs, ys, 'm-', linewidth=3, alpha=0.8, label='SARSA Path')
        ax.plot(xs[-1], ys[-1], 'm*', markersize=12)

    if path_ql:
        yq, xq = zip(*path_ql)
        if len(path_ql) > 1:
            ax.plot(xq, yq, 'g--', linewidth=3, alpha=0.8, label='Q-Learning Path')
        ax.plot(xq[-1], yq[-1], 'g*', markersize=12)

    ax.legend(loc='upper right')


# =================================================================
# ====== CORE EXPERIMENT ======
# =================================================================

def run_experiment(mode, pairs, gt_poses, frame_poses, world_dims,
                   global_start_cell, global_goal_cell, coverage_mask,
                   calculated_eps_decay, fig, ax,
                   alpha=RL_ALPHA, gamma=RL_GAMMA, theta=THETA_DEGREE,
                   global_occ=None, global_conf=None):
    """Runs a complete navigation mission for a single RL agent type."""
    rows, cols, res, world_x_min, world_y_min = world_dims

    accumulated_occ   = np.full((rows, cols), -1, dtype=np.int8)
    accumulated_conf  = np.zeros((rows, cols), dtype=np.float32)
    global_ray_counts = np.zeros((rows, cols), dtype=np.int32)
    conf_acc = ConfidenceAccumulator(rows=rows, cols=cols)

    agent   = RLAgent(n_actions=8, alpha=alpha, gamma=gamma)
    env     = None
    epsilon = RL_EPS_START

    path_lengths     = []
    is_found         = False
    found_frames_req = -1
    gif_frames       = []

    current_agent_cell = global_start_cell
    traveled_path      = [current_agent_cell]

    best_overall_path   = []
    best_overall_reward = -float('inf')
    final_greedy_reward = 0.0
    final_entropy       = 0.0

    total_scans_mission = 0
    sum_tension_mission = 0.0
    mission_avg_tension = 0.0

    global_first_ep    = -1
    processed_count    = 0
    agent_steps        = 0
    d_dir_stop_type    = None
    actual_path_reward = 0.0
    agent_heading      = 0

    print(f"\n{'='*40}")
    print(f"=== STARTING {mode.upper()} MISSION ===")
    print(f"{'='*40}")

    # --- Initial FOV scan ---
    dynamic_radius_m = PERCEPTION_RADIUS_M + (SCAN_DEPTH_THRESHOLD * res)
    vframes_radius   = int(dynamic_radius_m / res)
    if USE_ACTIVE_PERCEPTION:
        print(f"\n[{mode.upper()} INITIAL SCAN: {PERCEPTION_FOV_DEG}deg FOV, depth {dynamic_radius_m:.2f}m]")
        accumulated_occ, accumulated_conf = apply_vision_cone_raycast(
            current_agent_cell, agent_heading, dynamic_radius_m, PERCEPTION_FOV_DEG,
            global_occ, global_conf, accumulated_occ, accumulated_conf, res
        )
        processed_count += 1
        directional_distances, d_dir_stop_type = compute_directional_distances(accumulated_occ)
        directional_risk, vframes_current, vframes_base = compute_unbounded_risk(
            directional_distances, accumulated_occ, d_dir_stop_type=d_dir_stop_type,
            var_dist=VAR_DIST, theta_deg=theta, base_deg=THETA_DEGREE,
            hazard_obstacle=HAZARD_OBSTACLE, hazard_unknown=HAZARD_UNKNOWN,
            normalized=USE_NORMALIZED_RISK, radius=vframes_radius
        )
        env = GridEnv(
            accumulated_occ, directional_risk, directional_distances, global_ray_counts,
            global_goal_cell, stop_type=d_dir_stop_type,
            reward_goal=REWARD_GOAL, reward_collision=-20.0,
            progress_weight=PROGRESS_REWARD_SCALE, use_carrot=CARROT_REWARD_ENABLED,
            cost_cardinal=COST_CARDINAL, cost_diagonal=COST_DIAGONAL,
            max_steps_per_ep=MAX_RL_STEPS, coverage_mask=coverage_mask,
            reward_blind=REWARD_BLIND
        )

    safe_start_cell = _ensure_passable(accumulated_occ, current_agent_cell[0], current_agent_cell[1])
    safe_goal_cell  = _ensure_passable(accumulated_occ, global_goal_cell[0], global_goal_cell[1])

    scan_history   = [np.zeros((rows, cols, 8), dtype=bool)]
    last_plot_time = [0.0]

    def reset_scan_history():
        scan_history[0][:] = False

    def _scan_callback(agent_pos, direction):
        nonlocal accumulated_occ, accumulated_conf, directional_distances
        nonlocal directional_risk, d_dir_stop_type, env, processed_count
        nonlocal total_scans_mission, sum_tension_mission, mission_avg_tension

        r, c = agent_pos
        if scan_history[0][r, c, direction]:
            return

        if USE_ACTIVE_PERCEPTION and global_occ is not None:
            scan_history[0][r, c, direction] = True
            print(f"      [Scan] Pos: {agent_pos} | Dir: {direction} | Depth: {dynamic_radius_m:.2f}m")
            accumulated_occ, accumulated_conf = apply_vision_cone_raycast(
                agent_pos, direction, dynamic_radius_m, PERCEPTION_FOV_DEG,
                global_occ, global_conf, accumulated_occ, accumulated_conf, res
            )
            processed_count += 1
            directional_distances, d_dir_stop_type = compute_directional_distances(accumulated_occ)
            directional_risk, fresh_vframes_curr, fresh_vframes_base = compute_unbounded_risk(
                directional_distances, accumulated_occ, d_dir_stop_type=d_dir_stop_type,
                var_dist=VAR_DIST, theta_deg=theta, base_deg=THETA_DEGREE,
                hazard_obstacle=HAZARD_OBSTACLE, hazard_unknown=HAZARD_UNKNOWN,
                normalized=USE_NORMALIZED_RISK, training_mode=False, radius=vframes_radius
            )
            if env is not None:
                env.update_map(accumulated_occ, directional_risk, directional_distances,
                               global_ray_counts, stop_type=d_dir_stop_type)

            if ax is not None and (time.time() - last_plot_time[0]) > 0.5:
                _update_plot_live(ax, accumulated_occ, global_goal_cell, [], [], processed_count)
                last_plot_time[0] = time.time()
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                gif_frames.append(Image.open(buf))

            v_curr = float(fresh_vframes_curr[agent_pos[0], agent_pos[1], direction])
            v_base = float(fresh_vframes_base[agent_pos[0], agent_pos[1], direction])
            local_tension = abs(v_base - v_curr) / (v_curr + 1e-6)

            total_scans_mission += 1
            sum_tension_mission += local_tension
            mission_avg_tension  = sum_tension_mission / total_scans_mission
            print(f"      [Tension] Local: {local_tension:.2f} | Mission Avg: {mission_avg_tension:.2f}")
            return local_tension

    print(f"\n[{mode.upper()}] Starting training ({RL_EPISODES} episodes)...")
    epsilon = RL_EPS_START

    (_greedy, _best_ep_path, _max_rew, epsilon,
     first_ep, first_st, success_count, _greedy_rew, full_history) = train_online(
        env=env, agent=agent, start=safe_start_cell, goal=safe_goal_cell,
        n_episodes=RL_EPISODES, epsilon=epsilon,
        epsilon_min=RL_EPS_END, epsilon_decay=calculated_eps_decay,
        mode=mode, start_dir=agent_heading,
        scan_callback=_scan_callback if USE_ACTIVE_PERCEPTION else None,
        scan_depth_threshold=SCAN_DEPTH_THRESHOLD,
        episode_offset=0, total_episodes=RL_EPISODES,
        reset_scan_history=reset_scan_history
    )

    final_greedy_reward = _greedy_rew
    final_entropy       = agent.compute_exploration_entropy()

    display_path = _greedy
    if _greedy_rew < REWARD_GOAL * 0.9 and _max_rew >= REWARD_GOAL * 0.9 and _best_ep_path:
        display_path = _best_ep_path
        print(f"      [Display] Greedy missed goal (r={_greedy_rew:.1f}); "
              f"using best training path (r={_max_rew:.1f})")

    final_greedy_safety = 0.0
    if display_path:
        from rl_environment import compute_distance_transform
        dist_map    = compute_distance_transform(accumulated_occ)
        safe_values = [dist_map[r, c] for r, c in display_path]
        final_greedy_safety = np.mean(safe_values) if safe_values else 0.0

    global_first_ep = first_ep
    if global_first_ep > 0:
        print(f"      [Convergence] Goal first reached at episode {global_first_ep}")

    print(f"[{mode.upper()}] Training done. Best path reward: {final_greedy_reward:.2f}")

    if _max_rew > best_overall_reward:
        best_overall_reward = _max_rew
        best_overall_path   = _best_ep_path

    path_lengths.append(len(display_path) if display_path else 0)

    if ax is not None:
        s_path = display_path if mode == 'sarsa' else []
        q_path = display_path if mode == 'qlearning' else []
        _update_plot_live(ax, accumulated_occ, global_goal_cell, s_path, q_path, processed_count)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        gif_frames.append(Image.open(buf))

    print(f"      [Final] Goal hits during training: {success_count}")

    # --- Follow the greedy path physically ---
    if display_path:
        print(f"      [{mode.upper()}] Following path ({len(display_path)} steps)...")
        for i in range(len(display_path) - 1):
            p1, p2  = display_path[i], display_path[i + 1]
            step_dr = p2[0] - p1[0]
            step_dc = p2[1] - p1[1]
            current_agent_cell = p2
            traveled_path.append(current_agent_cell)
            agent_steps += 1

            step_cost = COST_DIAGONAL if (abs(step_dr) + abs(step_dc) == 2) else COST_CARDINAL

            if step_dr != 0 or step_dc != 0:
                from rl_environment import ACTIONS
                for a_idx, (dr, dc) in enumerate(ACTIONS):
                    if dr == step_dr and dc == step_dc:
                        agent_heading = a_idx
                        break

            r_val = directional_risk[p2[0], p2[1], agent_heading]
            actual_path_reward -= (step_cost + r_val)

            if current_agent_cell == safe_goal_cell:
                is_found = True
                actual_path_reward += REWARD_GOAL
                found_frames_req = processed_count
                break
    else:
        print(f"      [!] {mode} could not find any path after training.")

    if agent_steps >= MAX_MISSION_STEPS:
        print(f"\n[!] Mission terminated: reached MAX_MISSION_STEPS ({MAX_MISSION_STEPS})")

    # --- Save mission GIF ---
    if gif_frames:
        alg_name = "Q-Learning" if mode == "qlearning" else "SARSA"
        gif_path = os.path.join(
            _alg_dir(alg_name),
            f"mission_{mode.upper()}_Th{int(THETA_DEGREE)}_Scan{SCAN_DEPTH_THRESHOLD}.gif"
        )
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=200, loop=0)
        print(f"      [GIF] Saved to {gif_path}")

    return {
        'mode':                  mode,
        'path':                  traveled_path,
        'best_path':             best_overall_path,
        'actual_path_reward':    actual_path_reward,
        'occ':                   accumulated_occ,
        'risk':                  directional_risk,
        'frames_req':            found_frames_req,
        'found':                 is_found,
        'agent':                 agent,
        'path_lengths':          path_lengths,
        'agent_steps':           agent_steps,
        'final_greedy_reward':   final_greedy_reward,
        'entropy':               final_entropy,
        'total_scans':           total_scans_mission,
        'avg_tension':           mission_avg_tension,
        'history':               full_history,
        'first_goal_ep':         global_first_ep,
        'greedy_path_len':       len(display_path) if display_path else 0,
        'greedy_path_safety':    final_greedy_safety,
    }


# =================================================================
# ====== EXPORT & DIAGNOSTICS ======
# =================================================================

def _alg_dir(name: str) -> str:
    """results/<Algorithm>/Scan_<N>/ for per-run outputs."""
    path = os.path.join(OUTPUT_DIR, name, f"Scan_{SCAN_DEPTH_THRESHOLD}")
    os.makedirs(path, exist_ok=True)
    return path


def _fig_dir() -> str:
    """results/figures/ for cross-algorithm outputs."""
    path = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(path, exist_ok=True)
    return path


def export_mission_data(name, res_obj):
    """Save episode history, Q-table, and update the master summary CSV."""
    import csv
    import pandas as pd

    run_dir    = _alg_dir(name)
    clean_name = name.replace("-", "").upper()
    hist_filename = os.path.join(
        run_dir,
        f"{clean_name}_Th{int(THETA_DEGREE)}_Scan{SCAN_DEPTH_THRESHOLD}_History.csv"
    )
    hist = res_obj['history']

    with open(hist_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["Metric"] + [f"Episode_{i+1}" for i in range(len(hist['reward']))]
        writer.writerow(header)
        writer.writerow(["Reward"]             + [f"{v:.2f}"  for v in hist['reward']])
        writer.writerow(["Path_Length"]        + hist['length'])
        writer.writerow(["TD_Error"]           + [f"{v:.4f}" for v in hist['td_error']])
        writer.writerow(["Goal_Reached"]       + hist['success'])
        writer.writerow(["Avg_Safety_Margin"]  + [f"{v:.4f}" for v in hist['safety']])
        writer.writerow(["Avg_Tension"]        + [f"{v:.4f}" for v in hist['tension']])
        writer.writerow(["Exploration_Entropy"] + [f"{v:.4f}" for v in hist['entropy']])
    print(f"      [CSV] History -> {hist_filename}")

    qtable_filename = os.path.join(
        run_dir,
        f"{clean_name}_Qtable_Scan{SCAN_DEPTH_THRESHOLD}.csv"
    )
    with open(qtable_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "dir", "d_h_bin", "d_l_bin", "d_r_bin",
                          "action", "q_value", "sa_visit_count", "tension"])
        _agent = res_obj['agent']
        for (state, action), q_value in sorted(_agent.Q.items()):
            row_s, col_s, dir_s, d_h, d_l, d_r = state
            sa_visits  = _agent.sa_visit_counts.get((state, action), 0)
            sa_tension = _agent.get_sa_tension(state, action)
            writer.writerow([row_s, col_s, dir_s, d_h, d_l, d_r, action,
                              f"{q_value:.6f}", sa_visits, f"{sa_tension:.6f}"])
    print(f"      [CSV] Q-table ({len(res_obj['agent'].Q)} entries) -> {qtable_filename}")

    summary_filename = os.path.join(OUTPUT_DIR, "mission_summary.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    new_row = {
        'Algorithm':                    name,
        'SCAN_DEPTH_THRESHOLD':         SCAN_DEPTH_THRESHOLD,
        'THETA_DEGREE':                 THETA_DEGREE,
        'PERCEPTION_RADIUS_CELLS':      int(PERCEPTION_RADIUS_M / WORLD_CFG.resolution),
        'MAX_RL_STEPS':                 MAX_RL_STEPS,
        'Total_Scans':                  res_obj['total_scans'],
        'Exploration_Entropy':          f"{res_obj['entropy']:.4f}",
        'Final_Greedy_Reward':          f"{res_obj['final_greedy_reward']:.2f}",
        'Convergence_Episode':          res_obj['first_goal_ep'] if res_obj['first_goal_ep'] > 0 else "N/A",
        'Final_Greedy_Path_Length':     res_obj['greedy_path_len'],
        'Avg_Safety_Margin_Final_Path': f"{res_obj['greedy_path_safety']:.4f}",
    }

    try:
        cols_order = list(new_row.keys())
        df_old = pd.read_csv(summary_filename) if os.path.exists(summary_filename) \
                 else pd.DataFrame(columns=cols_order)
        df_new = pd.DataFrame([new_row])
        df_new['SCAN_DEPTH_THRESHOLD']    = df_new['SCAN_DEPTH_THRESHOLD'].astype(float)
        df_new['THETA_DEGREE']            = df_new['THETA_DEGREE'].astype(float)
        df_new['PERCEPTION_RADIUS_CELLS'] = df_new['PERCEPTION_RADIUS_CELLS'].astype(int)

        if not df_old.empty:
            df_old['SCAN_DEPTH_THRESHOLD'] = df_old['SCAN_DEPTH_THRESHOLD'].astype(float)
            df_old['THETA_DEGREE']         = df_old['THETA_DEGREE'].astype(float)
            if 'PERCEPTION_RADIUS_CELLS' in df_old.columns:
                df_old['PERCEPTION_RADIUS_CELLS'] = df_old['PERCEPTION_RADIUS_CELLS'].astype(int)

        df_final = (
            pd.concat([df_old, df_new], ignore_index=True)
              .drop_duplicates(
                  subset=['Algorithm', 'SCAN_DEPTH_THRESHOLD', 'THETA_DEGREE', 'PERCEPTION_RADIUS_CELLS'],
                  keep='last'
              )
        )
        df_final.to_csv(summary_filename, index=False)
        print(f"      [CSV] Summary updated -> {summary_filename}")
    except Exception as e:
        print(f"      [Warning] Could not update summary: {e}")


def _plot_convergence_diagnostics(results_dict: dict):
    """Plot convergence diagnostics (rewards, path lengths, TD errors) for all agents."""
    agents_to_plot = []
    if 'qlearning' in results_dict:
        r = results_dict['qlearning']
        agents_to_plot.append(("Q-Learning", r['agent'], r['path_lengths'], "#2ecc71", r))
    if 'sarsa' in results_dict:
        r = results_dict['sarsa']
        agents_to_plot.append(("SARSA", r['agent'], r['path_lengths'], "#e74c3c", r))

    if not agents_to_plot:
        return

    n_agents = len(agents_to_plot)
    fig, axes = plt.subplots(3, n_agents, figsize=(6 * n_agents, 12), squeeze=False)
    fig.suptitle(
        f"Convergence Diagnostics (theta={int(THETA_DEGREE)}, Scan={SCAN_DEPTH_THRESHOLD})",
        fontsize=16, fontweight="bold"
    )

    for col, (name, agent, path_lens, color, res_obj) in enumerate(agents_to_plot):
        ax = axes[0][col]
        rewards = agent.episode_rewards
        if rewards:
            ax.plot(range(len(rewards)), rewards, color=color, linewidth=1.5, label='Raw Reward')
            ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='--')
            ax.legend(loc='lower right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{name} - Episode Rewards", fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.grid(True, alpha=0.3)

        ax = axes[1][col]
        if path_lens:
            chunks = list(range(1, len(path_lens) + 1))
            ax.bar(chunks, path_lens, color=color, alpha=0.7, edgecolor='black')
            for i, v in enumerate(path_lens):
                ax.text(chunks[i], v + 0.5, str(v), ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{name} - Greedy Path Length", fontweight="bold")
        ax.set_xlabel("Chunk #")
        ax.set_ylabel("Path Length (cells)")
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[2][col]
        td_errs = agent.td_errors
        if td_errs:
            ax.plot(range(len(td_errs)), td_errs, color=color, linewidth=1.0,
                    label='Raw TD Error', alpha=0.8)
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{name} - |TD Error| per Update", fontweight="bold")
        ax.set_xlabel("Update Step")
        ax.set_ylabel("|TD Error|")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    diag_name = os.path.join(
        _fig_dir(),
        f"convergence_diagnostics_Th{int(THETA_DEGREE)}_Scan{SCAN_DEPTH_THRESHOLD}.png"
    )
    plt.savefig(diag_name, dpi=300)
    print(f"Saved convergence diagnostics -> {diag_name}")

    for name, agent, path_lens, color, res_obj in agents_to_plot:
        print(f"\n{'='*60}")
        print(f"  {name} Convergence Summary")
        print(f"{'='*60}")
        rewards = agent.episode_rewards
        if rewards:
            q1 = rewards[:len(rewards) // 4] if len(rewards) >= 4 else rewards
            q4 = rewards[-(len(rewards) // 4):] if len(rewards) >= 4 else rewards
            print(f"  Episodes trained       : {len(rewards)}")
            print(f"  Reward (first 25%)     : mean={np.mean(q1):.2f}  std={np.std(q1):.2f}")
            print(f"  Reward (last  25%)     : mean={np.mean(q4):.2f}  std={np.std(q4):.2f}")
            print(f"  Reward improvement     : {np.mean(q4) - np.mean(q1):+.2f}")
        if path_lens:
            print(f"  Path lengths by chunk  : {path_lens}")
            print(f"  Final path length      : {path_lens[-1]}")
        print(f"  Final Greedy Reward    : {res_obj['final_greedy_reward']:.2f}")
        print(f"  Exploration Entropy    : {res_obj['entropy']:.4f}")
        td_errs = agent.td_errors
        if td_errs:
            td1 = td_errs[:len(td_errs) // 4] if len(td_errs) >= 4 else td_errs
            td4 = td_errs[-(len(td_errs) // 4):] if len(td_errs) >= 4 else td_errs
            reduction = (1 - np.mean(td4) / max(np.mean(td1), 1e-9)) * 100
            print(f"  |TD| (first 25%)       : mean={np.mean(td1):.4f}")
            print(f"  |TD| (last  25%)       : mean={np.mean(td4):.4f}")
            print(f"  |TD| reduction         : {reduction:.1f}%")
        print(f"{'='*60}")


# =================================================================
# ====== MAIN PIPELINE ======
# =================================================================

def run_rl_pipeline():
    print("#" * 80)
    print("  Active Perception RL Navigation Pipeline")
    print(f"  RL Training : {RL_EPISODES} episodes")
    print(f"  Risk config : VAR_DIST={VAR_DIST}, theta={THETA_DEGREE}deg, normalised={USE_NORMALIZED_RISK}")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print("#" * 80 + "\n")

    try:
        pairs    = discover_pairs(DATASET_DIR)
        gt_poses = load_groundtruth(GT_FILE)
    except Exception as e:
        print(f"Dataset load error: {e}")
        return

    n_total   = len(pairs)
    start_idx = max(0, START_FRAME - 1)
    end_idx   = min(n_total, END_FRAME)

    frame_poses = [get_pose(gt_poses, i) for i in range(n_total)]

    valid_positions = [p for p in frame_poses[start_idx:end_idx] if p is not None]
    pos_arr     = np.array(valid_positions)
    world_x_min = pos_arr[:, 0].min() - GRID_EXPANSION_M
    world_x_max = pos_arr[:, 0].max() + GRID_EXPANSION_M
    world_y_min = pos_arr[:, 1].min() - GRID_EXPANSION_M
    world_y_max = pos_arr[:, 1].max() + GRID_EXPANSION_M

    WORLD_CFG.x_range = (world_x_min, world_x_max)
    WORLD_CFG.z_range = (world_y_min, world_y_max)
    res  = WORLD_CFG.resolution
    cols = int(math.ceil((world_x_max - world_x_min) / res))
    rows = int(math.ceil((world_y_max - world_y_min) / res))

    first_pos         = valid_positions[0]
    global_start_cell = (
        max(0, min(int((first_pos[1] - world_y_min) / res), rows - 1)),
        max(0, min(int((first_pos[0] - world_x_min) / res), cols - 1)),
    )

    goal_pose = get_pose(gt_poses, max(0, GOAL_FRAME - 1)) or valid_positions[-1]
    global_goal_cell = (
        max(0, min(int((goal_pose[1] - world_y_min) / res), rows - 1)),
        max(0, min(int((goal_pose[0] - world_x_min) / res), cols - 1)),
    )

    coverage_mask = None
    if USE_ACTIVE_PERCEPTION:
        dynamic_radius_m = PERCEPTION_RADIUS_M + (SCAN_DEPTH_THRESHOLD * res)
        print(f"  [Perception Mask] dynamic radius: {dynamic_radius_m:.2f}m")
        coverage_mask = compute_coverage_mask(
            rows, cols, res, world_x_min, world_y_min,
            gt_poses, dynamic_radius_m, PERCEPTION_FOV_DEG
        )

    if isinstance(RL_EPS_DECAY, str) and RL_EPS_DECAY.lower() == "auto":
        calculated_eps_decay = compute_epsilon_decay(RL_EPS_START, RL_EPS_END, 1, RL_EPISODES)
    else:
        calculated_eps_decay = float(RL_EPS_DECAY)

    fig, ax    = plt.subplots(figsize=(10, 8))
    plt.close(fig)
    world_dims = (rows, cols, res, world_x_min, world_y_min)
    global_occ, global_conf = (
        precompute_global_map(pairs, frame_poses, start_idx, end_idx, world_dims)
        if USE_ACTIVE_PERCEPTION else (None, None)
    )

    results_list  = []
    results_ql    = None
    results_sarsa = None

    if RUN_MODE in ["both", "qlearning"]:
        results_ql = run_experiment(
            'qlearning', pairs, gt_poses, frame_poses, world_dims,
            global_start_cell, global_goal_cell, coverage_mask,
            calculated_eps_decay, fig, ax,
            global_occ=global_occ, global_conf=global_conf
        )
        results_list.append(results_ql)
        export_mission_data("Q-Learning", results_ql)

    if RUN_MODE in ["both", "sarsa"]:
        results_sarsa = run_experiment(
            'sarsa', pairs, gt_poses, frame_poses, world_dims,
            global_start_cell, global_goal_cell, coverage_mask,
            calculated_eps_decay, fig, ax,
            global_occ=global_occ, global_conf=global_conf
        )
        results_list.append(results_sarsa)
        export_mission_data("SARSA", results_sarsa)

    print("\n" + "=" * 60 + "\n=== FINAL RESULTS ===")
    for r in results_list:
        status = f"after {r['frames_req']} scans" if r['found'] else "DID NOT FIND GOAL"
        print(f"  {r['mode'].upper():<12}: {status}")
    print("=" * 60 + "\n")

    # Comparison plot
    n_plots = len(results_list)
    fig_comp, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 8), squeeze=False)
    for i, r in enumerate(results_list):
        ax_sub = axes[0][i]
        occ    = r['occ']
        display_grid = np.zeros((*occ.shape, 3))
        display_grid[occ == -1] = [0.7, 0.7, 0.7]
        display_grid[occ == 0]  = [1.0, 1.0, 1.0]
        display_grid[occ == 1]  = [0.0, 0.0, 0.0]
        ax_sub.imshow(display_grid, origin='lower')
        ax_sub.plot(global_goal_cell[1], global_goal_cell[0], 'rx', markersize=12, label='Goal')
        if r['path']:
            y, x = zip(*r['path'])
            ax_sub.plot(x, y, 'b-', linewidth=2, label=f"{r['mode'].upper()} Greedy Path")
        ax_sub.set_title(f"Final Map: {r['mode'].upper()}")
        ax_sub.legend()
    plt.tight_layout()
    comp_name = os.path.join(
        _fig_dir(),
        f"map_comparison_Th{int(THETA_DEGREE)}_Scan{SCAN_DEPTH_THRESHOLD}.png"
    )
    plt.savefig(comp_name, dpi=300)
    print(f"Saved map comparison -> {comp_name}")

    # Overlay plot
    fig_overlay, ax_overlay = plt.subplots(figsize=(12, 10))
    occ = results_list[-1]['occ']
    display_grid = np.zeros((*occ.shape, 3))
    display_grid[occ == -1] = [0.7, 0.7, 0.7]
    display_grid[occ == 0]  = [1.0, 1.0, 1.0]
    display_grid[occ == 1]  = [0.0, 0.0, 0.0]
    ax_overlay.imshow(display_grid, origin='lower')
    ax_overlay.plot(global_goal_cell[1], global_goal_cell[0], 'y*', markersize=15, label='Goal')
    colors = {'qlearning': 'green', 'sarsa': 'magenta'}
    for r in results_list:
        if r['path']:
            y, x = zip(*r['path'])
            ax_overlay.plot(x, y, color=colors.get(r['mode'], 'blue'),
                            linewidth=3, label=f"{r['mode'].upper()}")
    ax_overlay.legend()
    plt.tight_layout()
    over_name = os.path.join(
        _fig_dir(),
        f"path_overlay_Th{int(THETA_DEGREE)}_Scan{SCAN_DEPTH_THRESHOLD}.png"
    )
    plt.savefig(over_name, dpi=300)
    print(f"Saved path overlay -> {over_name}")

    results_map = {}
    if results_ql:    results_map['qlearning'] = results_ql
    if results_sarsa: results_map['sarsa']      = results_sarsa
    _plot_convergence_diagnostics(results_map)

    try:
        from plot_results import plot_rl_comparison
        plot_rl_comparison(th=int(THETA_DEGREE), scan=SCAN_DEPTH_THRESHOLD)
    except Exception as e:
        print(f"[!] Could not auto-generate comparison charts: {e}")
        print("    Run manually: python plot_results.py")

    if os.environ.get("BATCH_MODE") != "1":
        plt.show()
        input("\n[DONE] Press Enter to exit...")
    else:
        print("\n[DONE] Batch mode - exiting.")


if __name__ == "__main__":
    run_rl_pipeline()
