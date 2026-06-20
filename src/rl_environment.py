"""
rl_environment.py
-----------------
Safety-aware MDP environment for tabular RL planners (SARSA / Q-Learning).

State encoding (goal-blind, per supervisor feedback):
    s = (row, col, dir_idx, d_h_bin, d_l_bin, d_r_bin)

    - (row, col)             current cell
    - dir_idx                facing direction 0-7
    - d_h_bin, d_l_bin, d_r_bin  nearest obstacle distance discretised to 5 bins:
                                   0 = <1 cell, 1 = 1-2, 2 = 2-4, 3 = 4-8, 4 = >8

    NOTE: The goal is NOT in the state. The agent discovers the +1000 goal
    reward purely through exploration, learning a general safe-navigation policy
    rather than a GPS-guided one. The env still tracks self.goal internally
    to detect arrival.

Reward function R(s, a, s'):
    +1000                   goal reached
    -100                    collision / obstacle / out-of-bounds
    -(d_step + R_direction) normal step:
                              d_step     = 1.0  (cardinal)  or 2.0 (diagonal)
                              R_direction = risk_tensor[nr, nc, action_idx]
                            This couples geometric cost and directional safety
                            into a single principled formula.
    +Progress Carrot         reward to point agent roughly to goal
"""

import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Tuple, List

# 8 action directions: (dr, dc) indexed 0-7
# Matches STEP_TO_DIR_IDX in view_cones.py
ACTIONS = [
    (1,  0),   # 0  North
    (1,  1),   # 1  Northeast  (diagonal)
    (0,  1),   # 2  East
    (-1, 1),   # 3  Southeast  (diagonal)
    (-1, 0),   # 4  South
    (-1,-1),   # 5  Southwest  (diagonal)
    (0, -1),   # 6  West
    (1, -1),   # 7  Northwest   (diagonal)
]
N_ACTIONS = len(ACTIONS)

# Removed static _D_STEP array to allow dynamic configuration from main.py

# Distance bins (in grid cells)
DIST_BINS = [1.0, 2.0, 4.0, 8.0]   # 4 thresholds -> 5 bins


def _bin_distance(d: float) -> int:
    """Discretise a continuous distance into 5 bins 0-4."""
    for i, threshold in enumerate(DIST_BINS):
        if d < threshold:
            return i
    return len(DIST_BINS)   # bin 4 = > 8 cells


class GridEnv:
    """
    Safety-aware grid MDP wrapper.

    Parameters
    ----------
    occ_grid    : (rows, cols) int8 — occupancy (0=free, 1=obstacle, -1=unknown)
    risk_tensor : (rows, cols, 8) float32 — directional unbounded risk OR
                  (rows, cols) float32 — bounded risk (2D)
    dist_tensor : (rows, cols, 8) float32 — directional distance sweeps
    ray_counts  : (rows, cols) int32 — frame counts per cell
    goal        : (row, col) target cell  [stored but NOT put in state]
    reward_goal         : scalar reward for reaching goal
    reward_collision    : scalar reward for hitting obstacle
    max_steps_per_ep    : hard step limit per episode (prevents infinite loops)
    """

    def __init__(
        self,
        occ_grid: np.ndarray,
        risk_tensor: np.ndarray,
        dist_tensor: np.ndarray,
        ray_counts: np.ndarray,
        goal: Tuple[int, int],
        stop_type: np.ndarray = None,
        reward_goal: float = 1000.0,
        reward_collision: float = -100.0,
        progress_weight: float = 10.0,
        use_carrot: bool = True,
        cost_cardinal: float = 1.0,
        cost_diagonal: float = 2.0,
        max_steps_per_ep: int = 500,
        **kwargs
    ):
        self.occ  = occ_grid
        self.risk = risk_tensor
        self.dist = dist_tensor
        self.stop_type = stop_type
        self.ray  = ray_counts
        self.goal = goal
        self.rows, self.cols = occ_grid.shape

        self.R_GOAL  = reward_goal
        self.R_COLL  = reward_collision
        self.progress_weight = progress_weight
        self.use_carrot = use_carrot
        self.max_steps = max_steps_per_ep
        self.coverage_mask = kwargs.get('coverage_mask', None)
        self.R_BLIND = kwargs.get('reward_blind', 0.0)

        # Dynamically build step costs based on user inputs
        self.d_step_costs = [
            cost_cardinal, # 0  North
            cost_diagonal, # 1  Northeast
            cost_cardinal, # 2  East
            cost_diagonal, # 3  Southeast
            cost_cardinal, # 4  South
            cost_diagonal, # 5  Southwest
            cost_cardinal, # 6  West
            cost_diagonal, # 7  Northwest
        ]

        self._state: tuple = None
        self._step_count: int = 0

    # -------------------------------------------------------------------
    def update_map(
        self,
        occ_grid: np.ndarray,
        risk_tensor: np.ndarray,
        dist_tensor: np.ndarray,
        ray_counts: np.ndarray,
        stop_type: np.ndarray = None,
    ):
        """Called each time a new sensor frame arrives and the map is updated."""
        self.occ  = occ_grid
        self.risk = risk_tensor
        self.dist = dist_tensor
        self.stop_type = stop_type
        self.ray  = ray_counts
        self.rows, self.cols = occ_grid.shape

    # -------------------------------------------------------------------
    def reset(self, start: Tuple[int, int], start_dir: int = 0) -> tuple:
        """Start a new episode from `start`, return initial state."""
        self._step_count = 0
        self._state = self._encode(start[0], start[1], start_dir)
        return self._state

    # -------------------------------------------------------------------
    def step(self, action_idx: int) -> Tuple[tuple, float, bool]:
        """
        Execute `action_idx`, return (next_state, reward, done).
        Reward formula (supervisor feedback):
            goal      -> +R_GOAL
            collision -> +R_COLL  (negative, episode ends)
       # Normal step: -(d_step + R_direction)
    # d_step = 1 or 2 depending on cardinal/diagonal
    # R_direction = directional risk at destination cell
        """
        r, c, dir_idx = self._decode_pos(self._state)
        dr, dc = ACTIONS[action_idx]
        nr, nc = r + dr, c + dc

        self._step_count += 1
        done = False

        # Out of bounds or non-free cell -> collision penalty, stay in place
        if (nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols
                or self.occ[nr, nc] != 0):
            reward = self.R_COLL
            done = False
            next_state = self._state   # stay in place
        elif (nr, nc) == self.goal:
            reward = self.R_GOAL
            done = True
            next_state = self._encode(nr, nc, action_idx)
        else:
            # Progress Reward: encourage moving closer to the goal
            progress_to_goal = 0.0
            if self.use_carrot:
                dist_old = math.hypot(self.goal[0] - r, self.goal[1] - c)
                dist_new = math.hypot(self.goal[0] - nr, self.goal[1] - nc)
                progress_to_goal = (dist_old - dist_new) * self.progress_weight

            # Normal step: -(d_step + R_direction) + progress
            d_step  = self.d_step_costs[action_idx]
            r_dir   = self._get_risk(nr, nc, action_idx)
            reward  = -(d_step + r_dir) + progress_to_goal

            # --- Active Perception Check ---
            # If we have a coverage mask and the current state is "Blind", apply penalty
            if self.coverage_mask is not None:
                # Check coverage for the next state (nr, nc) facing action_idx
                if not self.coverage_mask[nr, nc, action_idx]:
                    reward += self.R_BLIND # R_BLIND is negative

            next_state = self._encode(nr, nc, action_idx)

        # Hard step limit
        if self._step_count >= self.max_steps:
            done = True

        self._state = next_state
        return next_state, reward, done

    # -------------------------------------------------------------------
    def valid_actions(self) -> List[int]:
        """Return action indices that lead to free cells (occ == 0) only."""
        r, c, _ = self._decode_pos(self._state)
        valid = []
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.occ[nr, nc] == 0:
                valid.append(a)
        return valid if valid else list(range(N_ACTIONS))

    # -------------------------------------------------------------------
    def _encode(self, row: int, col: int, dir_idx: int) -> tuple:
        """Build the goal-blind perception state tuple (6 elements)."""
        vl_idx = (dir_idx - 1) % 8
        vh_idx = dir_idx
        vr_idx = (dir_idx + 1) % 8

        if self.dist.ndim == 3:
            d_h = float(self.dist[row, col, vh_idx])
            d_l = float(self.dist[row, col, vl_idx])
            d_r = float(self.dist[row, col, vr_idx])
        else:
            d_h = d_l = d_r = 999.0

        return (
            row, col,
            dir_idx,
            _bin_distance(d_h),
            _bin_distance(d_l),
            _bin_distance(d_r),
        )

    # -------------------------------------------------------------------
    @staticmethod
    def _decode_pos(state: tuple) -> Tuple[int, int, int]:
        """Extract (row, col, dir_idx) from 6-tuple state."""
        return state[0], state[1], state[2]

    # -------------------------------------------------------------------
    def _get_risk(self, row: int, col: int, dir_idx: int) -> float:
        """Get directional risk for moving INTO (row, col) from direction dir_idx."""
        if self.risk.ndim == 3:
            return float(self.risk[row, col, dir_idx])
        return float(self.risk[row, col])

    # -------------------------------------------------------------------
    def extract_greedy_path(
        self,
        agent,
        start: Tuple[int, int],
        start_dir: int = 0,
        max_steps: int = 300,
    ) -> Tuple[List[Tuple[int, int]], float]:
        """
        Follow the greedy policy (no exploration) from start -> goal.

        Tie-breaking: when multiple actions share the same max Q-value (including
        all-zero for unvisited states), pick the one whose resulting cell is
        geometrically closest to the goal.  This makes extraction deterministic
        and purposeful even in areas the agent never visited during training.
        """
        state = self.reset(start, start_dir)
        path = [start]
        visited = {start}
        cumulative_reward = 0.0

        for _ in range(max_steps):
            r, c, dir_idx = self._decode_pos(state)

            valid = self.valid_actions()

            q_values = [agent.Q.get((state, a), 0.0) for a in valid]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(valid, q_values) if math.isclose(q, max_q, abs_tol=1e-6)]

            if len(best_actions) == 1:
                action = best_actions[0]
            else:
                # Break ties by Euclidean distance to goal — pick the action
                # that lands closest to the goal among equally-valued options.
                gr, gc = self.goal
                action = min(
                    best_actions,
                    key=lambda a: math.hypot(
                        (r + ACTIONS[a][0]) - gr,
                        (c + ACTIONS[a][1]) - gc,
                    ),
                )

            dr, dc = ACTIONS[action]
            nr, nc = r + dr, c + dc

            if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols or self.occ[nr, nc] != 0:
                break

            if (nr, nc) in visited:
                break   # cycle detected — stop
            visited.add((nr, nc))
            path.append((nr, nc))

            next_state, reward, done = self.step(action)
            cumulative_reward += reward
            state = next_state

            if done or (nr, nc) == self.goal:
                break

        return path, cumulative_reward

def compute_distance_transform(occ: np.ndarray) -> np.ndarray:
    """
    Computes Euclidean distance to the nearest obstacle (occ=1).
    Unknown (-1) and Free (0) are treated as safe for the purpose of the margin.
    """
    # 0 where obstacle, 1 where safe
    binary_mask = (occ != 1).astype(float)
    return distance_transform_edt(binary_mask)
