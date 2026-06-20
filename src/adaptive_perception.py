"""
adaptive_perception.py
----------------------
Implements Algorithm 1: Perception Update for Safety-aware Navigation.

Only active when score_type == 1 (Unbounded mode).

The PerceptionModel maintains a lookup table:
    key   = (Cs, Ct, action_dir_idx)
    value = [(d_hat_l, f_hat_l), (d_hat_h, f_hat_h), (d_hat_r, f_hat_r)]

At each replanning step (Modes 2 & 3) the model:
  1. Measures (d, f) at the SOURCE cell Cs for the 3 facing cones.
  2. Predicts from the table (cold-start on first visit).
  3. Executes the A* move.
  4. Measures actual (d', f') at the TARGET cell Ct.
  5. Updates the model via learning rate lambda.
  6. Adapts lambda and psi based on local heterogeneity.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────
# Inline direction index mapping (mirrors view_cones.py / pathfinder.py)
# to avoid circular imports
_STEP_TO_DIR = {
    (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
    (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7
}

_CONE_NAMES = ['left', 'ahead', 'right']


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _heading_from_step(dr: int, dc: int) -> int:
    dr_n = 0 if dr == 0 else int(dr / abs(dr))
    dc_n = 0 if dc == 0 else int(dc / abs(dc))
    return _STEP_TO_DIR.get((dr_n, dc_n), 0)


# ─────────────────────────────────────────────
MeasurementTuple = List[Tuple[float, float]]   # [(d_l, f_l), (d_h, f_h), (d_r, f_r)]


class PerceptionModel:
    """
    Stateful perception model that learns (d, f) predictions per state-action.

    Parameters
    ----------
    lambda_base : float   Base learning rate.
    alpha       : float   Error scaling for adaptive lambda.
    psi_min     : int     Minimum A* look-ahead horizon (cells).
    psi_max     : int     Maximum A* look-ahead horizon (cells).
    rho         : int     Neighbourhood radius for heterogeneity (cells).
    change_thr  : float   Minimum change for lambda/psi before printing.
    """

    def __init__(
        self,
        lambda_base: float = 0.1,
        alpha: float = 2.0,
        psi_min: int = 3,
        psi_max: int = 20,
        rho: int = 3,
        change_thr: float = 0.005,
    ):
        self.lambda_base = lambda_base
        self.lambda_rate = lambda_base
        self.alpha = alpha
        self.psi_min = psi_min
        self.psi_max = psi_max
        self.psi = psi_max            # start with full look-ahead
        self.rho = rho
        self.change_thr = change_thr

        self.table: Dict[tuple, MeasurementTuple] = {}

        # Tracking for final summary
        self.n_lambda_changes = 0
        self.n_psi_changes = 0
        self.all_error_norms: List[float] = []
        self.n_steps = 0

    # ──────────────────────────────────────────────
    def measure(
        self,
        cell: Tuple[int, int],
        dist_tensor: np.ndarray,     # (rows, cols, 8)
        ray_counts: np.ndarray,      # (rows, cols)
        dir_idx: int,
    ) -> MeasurementTuple:
        """
        Lines 3–10 of Algorithm 1.
        Reads (d, f) from the 3 facing view cones at `cell`.
        """
        r, c = cell
        vl_idx = (dir_idx - 1) % 8
        vh_idx = dir_idx
        vr_idx = (dir_idx + 1) % 8

        f = float(ray_counts[r, c])

        d_l = float(dist_tensor[r, c, vl_idx])
        d_h = float(dist_tensor[r, c, vh_idx])
        d_r = float(dist_tensor[r, c, vr_idx])

        return [(d_l, f), (d_h, f), (d_r, f)]

    # ──────────────────────────────────────────────
    def predict(
        self,
        key: tuple,
        M_measured: MeasurementTuple,
    ) -> Tuple[MeasurementTuple, bool]:
        """
        Lines 11–15 of Algorithm 1.
        Returns (M_hat, is_cold_start).
        """
        if key not in self.table:
            # Cold start: use measured values as initial prediction
            return [list(m) for m in M_measured], True
        return [list(m) for m in self.table[key]], False

    # ──────────────────────────────────────────────
    def update(
        self,
        key: tuple,
        M_hat: MeasurementTuple,
        M_actual: MeasurementTuple,
    ) -> Tuple[MeasurementTuple, List[float], List[float], float]:
        """
        Lines 18–29 of Algorithm 1.
        Updates M_hat via learning rate, stores into table.

        Returns
        -------
        M_hat_updated  : updated prediction tuples
        e_d_list       : per-cone distance errors
        e_f_list       : per-cone frame errors
        error_norm     : sqrt(sum of all squared errors)
        """
        e_d_list = []
        e_f_list = []
        M_hat_updated = []

        for i in range(3):  # l, h, r
            d_hat, f_hat = M_hat[i]
            d_actual, f_actual = M_actual[i]

            e_d = d_actual - d_hat
            e_f = f_actual - f_hat

            d_hat_new = d_hat + self.lambda_rate * e_d
            f_hat_new = f_hat + self.lambda_rate * e_f

            e_d_list.append(e_d)
            e_f_list.append(e_f)
            M_hat_updated.append((d_hat_new, f_hat_new))

        error_norm = math.sqrt(sum(e**2 for e in e_d_list + e_f_list))
        self.all_error_norms.append(error_norm)
        self.n_steps += 1

        self.table[key] = M_hat_updated
        return M_hat_updated, e_d_list, e_f_list, error_norm

    # ──────────────────────────────────────────────
    def adapt(
        self,
        cell: Tuple[int, int],
        risk_tensor: np.ndarray,    # (rows, cols, 8)
        error_norm: float,
    ) -> Tuple[float, int, bool]:
        """
        Lines 30–31 of Algorithm 1: AdaptParameters.

        Computes local risk variance in rho-radius neighbourhood to set psi_v.
        Uses error-adaptive learning rate formula.

        Returns
        -------
        lambda_new : float
        psi_new    : int
        changed    : bool  — True if either changed meaningfully
        """
        r, c = cell
        rows, cols = risk_tensor.shape[0], risk_tensor.shape[1]

        # Neighbourhood slice
        r0 = max(0, r - self.rho)
        r1 = min(rows, r + self.rho + 1)
        c0 = max(0, c - self.rho)
        c1 = min(cols, c + self.rho + 1)

        neighbourhood = risk_tensor[r0:r1, c0:c1, :]   # (H, W, 8)

        # Per-direction variance over neighbourhood cells
        # vl=left=-1, vh=ahead=0, vr=right=+1 relative to current dir
        # We use all 8 channels averaged for simplicity of heterogeneity
        valid_mask = neighbourhood < 900.0   # exclude sentinel 999
        if valid_mask.sum() > 1:
            risk_vals = neighbourhood[valid_mask]
            V = float(np.var(risk_vals))
        else:
            V = 0.0

        # Normalise variance — use observed min/max with soft clipping
        V_min, V_max = 0.0, 1.0        # reasonable bounds for unbounded risk
        V_tilde = float(np.clip((V - V_min) / max(V_max - V_min, 1e-6), 0.0, 1.0))

        # Directional horizon: high variance → smaller psi
        psi_float = self.psi_min + (1.0 - V_tilde) * (self.psi_max - self.psi_min)
        psi_new = int(round(psi_float))
        psi_new = int(np.clip(psi_new, self.psi_min, self.psi_max))

        # Error-adaptive learning rate: large error → lambda closer to lambda_base
        lambda_new = float(self.lambda_base * _sigmoid(self.alpha * error_norm))

        # Determine if anything changed meaningfully
        lambda_changed = abs(lambda_new - self.lambda_rate) > self.change_thr
        psi_changed    = abs(psi_new - self.psi) >= 1

        changed = lambda_changed or psi_changed

        if lambda_changed:
            self.n_lambda_changes += 1
        if psi_changed:
            self.n_psi_changes += 1

        self.lambda_rate = lambda_new
        self.psi = psi_new

        return lambda_new, psi_new, changed

    # ──────────────────────────────────────────────
    def print_step(
        self,
        frame_num: int,
        Cs: Tuple[int, int],
        Ct: Tuple[int, int],
        dir_idx: int,
        M_before: MeasurementTuple,
        M_hat: MeasurementTuple,
        M_after: MeasurementTuple,
        e_d: List[float],
        e_f: List[float],
        error_norm: float,
        lambda_old: float,
        lambda_new: float,
        psi_old: int,
        psi_new: int,
        changed: bool,
        cold_start: bool,
    ):
        """Prints Algorithm 1 state for this step to terminal."""
        dir_names = ['N','NE','E','SE','S','SW','W','NW']
        action = dir_names[dir_idx % 8]
        print(f"  [Alg1 | Frame {frame_num} | Cs={Cs} -> Ct={Ct}] Action: {action}")
        print(f"    BEFORE  Vl=(d={M_before[0][0]:.2f}, f={M_before[0][1]:.0f})"
              f"  Vh=(d={M_before[1][0]:.2f}, f={M_before[1][1]:.0f})"
              f"  Vr=(d={M_before[2][0]:.2f}, f={M_before[2][1]:.0f})")
        pred_tag = "[cold start]" if cold_start else "[from model]"
        print(f"    PREDICT {pred_tag}"
              f"  Vl=(d={M_hat[0][0]:.2f}, f={M_hat[0][1]:.0f})"
              f"  Vh=(d={M_hat[1][0]:.2f}, f={M_hat[1][1]:.0f})"
              f"  Vr=(d={M_hat[2][0]:.2f}, f={M_hat[2][1]:.0f})")
        print(f"    AFTER   Vl=(d={M_after[0][0]:.2f}, f={M_after[0][1]:.0f})"
              f"  Vh=(d={M_after[1][0]:.2f}, f={M_after[1][1]:.0f})"
              f"  Vr=(d={M_after[2][0]:.2f}, f={M_after[2][1]:.0f})")
        print(f"    ERRORS  e_d=[{e_d[0]:+.3f}, {e_d[1]:+.3f}, {e_d[2]:+.3f}]"
              f"   e_f=[{e_f[0]:+.1f}, {e_f[1]:+.1f}, {e_f[2]:+.1f}]"
              f"   ||e||={error_norm:.4f}")
        if changed:
            lam_arrow = f"lambda {lambda_old:.4f} -> {lambda_new:.4f}" if abs(lambda_new - lambda_old) > self.change_thr else f"lambda {lambda_new:.4f} (unchanged)"
            psi_arrow = f"psi {psi_old} -> {psi_new}" if psi_old != psi_new else f"psi {psi_new} (unchanged)"
            print(f"  -> {lam_arrow}   {psi_arrow}  [ADAPTED]")

    def print_summary(self):
        """Prints the final adaptive perception summary block."""
        mean_err = float(np.mean(self.all_error_norms)) if self.all_error_norms else 0.0
        print()
        print("  " + "="*62)
        print("  Adaptive Perception Summary (Algorithm 1)")
        print("  " + "="*62)
        print(f"  Model entries learned   : {len(self.table)}")
        print(f"  Total steps run         : {self.n_steps}")
        print(f"  Final lambda (lr)       : {self.lambda_rate:.4f}")
        print(f"  Final psi (horizon)     : {self.psi} cells")
        print(f"  Total lambda changes    : {self.n_lambda_changes}")
        print(f"  Total psi changes       : {self.n_psi_changes}")
        print(f"  Mean prediction error   : {mean_err:.4f}")
        print("  " + "="*62)
        print()
