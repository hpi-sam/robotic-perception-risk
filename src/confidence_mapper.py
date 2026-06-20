"""
confidence_mapper.py
--------------------
Computes per-cell confidence from depth quality and repeated observations.

Confidence is in [0, 1]:
  High = reliable, repeated, low-variance observations.
  Low  = noisy, sparse, or no observations.
"""

import numpy as np
from typing import Optional, Tuple

from grid_mapper import CameraIntrinsics, GridConfig, back_project


def _cell_indices(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: GridConfig,
    rows: int,
    cols: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-point cell indices and depth values for valid in-range points.

    Returns  (ri, ci, z_values, valid_y_mask)
    """
    pts = back_project(depth, intrinsics)
    if pts.shape[0] == 0:
        empty = np.array([], dtype=np.int32)
        return empty, empty, np.array([], dtype=np.float32), np.array([], dtype=bool)

    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

    in_x = (X >= config.x_range[0]) & (X < config.x_range[1])
    in_z = (Z >= config.z_range[0]) & (Z < config.z_range[1])
    mask = in_x & in_z

    ci = ((X[mask] - config.x_range[0]) / config.resolution).astype(np.int32)
    ri = ((Z[mask] - config.z_range[0]) / config.resolution).astype(np.int32)
    ci = np.clip(ci, 0, cols - 1)
    ri = np.clip(ri, 0, rows - 1)

    return ri, ci, Z[mask], mask


class ConfidenceAccumulator:
    """
    Maintains running confidence statistics across frames.

    For every grid cell we track:
      - observation_count : how many depth points accumulated
      - depth_sum         : running sum of depth values
      - depth_sum_sq      : running sum of squared depth values
      - valid_ratio_sum   : sum of per-frame validity ratios contributing to cell

    Confidence for a cell is computed as:

        C = alpha * consistency_score  +  (1 - alpha) * observation_score

    where
      consistency_score = 1 / (1 + normalised_depth_variance)
      observation_score = 1 - exp(-observation_count / tau)
    """

    def __init__(self, rows: int, cols: int, alpha: float = 0.5, tau: float = 50.0):
        self.rows = rows
        self.cols = cols
        self.alpha = alpha
        self.tau = tau

        self.obs_count = np.zeros((rows, cols), dtype=np.float64)
        self.depth_sum = np.zeros((rows, cols), dtype=np.float64)
        self.depth_sq_sum = np.zeros((rows, cols), dtype=np.float64)
        self.valid_ratio_sum = np.zeros((rows, cols), dtype=np.float64)
        self.frame_count = np.zeros((rows, cols), dtype=np.float64)

    def update(
        self,
        depth: np.ndarray,
        intrinsics: Optional[CameraIntrinsics] = None,
        config: Optional[GridConfig] = None,
    ) -> None:
        """Accumulate statistics from one depth frame."""
        if intrinsics is None:
            intrinsics = CameraIntrinsics()
        if config is None:
            config = GridConfig()

        ri, ci, z_vals, _ = _cell_indices(
            depth, intrinsics, config, self.rows, self.cols
        )
        if ri.size == 0:
            return

        # Validity ratio: fraction of non-zero pixels in the depth image
        total_pixels = depth.size
        valid_pixels = np.count_nonzero(depth)
        validity = valid_pixels / max(total_pixels, 1)

        np.add.at(self.obs_count, (ri, ci), 1)
        np.add.at(self.depth_sum, (ri, ci), z_vals.astype(np.float64))
        np.add.at(self.depth_sq_sum, (ri, ci), (z_vals.astype(np.float64)) ** 2)

        # Record validity per unique cell touched in this frame
        touched = np.zeros((self.rows, self.cols), dtype=bool)
        touched[ri, ci] = True
        self.valid_ratio_sum[touched] += validity
        self.frame_count[touched] += 1

    def update_world(
        self,
        world_points: np.ndarray,
        config: Optional[GridConfig] = None,
    ) -> None:
        """Accumulate statistics from world-frame points (SLAM mode)."""
        if config is None:
            config = GridConfig()

        if world_points.shape[0] == 0:
            return

        WX = world_points[:, 0]
        WY = world_points[:, 1]
        # Use depth-like value (distance from camera in Z_cam preserved as magnitude)
        # For confidence we use the ground-plane distance as a proxy
        depth_proxy = np.sqrt(WX**2 + WY**2)

        in_x = (WX >= config.x_range[0]) & (WX < config.x_range[1])
        in_y = (WY >= config.z_range[0]) & (WY < config.z_range[1])
        mask = in_x & in_y

        ci = ((WX[mask] - config.x_range[0]) / config.resolution).astype(np.int32)
        ri = ((WY[mask] - config.z_range[0]) / config.resolution).astype(np.int32)
        ci = np.clip(ci, 0, self.cols - 1)
        ri = np.clip(ri, 0, self.rows - 1)

        z_vals = depth_proxy[mask].astype(np.float64)

        np.add.at(self.obs_count, (ri, ci), 1)
        np.add.at(self.depth_sum, (ri, ci), z_vals)
        np.add.at(self.depth_sq_sum, (ri, ci), z_vals ** 2)

        touched = np.zeros((self.rows, self.cols), dtype=bool)
        touched[ri, ci] = True
        self.valid_ratio_sum[touched] += 1.0
        self.frame_count[touched] += 1

    def confidence_grid(self) -> np.ndarray:
        """
        Compute and return the confidence grid in [0, 1].
        """
        conf = np.zeros((self.rows, self.cols), dtype=np.float32)

        observed = self.obs_count > 0

        # --- consistency score (low variance = high confidence) ---
        mean_d = np.zeros_like(self.depth_sum)
        var_d = np.zeros_like(self.depth_sum)
        mean_d[observed] = self.depth_sum[observed] / self.obs_count[observed]
        var_d[observed] = (
            self.depth_sq_sum[observed] / self.obs_count[observed]
            - mean_d[observed] ** 2
        )
        var_d = np.clip(var_d, 0, None)

        # Normalise variance by mean depth squared so it is scale-invariant
        norm_var = np.zeros_like(var_d)
        nonzero_mean = mean_d > 0
        norm_var[nonzero_mean] = var_d[nonzero_mean] / (mean_d[nonzero_mean] ** 2)

        consistency = 1.0 / (1.0 + norm_var * 100.0)  # steepness factor

        # --- observation score (more obs = higher confidence) ---
        obs_score = 1.0 - np.exp(-self.obs_count / self.tau)

        # --- combine ---
        conf[observed] = (
            self.alpha * consistency[observed]
            + (1.0 - self.alpha) * obs_score[observed]
        ).astype(np.float32)

        return conf
