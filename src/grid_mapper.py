"""
grid_mapper.py
--------------
Projects a depth image into a top-down 2D occupancy grid.

Occupancy semantics
    1  = occupied  (obstacle)
    0  = free      (ray passed through, no obstacle)
   -1  = unknown   (no observation)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics."""
    fx: float = 525.0   # focal length x  (px)
    fy: float = 525.0   # focal length y  (px)
    cx: float = 319.5   # principal point x (px)
    cy: float = 239.5   # principal point y (px)


@dataclass
class GridConfig:
    """Top-down grid parameters."""
    resolution: float = 0.05           # metres per cell
    x_range: Tuple[float, float] = (-3.0, 3.0)   # left-right  (metres)
    z_range: Tuple[float, float] = (0.2, 6.0)     # near-far    (metres, depth axis)
    obstacle_height_min: float = 0.8   # metres above ground (raise to skip ground clutter)
    obstacle_height_max: float = 2.5   # metres above ground
    min_points_per_cell: int = 3      # hits needed to mark as occupied
    obstacle_hit_ratio: float = 0.3    # fraction of rays that must be hits to mark occupied
    near_clip: float = 0.6             # ignore depth closer than this (metres)


def back_project(depth: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """
    Back-project a depth image to 3D points in the camera frame.

    Parameters
    ----------
    depth      : (H, W) float32 depth in metres.  0 = invalid.
    intrinsics : camera intrinsics.

    Returns
    -------
    points : (N, 3) float32 array of [X, Y, Z] in camera frame.
             Convention:  X→right, Y→down, Z→forward.
    """
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    valid = depth > 0
    z = depth[valid]
    x = (u[valid] - intrinsics.cx) * z / intrinsics.fx
    y = (v[valid] - intrinsics.cy) * z / intrinsics.fy

    return np.stack([x, y, z], axis=-1)


def back_project_clipped(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    near_clip: float = 0.6,
    far_clip: float = 10.0,
) -> np.ndarray:
    """
    Back-project with a near/far clip to reject ultra-close noise and
    far-range noise.
    """
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    valid = (depth > near_clip) & (depth < far_clip)
    z = depth[valid]
    x = (u[valid] - intrinsics.cx) * z / intrinsics.fx
    y = (v[valid] - intrinsics.cy) * z / intrinsics.fy

    return np.stack([x, y, z], axis=-1)


def depth_to_occupancy_grid(
    depth: np.ndarray,
    intrinsics: Optional[CameraIntrinsics] = None,
    config: Optional[GridConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert a single depth frame into a 2D occupancy grid.

    Parameters
    ----------
    depth      : (H, W) float32 depth in metres.
    intrinsics : camera params (uses defaults if None).
    config     : grid params   (uses defaults if None).

    Returns
    -------
    grid       : (rows, cols) int8 array  (1=occupied, 0=free, -1=unknown).
    hit_counts : (rows, cols) int32 array  (number of obstacle points per cell).
    meta       : dict with grid extents for plotting.
    """
    if intrinsics is None:
        intrinsics = CameraIntrinsics()
    if config is None:
        config = GridConfig()

    # ---- compute grid dimensions ----
    cols = int(np.ceil((config.x_range[1] - config.x_range[0]) / config.resolution))
    rows = int(np.ceil((config.z_range[1] - config.z_range[0]) / config.resolution))

    hit_counts = np.zeros((rows, cols), dtype=np.int32)
    ray_counts = np.zeros((rows, cols), dtype=np.int32)

    # ---- back-project to 3D ----
    pts = back_project(depth, intrinsics)
    if pts.shape[0] == 0:
        grid = np.full((rows, cols), -1, dtype=np.int8)
        meta = _make_meta(config, rows, cols)
        return grid, hit_counts, meta

    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

    # ---- mark ray traversal (free space) along each column ----
    # For every valid pixel we know that all cells between the camera and the
    # point's Z are free (unless another point occupies them).
    in_x = (X >= config.x_range[0]) & (X < config.x_range[1])
    in_z = (Z >= config.z_range[0]) & (Z < config.z_range[1])
    valid_xz = in_x & in_z

    ci = ((X[valid_xz] - config.x_range[0]) / config.resolution).astype(np.int32)
    ri = ((Z[valid_xz] - config.z_range[0]) / config.resolution).astype(np.int32)
    ci = np.clip(ci, 0, cols - 1)
    ri = np.clip(ri, 0, rows - 1)

    # Mark free space: all cells from row 0 up to (but not including) the hit row
    for c, r in zip(ci, ri):
        ray_counts[:r, c] += 1

    # ---- classify obstacle vs. non-obstacle by height ----
    Y_valid = Y[valid_xz]
    # In typical depth camera frame, Y points down, so ground ~ positive Y.
    # We use the height band relative to camera:
    # A point is 'relevant' for ratio if it's not clearly the floor
    # (e.g., higher than 0.3m above ground)
    is_relevant = (-Y_valid >= 0.3)
    is_obstacle = (
        (-Y_valid >= config.obstacle_height_min) &
        (-Y_valid <= config.obstacle_height_max)
    )

    relevant_counts = np.zeros((rows, cols), dtype=np.int32)
    obs_ci = ci[is_obstacle]
    obs_ri = ri[is_obstacle]
    np.add.at(hit_counts, (obs_ri, obs_ci), 1)

    rel_ci = ci[is_relevant]
    rel_ri = ri[is_relevant]
    np.add.at(relevant_counts, (rel_ri, rel_ci), 1)

    # ray_counts for cells that received a hit (all points)
    np.add.at(ray_counts, (ri, ci), 1)

    # ---- assemble grid (ratio-based obstacle detection) ----
    grid = np.full((rows, cols), -1, dtype=np.int8)  # unknown
    grid[ray_counts > 0] = 0                          # free (ray passed through)

    # A cell is occupied only if BOTH conditions are met:
    #   1) enough total hits, AND
    #   2) a significant fraction of RELEVANT rays were obstacle hits
    hit_ratio = hit_counts.astype(np.float32) / np.maximum(relevant_counts, 1)
    occ_mask = (
        (hit_counts >= config.min_points_per_cell) &
        (hit_ratio > config.obstacle_hit_ratio)
    )
    grid[occ_mask] = 1  # occupied

    meta = _make_meta(config, rows, cols)
    return grid, hit_counts, meta


def _make_meta(config: GridConfig, rows: int, cols: int) -> dict:
    return {
        "rows": rows,
        "cols": cols,
        "resolution": config.resolution,
        "x_range": config.x_range,
        "z_range": config.z_range,
        "x_extent": [config.x_range[0], config.x_range[1]],
        "z_extent": [config.z_range[0], config.z_range[1]],
    }


def depth_to_occupancy_grid_world(
    world_points: np.ndarray,
    config: 'GridConfig',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Map world-frame 3D points into a 2D occupancy grid in world coordinates.

    Returns
    -------
    grid       : (rows, cols) int8 array  (1=occ, 0=free, -1=unknown).
    hit_counts : (rows, cols) int32 array.
    ray_counts : (rows, cols) int32 array.
    frame_conf : (rows, cols) float32 — per-cell confidence for THIS single frame,
                 based on how many points observed the cell (saturates via tanh).
    meta       : dict with grid extents.
    """
    cols = int(np.ceil((config.x_range[1] - config.x_range[0]) / config.resolution))
    rows = int(np.ceil((config.z_range[1] - config.z_range[0]) / config.resolution))

    hit_counts = np.zeros((rows, cols), dtype=np.int32)
    ray_counts = np.zeros((rows, cols), dtype=np.int32)
    frame_conf = np.zeros((rows, cols), dtype=np.float32)

    if world_points.shape[0] == 0:
        grid = np.full((rows, cols), -1, dtype=np.int8)
        meta = _make_meta(config, rows, cols)
        return grid, hit_counts, ray_counts, frame_conf, meta

    WX = world_points[:, 0]
    WY = world_points[:, 1]
    VZ = world_points[:, 2]

    in_x = (WX >= config.x_range[0]) & (WX < config.x_range[1])
    in_y = (WY >= config.z_range[0]) & (WY < config.z_range[1])
    valid = in_x & in_y

    ci = ((WX[valid] - config.x_range[0]) / config.resolution).astype(np.int32)
    ri = ((WY[valid] - config.z_range[0]) / config.resolution).astype(np.int32)
    ci = np.clip(ci, 0, cols - 1)
    ri = np.clip(ri, 0, rows - 1)

    np.add.at(ray_counts, (ri, ci), 1)

    VZ_valid = VZ[valid]
    # Ground-plane is Z_world (VZ_valid in this context).
    # Ignore points below 0.3m for the ratio denominator to avoid floor clutter.
    is_relevant = (-VZ_valid >= 0.3)
    is_obstacle = (
        (-VZ_valid >= config.obstacle_height_min) &
        (-VZ_valid <= config.obstacle_height_max)
    )
    
    relevant_counts = np.zeros((rows, cols), dtype=np.int32)
    obs_ci = ci[is_obstacle]
    obs_ri = ri[is_obstacle]
    np.add.at(hit_counts, (obs_ri, obs_ci), 1)
    
    rel_ci = ci[is_relevant]
    rel_ri = ri[is_relevant]
    np.add.at(relevant_counts, (rel_ri, rel_ci), 1)

    # Assemble grid (ratio-based obstacle detection)
    grid = np.full((rows, cols), -1, dtype=np.int8)
    grid[ray_counts > 0] = 0

    hit_ratio = hit_counts.astype(np.float32) / np.maximum(relevant_counts, 1)
    occ_mask = (
        (hit_counts >= config.min_points_per_cell) &
        (hit_ratio > config.obstacle_hit_ratio)
    )
    grid[occ_mask] = 1

    # Per-cell frame confidence: tanh(ray_count / scale) -> [0, 1]
    seen = ray_counts > 0
    frame_conf[seen] = np.tanh(ray_counts[seen].astype(np.float32) / 20.0)

    meta = _make_meta(config, rows, cols)
    return grid, hit_counts, ray_counts, frame_conf, meta


def debug_cell_stats(hit_counts, ray_counts, grid, meta, trajectory=None):
    """
    Print statistics about cell hit/ray counts near the center or trajectory.
    Useful for tuning obstacle thresholds.
    """
    rows, cols = hit_counts.shape
    res = meta['resolution']

    # If trajectory provided, sample cells under it
    if trajectory and len(trajectory) > 0:
        sample_cells = []
        for wx, wy in trajectory[::max(1, len(trajectory)//20)]:
            c = int((wx - meta['x_range'][0]) / res)
            r = int((wy - meta['z_range'][0]) / res)
            c = max(0, min(c, cols - 1))
            r = max(0, min(r, rows - 1))
            sample_cells.append((r, c))
    else:
        # Sample the center strip
        mid_c = cols // 2
        sample_cells = [(r, mid_c) for r in range(0, rows, max(1, rows // 20))]

    print(f"\n{'='*70}")
    print(f"Cell Stats Debug (sampled {len(sample_cells)} cells)")
    print(f"{'Cell(r,c)':>14} {'hits':>6} {'rays':>6} {'ratio':>8} {'status':>8}")
    print(f"{'-'*70}")

    _S = {-1: 'UNK', 0: 'FREE', 1: 'OCC'}
    for r, c in sample_cells:
        h = int(hit_counts[r, c])
        ray = int(ray_counts[r, c])
        ratio = h / max(ray, 1)
        status = _S.get(int(grid[r, c]), '?')
        print(f"  ({r:4d},{c:4d}) {h:6d} {ray:6d} {ratio:8.3f} {status:>8}")

    # Summary
    seen = ray_counts > 0
    if seen.any():
        ratios = hit_counts[seen].astype(float) / ray_counts[seen]
        print(f"\n  Observed cells: {seen.sum():,}")
        print(f"  Hit ratio -- mean={ratios.mean():.4f}  median={np.median(ratios):.4f}  "
              f"max={ratios.max():.4f}  p95={np.percentile(ratios, 95):.4f}")
    print(f"{'='*70}")


def accumulate_with_confidence(
    acc_grid: Optional[np.ndarray],
    acc_conf: Optional[np.ndarray],
    new_grid: np.ndarray,
    new_conf: np.ndarray,
    free_clear_threshold: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Confidence-weighted map accumulation.

    Rules:
      1. If the new frame has HIGHER confidence for a cell than the existing
         best, the new observation's status (free/occupied) wins.
      2. Clear Rule: If the new frame says FREE with reasonable confidence 
         (>= free_clear_threshold), it overrides any existing OBSTACLE.
      3. Confidence only goes UP — monotonic max (except when forcefully cleared).
      4. Unknown cells (-1) are always replaced by any observation.

    Returns
    -------
    updated_grid : (rows, cols) int8
    updated_conf : (rows, cols) float32
    """
    if acc_grid is None:
        return new_grid.copy(), new_conf.copy()

    out_grid = acc_grid.copy()
    out_conf = acc_conf.copy()

    # Cells where new frame actually observed something (not unknown)
    new_observed = new_grid != -1

    # Case A: cell was unknown → always accept new observation
    was_unknown = acc_grid == -1
    accept_a = was_unknown & new_observed
    out_grid[accept_a] = new_grid[accept_a]
    out_conf[accept_a] = new_conf[accept_a]

    # Case B: cell was known → accept if new confidence is higher
    was_known = acc_grid != -1
    better_conf = new_conf > acc_conf
    accept_b = was_known & new_observed & better_conf
    out_grid[accept_b] = new_grid[accept_b]

    # Case C: Force Clear Obstacles
    # If the map says OBSTACLE (1) but the new scan says FREE (0) with good confidence
    clear_override = (acc_grid == 1) & (new_grid == 0) & (new_conf >= free_clear_threshold)
    out_grid[clear_override] = 0

    # Confidence Update: 
    # For cells that were forcefully cleared, reset their confidence to the new scan's value.
    # Otherwise, confidence monotonically increases (takes the max of old/new).
    out_conf = np.where(clear_override, new_conf, np.maximum(out_conf, new_conf))

    return out_grid, out_conf


# Legacy wrapper for backward compatibility
def accumulate_grid(
    accumulated: Optional[np.ndarray],
    new_grid: np.ndarray,
) -> np.ndarray:
    """
    Simple accumulation (no confidence weighting).
    Kept for backward compatibility; prefer accumulate_with_confidence.
    """
    if accumulated is None:
        return new_grid.copy()
    result = accumulated.copy()
    result[new_grid == 1] = 1
    upgrade = (new_grid == 0) & (result == -1)
    result[upgrade] = 0
    return result

