"""
pose_loader.py
--------------
Loads ground-truth poses from a text file by INDEX (line i → frame i).
Heading is estimated from consecutive position deltas.
"""

import math
import numpy as np
from typing import List, Optional


def load_groundtruth(filepath: str) -> np.ndarray:
    """
    Parse ground-truth file.  Each line:
        timestamp  x  y  z  [qx qy qz qw]   (extra columns ignored)

    Returns
    -------
    poses : (N, 3) array of [x, y, z] positions, indexed by line number.
    """
    poses = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            # Skip timestamp (parts[0]); take the 3 position values
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            poses.append([x, y, z])
    return np.array(poses, dtype=np.float64)


def get_pose(poses: np.ndarray, frame_idx: int) -> Optional[np.ndarray]:
    """
    Get pose for frame_idx (0-indexed).
    Returns [x, y, z] or None if index is out of range.
    """
    if frame_idx < 0 or frame_idx >= len(poses):
        return None
    return poses[frame_idx]


def estimate_heading(pos_prev: np.ndarray, pos_curr: np.ndarray) -> float:
    """
    Estimate yaw heading (radians) from two consecutive 2D positions.
    Uses ground-plane displacement (x, y) → heading = atan2(dy, dx).
    Returns 0.0 if positions are identical.
    """
    dx = pos_curr[0] - pos_prev[0]
    dy = pos_curr[1] - pos_prev[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return math.atan2(dy, dx)


def transform_points(
    points_cam: np.ndarray,
    position: np.ndarray,
    yaw: float,
) -> np.ndarray:
    """
    Transform camera-frame 3D points into world coordinates.

    Parameters
    ----------
    points_cam : (N, 3) in camera frame [X_right, Y_down, Z_forward].
    position   : [x, y, z] world position of the camera.
    yaw        : heading angle in radians.

    Returns
    -------
    points_world : (N, 3) in world frame [X_world, Y_world, Z_vertical].

    Camera convention → world:
        Camera Z (forward)  →  direction determined by yaw
        Camera X (right)    →  perpendicular to heading
        Camera Y (down)     →  kept as Z_vertical for height filtering
    """
    if points_cam.shape[0] == 0:
        return points_cam.copy()

    cam_x = points_cam[:, 0]  # right
    cam_y = points_cam[:, 1]  # down (vertical)
    cam_z = points_cam[:, 2]  # forward

    c, s = math.cos(yaw), math.sin(yaw)
    world_x = c * cam_z - s * cam_x + position[0]
    world_y = s * cam_z + c * cam_x + position[1]
    world_z = cam_y  # vertical (height), used for obstacle filtering

    return np.stack([world_x, world_y, world_z], axis=-1)
