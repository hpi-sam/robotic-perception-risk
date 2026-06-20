import math
import numpy as np
from typing import Tuple, Optional
from scipy.signal import convolve2d
from view_cones import STEP_TO_DIR_IDX

# ── Normalized Risk Constants ───────────────────────────
# User defined variables for Risk scaling
VAR_DIST = 1.0     # Equivalent to 'the variable' for d
# ──────────────────────────────────────────────────────

def compute_unbounded_confidence(
    global_ray_counts: np.ndarray,
    accumulated_occ: np.ndarray,
) -> np.ndarray:
    """
    Confidence is just raw total ray counts per cell.
    """
    conf = global_ray_counts.astype(np.float32)
    conf[accumulated_occ == -1] = 0.0
    return conf

def compute_directional_distances(occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fully vectorized computation of straight-line distances to nearest obstacle/unknown.
    Uses Numpy shifts to propagate distances across the grid in O(MaxDist) instead of O(N_cells).
    """
    rows, cols = occ.shape
    dist = np.zeros((rows, cols, 8), dtype=np.float32)
    stop_type = np.zeros((rows, cols, 8), dtype=np.int8)

    dir_to_step = {v: k for k, v in STEP_TO_DIR_IDX.items()}
    
    # We propagate distances by repeatedly shifting the grid.
    # Max possible distance is the diagonal of the grid.
    max_iter = max(rows, cols)
    
    # Initial state: obstacles/unknowns have 0 distance. Free cells start at infinity.
    mask_blocked = (occ == 1) | (occ == -1)
    
    for dir_idx in range(8):
        dr, dc = dir_to_step[dir_idx]
        step_len = float(math.hypot(dr, dc))
        
        # Current distance map for this direction
        d = np.full((rows, cols), 1e6, dtype=np.float32)
        s = np.zeros((rows, cols), dtype=np.int8)
        
        # Blocked cells are the "sources" of 0 distance
        d[mask_blocked] = 0.0
        s[mask_blocked] = occ[mask_blocked]
        
        # Propagate: d[r, c] = min(d[r, c], d[nr, nc] + step_len)
        # We only need to iterate enough to cover the grid. 
        # But we can do it more efficiently by sweeping.
        
        # --- Faster Sweep Logic ---
        # For a given direction, we can compute the distance in one pass 
        # by iterating in the correct order.
        r_range = range(rows-1, -1, -1) if dr > 0 else range(rows)
        c_range = range(cols-1, -1, -1) if dc > 0 else range(cols)
        
        # We still need a loop, but we can vectorize one axis
        if dr != 0 and dc == 0:
            # Vertical sweep (vectorize columns)
            for r in r_range:
                nr = r + dr
                if 0 <= nr < rows:
                    update_mask = (occ[r, :] == 0)
                    d[r, update_mask] = d[nr, update_mask] + step_len
                    s[r, update_mask] = s[nr, update_mask]
                else:
                    # Edge of map
                    update_mask = (occ[r, :] == 0)
                    d[r, update_mask] = 0.0
                    s[r, update_mask] = 0
        elif dr == 0 and dc != 0:
            # Horizontal sweep (vectorize rows)
            for c in c_range:
                nc = c + dc
                if 0 <= nc < cols:
                    update_mask = (occ[:, c] == 0)
                    d[update_mask, c] = d[update_mask, nc] + step_len
                    s[update_mask, c] = s[update_mask, nc]
                else:
                    update_mask = (occ[:, c] == 0)
                    d[update_mask, c] = 0.0
                    s[update_mask, c] = 0
        else:
            # Diagonal sweep (harder to vectorize fully without loops)
            # We'll use the semi-vectorized approach
            for r in r_range:
                nr = r + dr
                if 0 <= nr < rows:
                    for c in c_range:
                        nc = c + dc
                        if 0 <= nc < cols and occ[r, c] == 0:
                            d[r, c] = d[nr, nc] + step_len
                            s[r, c] = s[nr, nc]
                        elif occ[r, c] == 0:
                            d[r, c] = 0.0
                            s[r, c] = 0
                else:
                    d[r, occ[r, :] == 0] = 0.0
                    s[r, occ[r, :] == 0] = 0

        dist[:, :, dir_idx] = d
        stop_type[:, :, dir_idx] = s

    return dist, stop_type


def get_cone_kernel(dir_idx: int, theta_deg: float, radius: int = 15) -> np.ndarray:
    """
    Generates a 2D binary kernel representing a vision cone.
    """
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = radius
    
    dir_to_angle = {
        0: -math.pi/2, 1: -math.pi/4, 2: 0, 3: math.pi/4,
        4: math.pi/2, 5: 3*math.pi/4, 6: math.pi, 7: -3*math.pi/4
    }
    
    base_angle = dir_to_angle[dir_idx]
    half_theta = math.radians(theta_deg) / 2.0
    
    for r in range(size):
        for c in range(size):
            dr = r - center
            dc = c - center
            dist = math.hypot(dr, dc)
            if dist > radius or dist < 0.5:
                continue
            
            angle = math.atan2(dr, dc)
            diff = (angle - base_angle + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) <= half_theta:
                kernel[r, c] = 1.0
    return kernel

def compute_cone_exploration(
    directional_distances: np.ndarray,
    accumulated_occ: np.ndarray,
    theta_deg: float,
    radius: int
) -> np.ndarray:
    """
    Optimized Vframes calculation using 2D Convolution.
    """
    rows, cols, _ = directional_distances.shape
    vframes = np.zeros((rows, cols, 8), dtype=np.float32)
    
    # Known cells = free (0) + occupied (1); unknown (-1) does NOT count.
    # Obstacles are observed terrain and should reduce risk just like free cells.
    known_mask = (accumulated_occ != -1).astype(np.float32)

    for dir_idx in range(8):
        kernel = get_cone_kernel(dir_idx, theta_deg, radius=radius)
        cone_area = np.sum(kernel)
        if cone_area == 0:
            continue

        known_count = convolve2d(known_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        # Cap at cone_area - 1 so V_phi never reaches 1.0 (risk never collapses to zero
        # in fully-mapped areas — a small residual uncertainty is always retained).
        known_count = np.minimum(known_count, cone_area - 1)
        vframes[:, :, dir_idx] = known_count / cone_area
        
    return vframes

def compute_unbounded_risk(
    directional_distances: np.ndarray,
    accumulated_occ: np.ndarray,
    d_dir_stop_type: np.ndarray = None,
    var_dist: float = VAR_DIST,
    theta_deg: float = 45.0,
    base_deg: float = 60.0,
    hazard_obstacle: float = 10.0,
    hazard_unknown: float = 1.0,
    normalized: bool = True,
    training_mode: bool = False,
    radius: int = None,  # Now required
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Computes Risk using area-based cone exploration with Hazard multiplier.
    """
    rows, cols, _ = directional_distances.shape
    risk = np.zeros((rows, cols, 8), dtype=np.float32)

    # 1. Compute Vframes_current (always needed)
    vframes_current = compute_cone_exploration(directional_distances, accumulated_occ, theta_deg, radius=radius)
    
    # 2. Skip Vframes_base if in training mode (it's slow and only for visualization)
    vframes_base = None
    if not training_mode:
        vframes_base = compute_cone_exploration(directional_distances, accumulated_occ, base_deg, radius=radius)

    # 3. Build Hazard array
    hazard = np.ones((rows, cols, 8), dtype=np.float32)
    if d_dir_stop_type is not None:
        hazard[d_dir_stop_type == 1] = hazard_obstacle
        hazard[d_dir_stop_type == -1] = hazard_unknown
    
    # 4. Compute Risk
    if normalized:
        risk = hazard * (var_dist / (directional_distances + var_dist)) * (1.0 - vframes_current)
    else:
        risk = hazard * (1.0 / (directional_distances + 0.1))

    return risk, vframes_current, vframes_base
