"""
view_cones.py
-------------
Implements the 8-directional View Cones (V) logic based on world coordinates
from the ground truth.

Coordinate mapping:
    North : +Y in absolute groundtruth coordinate frame (+r in grid array)
    East  : +X in absolute groundtruth coordinate frame (+c in grid array)
"""

DIRECTIONS = [
    'North',      # 0
    'Northeast',  # 1
    'East',       # 2
    'Southeast',  # 3
    'South',      # 4
    'Southwest',  # 5
    'West',       # 6
    'Northwest'   # 7
]

# Map from a grid step (dr, dc) to the canonical direction index
STEP_TO_DIR_IDX = {
    (1, 0):  0,  # North (+Y)
    (1, 1):  1,  # Northeast (+Y, +X)
    (0, 1):  2,  # East (+X)
    (-1, 1): 3,  # Southeast (-Y, +X)
    (-1, 0): 4,  # South (-Y)
    (-1, -1):5,  # Southwest (-Y, -X)
    (0, -1): 6,  # West (-X)
    (1, -1): 7   # Northwest (+Y, -X)
}

def get_view_cones(dr: int, dc: int):
    """
    Given an A* step (change in row, change in col), determines the active 
    View Cones based on the absolute world coordinates.

    Returns:
        facing_dir (str): The direction the rover is facing.
        Vl (str)        : Left view cone.
        Vh (str)        : Center (head) view cone.
        Vr (str)        : Right view cone.
    """
    # Normalize step just in case there are multi-cell jumps
    dr_norm = 0 if dr == 0 else int(dr / abs(dr))
    dc_norm = 0 if dc == 0 else int(dc / abs(dc))
    
    if dr_norm == 0 and dc_norm == 0:
        return 'Stationary', 'None', 'None', 'None'
    
    idx = STEP_TO_DIR_IDX.get((dr_norm, dc_norm), 0)
    
    vh = DIRECTIONS[idx]
    vl = DIRECTIONS[(idx - 1) % 8]  # Left of heading
    vr = DIRECTIONS[(idx + 1) % 8]  # Right of heading
    
    facing = vh
    return facing, vl, vh, vr
