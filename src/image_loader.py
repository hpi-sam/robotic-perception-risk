"""
image_loader.py
---------------
Discovers and loads paired RGB + depth images from two directories.
Frames are matched by sorted filename order.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional


# Supported image extensions
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _sorted_image_paths(directory: str) -> List[str]:
    """Return sorted list of image file paths in *directory*."""
    paths = []
    for f in sorted(os.listdir(directory)):
        if os.path.splitext(f)[1].lower() in _IMAGE_EXTS:
            paths.append(os.path.join(directory, f))
    return paths


def _parse_tum_file(filepath: str) -> List[Tuple[float, str]]:
    """Parse a TUM format timestamp file returning (timestamp, relative_path) tuples."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                data.append((float(parts[0]), parts[1]))
    return data

def discover_pairs(
    dataset_dir: str,
    rgb_dir_name: str = "rgb",
    depth_dir_name: str = "depth",
    max_time_diff: float = 0.03
) -> List[Tuple[str, str]]:
    """
    Match RGB and depth frames by TUM timestamp if rgb.txt and depth.txt exist.
    Otherwise, fallback to matching by sorted filename order.

    Parameters
    ----------
    dataset_dir : path to the overarching dataset folder.
    rgb_dir_name : folder name for RGB images.
    depth_dir_name : folder name for Depth images.
    max_time_diff : maximum allowed difference in seconds between paired frames.

    Returns
    -------
    List of (rgb_absolute_path, depth_absolute_path) tuples.
    """
    rgb_txt = os.path.join(dataset_dir, "rgb.txt")
    depth_txt = os.path.join(dataset_dir, "depth.txt")

    if os.path.exists(rgb_txt) and os.path.exists(depth_txt):
        # TUM format timestamp matching
        rgb_data = _parse_tum_file(rgb_txt)
        depth_data = _parse_tum_file(depth_txt)
        
        pairs = []
        depth_idx = 0
        depth_len = len(depth_data)
        
        for rgb_time, rgb_rel_path in rgb_data:
            best_diff = float('inf')
            best_depth_idx = -1
            
            # Simple window search forward for closest timestamp
            for i in range(depth_idx, depth_len):
                dt, dp = depth_data[i]
                diff = abs(rgb_time - dt)
                if diff < best_diff:
                    best_diff = diff
                    best_depth_idx = i
                
                if dt > rgb_time + max_time_diff:
                    break # since times are monotonic
            
            if best_diff <= max_time_diff and best_depth_idx != -1:
                rgb_abs = os.path.normpath(os.path.join(dataset_dir, rgb_rel_path))
                depth_abs = os.path.normpath(os.path.join(dataset_dir, depth_data[best_depth_idx][1]))
                pairs.append((rgb_abs, depth_abs))
                depth_idx = best_depth_idx # optimization for sequential search
        
        if len(pairs) == 0:
            raise FileNotFoundError(f"No matching image pairs found within {max_time_diff}s difference.")
        return pairs

    # Fallback: sorted directory matching
    rgb_dir = os.path.join(dataset_dir, rgb_dir_name)
    depth_dir = os.path.join(dataset_dir, depth_dir_name)
    
    rgb_paths = _sorted_image_paths(rgb_dir)
    depth_paths = _sorted_image_paths(depth_dir)

    n = min(len(rgb_paths), len(depth_paths))
    if n == 0:
        raise FileNotFoundError(
            f"No image pairs found.\n  rgb_dir={rgb_dir}  ({len(rgb_paths)} images)"
            f"\n  depth_dir={depth_dir}  ({len(depth_paths)} images)"
        )
    return list(zip(rgb_paths[:n], depth_paths[:n]))


def load_rgb(path: str) -> np.ndarray:
    """Load an RGB image as float32 array in [0, 1], shape (H, W, 3)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read RGB image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def load_depth(path: str, scale: float = 1e-3) -> np.ndarray:
    """
    Load a depth image as float32 array in metres.

    Supports:
      - 16-bit PNG (millimetres → metres by default via *scale*).
      - 32-bit float EXR / TIFF (assumed already in metres, *scale* ignored).

    Parameters
    ----------
    path  : image file path.
    scale : multiplier to convert raw pixel value to metres (default 1e-3).

    Returns
    -------
    depth : float32 array, shape (H, W), in metres.  Zero means invalid.
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"Cannot read depth image: {path}")

    if raw.dtype == np.float32 or raw.dtype == np.float64:
        return raw.astype(np.float32)

    # Assume 16-bit integer depth in millimetres
    return raw.astype(np.float32) * scale
