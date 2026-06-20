"""
compare_safety.py
-----------------
Aggregates Q-Learning safety margins across scan depths and saves a
comparison CSV used by hypothesis_tests.py and plot_safety_comparison.py.
"""

import os
import pandas as pd
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR  = os.path.join(_SCRIPT_DIR, "..", "results")
_DEFAULT_OUTPUT    = os.path.join(_SCRIPT_DIR, "..", "results", "figures",
                                  "safety_comparison_qlearning.csv")


def extract_safety(scan, data_dir, theta=60):
    """Read Avg_Safety_Margin row from Q-Learning History CSV for *scan* depth."""
    path = os.path.join(data_dir, "Q-Learning", f"Scan_{scan}",
                        f"QLEARNING_Th{theta}_Scan{scan}_History.csv")
    if not os.path.exists(path):
        print(f"Error: File not found — {path}")
        return None
    df = pd.read_csv(path, index_col=0)
    if "Avg_Safety_Margin" not in df.index:
        print(f"Error: Avg_Safety_Margin not found in {path}")
        return None
    return pd.to_numeric(df.loc["Avg_Safety_Margin"].values, errors='coerce')


def generate_comparison(data_dir=None, output_path=None, thresholds=None, theta=60):
    """Build and save a safety-margin comparison CSV for Q-Learning."""
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    if output_path is None:
        output_path = _DEFAULT_OUTPUT
    if thresholds is None:
        thresholds = [5, 10, 15, 20]

    series = {}
    for t in thresholds:
        s = extract_safety(t, data_dir, theta)
        if s is None:
            return
        series[t] = s

    min_len = min(len(v) for v in series.values())
    for t in thresholds:
        series[t] = series[t][:min_len]

    ref = thresholds[len(thresholds) // 2]
    episodes = [f"Ep_{i+1}" for i in range(min_len)]

    rows = [[f"Scan_{t}"] + list(series[t]) for t in thresholds]
    for t in thresholds:
        if t != ref:
            rows.append([f"Abs_Diff_{t}_{ref}"] + list(np.abs(series[t] - series[ref])))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows, columns=["Metric"] + episodes).to_csv(output_path, index=False)
    print(f"[DONE] Q-Learning safety comparison -> {output_path}")


if __name__ == "__main__":
    generate_comparison()
