"""
plot_results.py
---------------
Generates per-scan-depth comparison plots (Q-Learning vs SARSA) for all
standard metrics: reward, path length, TD error, goal success, safety margin,
tension, and perception entropy.

Reads from results/{ALGORITHM}/Scan_{N}/ and saves figures to results/figures/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR   = os.path.join(_SCRIPT_DIR, "..", "results")
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "results", "figures")


def smooth_data(data, window=20):
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean()


def find_file(algorithm, th, scan, data_dir=None):
    """Locate the History CSV for *algorithm* / Th*th* / Scan*scan*."""
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR

    subdir = os.path.join(data_dir, algorithm, f"Scan_{scan}")
    patterns = [
        f"{algorithm}_Th{th}_Scan{scan}_History.csv",
        f"{algorithm.upper()}_Th{th}_Scan{scan}_History.csv",
        f"{algorithm.capitalize()}_Th{th}_Scan{scan}_History.csv",
    ]
    for p in patterns:
        full_path = os.path.join(subdir, p)
        if os.path.exists(full_path):
            return full_path
    return None


def plot_rl_comparison(th=60, scan=5, data_dir=None, output_dir=None):
    """Plot all metrics for Q-Learning vs SARSA at a given theta/scan depth."""
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR

    q_file = find_file("QLEARNING", th, scan, data_dir)
    s_file = find_file("SARSA", th, scan, data_dir)

    if not q_file or not s_file:
        print(f"Error: Could not find history files for Th{th} Scan{scan}.")
        return

    print(f"Loading data from:\n  Q: {q_file}\n  S: {s_file}")
    q_df = pd.read_csv(q_file, index_col=0)
    s_df = pd.read_csv(s_file, index_col=0)

    potential_metrics = {
        "Reward":             "Cumulative Reward",
        "Path_Length":        "Episode Path Length",
        "TD_Error":           "TD Error (Learning Surprise)",
        "Goal_Reached":       "Goal Success (Rolling %)",
        "Avg_Safety_Margin":  "Average Safety Margin",
        "Avg_Tension":        "Environmental Tension",
        "Exploration_Entropy":"Perception Entropy",
    }

    metrics = [m for m in potential_metrics if m in q_df.index]
    titles  = [potential_metrics[m] for m in metrics]

    sns.set_theme(style="whitegrid", context="notebook")
    n_metrics = len(metrics)
    rows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(15, 6 * rows))
    fig.suptitle(f"Algorithm Comparison (Theta: {th}, Scan Depth: {scan})",
                 fontsize=22, fontweight='bold', y=0.98)

    colors  = sns.color_palette("Set1", 2)
    q_color = colors[1]
    s_color = colors[0]
    axes    = axes.flatten()

    q_success = q_df.loc["Goal_Reached"].values.astype(float)
    s_success = s_df.loc["Goal_Reached"].values.astype(float)
    q_first   = np.where(q_success == 1)[0]
    s_first   = np.where(s_success == 1)[0]
    start_idx = min(
        q_first[0] if len(q_first) > 0 else len(q_success),
        s_first[0] if len(s_first) > 0 else len(s_success),
    )

    for i, metric in enumerate(metrics):
        ax     = axes[i]
        q_data = q_df.loc[metric].values.astype(float)
        s_data = s_df.loc[metric].values.astype(float)
        x_idx  = np.arange(len(q_data))

        if metric == "Path_Length":
            q_plot, s_plot = q_data[start_idx:], s_data[start_idx:]
            x_idx = x_idx[start_idx:]
            ax.set_title(f"{titles[i]} (From first success)", fontweight='bold')
        elif metric == "Goal_Reached":
            q_plot, s_plot = smooth_data(q_data, 50) * 100, smooth_data(s_data, 50) * 100
            ax.set_ylabel("Success Rate (%)")
        elif metric in ("Reward", "TD_Error"):
            q_plot, s_plot = smooth_data(q_data, 20), smooth_data(s_data, 20)
            ax.set_ylabel("Value (Smoothed)")
        else:
            q_plot, s_plot = q_data, s_data
            ax.set_ylabel("Value")

        ax.plot(x_idx, q_plot, label="Q-Learning", color=q_color, linewidth=2.5)
        ax.plot(x_idx, s_plot, label="SARSA",      color=s_color, linewidth=2.5, alpha=0.85)
        ax.set_title(titles[i], fontweight='bold', fontsize=14)
        ax.set_xlabel("Episode")
        ax.legend(frameon=True, facecolor='white', framealpha=0.8)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, f"comparison_results_Th{th}_Scan{scan}.png")
    plt.savefig(output_name, dpi=300)
    plt.close()
    print(f"[DONE] Chart saved to {output_name}")


if __name__ == "__main__":
    scan_depths = [5, 10, 15, 20]
    for s in scan_depths:
        print(f"\n--- Generating Comparison for Scan Depth: {s} ---")
        plot_rl_comparison(th=60, scan=s)
