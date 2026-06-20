"""
generate_paper_figures.py
-------------------------
Generates figures for the paper:
  Fig 1.1 — Reward Convergence (Scan 5, 15, 25) for Q-Learning and SARSA
  Fig 1.2 — Reward vs Perception Entropy correlation for Q-Learning and SARSA

Reads from results/{ALGORITHM}/Scan_{N}/  (produced by the main pipeline).
Saves figures to results/figures/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR   = os.path.join(_SCRIPT_DIR, "..", "results")
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "results", "figures")


def find_file(algorithm, th, scan, data_dir=None):
    """Locate the History CSV for *algorithm* / Th*th* / Scan*scan*."""
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR

    subdir = os.path.join(data_dir, algorithm, f"Scan_{scan}")
    patterns = [
        f"{algorithm}_Th{th}_Scan{scan}_History.csv",
        f"{algorithm.upper()}_Th{th}_Scan{scan}_History.csv",
        f"{algorithm.capitalize()}_Th{th}_Scan{scan}_History.csv",
        f"{algorithm.lower()}_Th{th}_Scan{scan}_History.csv",
    ]
    for p in patterns:
        full_path = os.path.join(subdir, p)
        if os.path.exists(full_path):
            return full_path
    return None


def smooth_data(data, window=20):
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean()


def load_metric(agent_name, th, scan, metric_name, data_dir=None):
    csv_path = find_file(agent_name, th, scan, data_dir)
    if not csv_path:
        print(f"Warning: File not found for {agent_name} Th{th} Scan{scan}")
        return None
    df = pd.read_csv(csv_path, index_col=0)
    if metric_name not in df.index:
        print(f"Warning: Metric '{metric_name}' not in {csv_path}")
        return None
    return pd.to_numeric(df.loc[metric_name].values, errors='coerce')


def generate_fig1_1(agent_name, output_dir=None, data_dir=None, thresholds=None, theta=60):
    """Convergence plot: cumulative reward vs episode for each scan depth."""
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR
    if thresholds is None:
        thresholds = [5, 10, 15, 20]

    print(f"Generating Fig 1.1 for {agent_name}...")
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 7))

    colors = sns.color_palette("viridis", len(thresholds))

    for i, s in enumerate(thresholds):
        rewards = load_metric(agent_name, theta, s, "Reward", data_dir)
        if rewards is not None:
            episodes = np.arange(len(rewards))
            smoothed = smooth_data(rewards, window=20)
            plt.plot(episodes, smoothed,
                     label=f"Scan Depth Threshold: {s}",
                     color=colors[i], linewidth=2.5)
            plt.plot(episodes, rewards, color=colors[i], alpha=0.1, linewidth=1)

    plt.title(f"{agent_name} Training Convergence: Impact of Scan Depth",
              fontsize=18, fontweight='bold')
    plt.xlabel("Episode Number", fontsize=14)
    plt.ylabel("Reward (Moving Avg, window=20)", fontsize=14)
    plt.legend(title="Perception Setting")

    out_name = f"Fig_rq1_1_{agent_name.replace('-', '_')}.png"
    out_path = os.path.join(output_dir, out_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def generate_fig1_2(agent_name, output_dir=None, data_dir=None, thresholds=None, theta=60):
    """Correlation plot: episode reward vs perception entropy (Spearman ρ)."""
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR
    if thresholds is None:
        thresholds = [5, 10, 15, 20]

    print(f"Generating Fig 1.2 for {agent_name}...")
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 8))

    colors = sns.color_palette("magma", len(thresholds))

    for i, s in enumerate(thresholds):
        rewards = load_metric(agent_name, theta, s, "Reward", data_dir)
        entropy = load_metric(agent_name, theta, s, "Exploration_Entropy", data_dir)

        if rewards is not None and entropy is not None:
            min_len = min(len(rewards), len(entropy))
            r_f, e_f = rewards[:min_len], entropy[:min_len]
            mask = ~np.isnan(r_f) & ~np.isnan(e_f)
            r_final, e_final = r_f[mask], e_f[mask]

            rho, p_val = spearmanr(e_final, r_final)
            label_str = f"Scan Depth Threshold: {s} (ρ={rho:.2f}, p={p_val:.1e})"

            sns.regplot(
                x=e_final, y=r_final,
                label=label_str,
                color=colors[i],
                scatter_kws={'alpha': 0.3, 's': 20},
                line_kws={'linewidth': 2, 'linestyle': '--'},
            )

    plt.title(f"Statistical Analysis: Performance vs. Uncertainty ({agent_name})",
              fontsize=18, fontweight='bold')
    plt.xlabel("Perception Entropy", fontsize=14)
    plt.ylabel("Episode Reward", fontsize=14)
    plt.legend(title="Perception Setting")

    out_name = f"Fig_rq1_2_{agent_name.replace('-', '_')}.png"
    out_path = os.path.join(output_dir, out_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def generate_all_paper_figures(output_dir=None, data_dir=None, thresholds=None, theta=60):
    """Generate all paper figures for Q-Learning and SARSA."""
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR

    for agent in ("QLEARNING", "SARSA"):
        generate_fig1_1(agent, output_dir, data_dir, thresholds, theta)
        generate_fig1_2(agent, output_dir, data_dir, thresholds, theta)

    print("\n[ALL DONE] Figures 1.1 and 1.2 for both agents generated.")


if __name__ == "__main__":
    generate_all_paper_figures()
