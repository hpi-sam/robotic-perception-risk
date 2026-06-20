"""
plot_comparison_qlearning_vs_sarsa.py
--------------------------------------
Side-by-side box-plot and moving-average line-plot comparing Q-Learning vs
SARSA safety margins across all scan depth thresholds.

Reads from results/figures/safety_comparison_qlearning.csv and
results/figures/safety_comparison_sarsa.csv (produced by compare_safety.py
and compare_safety_sarsa.py).
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_QL_CSV    = os.path.join(_SCRIPT_DIR, "..", "results", "figures",
                                  "safety_comparison_qlearning.csv")
_DEFAULT_SARSA_CSV = os.path.join(_SCRIPT_DIR, "..", "results", "figures",
                                  "safety_comparison_sarsa.csv")
_DEFAULT_OUTPUT    = os.path.join(_SCRIPT_DIR, "..", "results", "figures",
                                  "comparison_plot.png")


def plot_comparison(qlearning_csv=None, sarsa_csv=None, output_path=None):
    """
    Generate comparison plots for Q-Learning vs SARSA.

    Parameters
    ----------
    qlearning_csv : str  path to Q-Learning comparison CSV
    sarsa_csv     : str  path to SARSA comparison CSV
    output_path   : str  where to save the figure (None → plt.show())
    """
    if qlearning_csv is None:
        qlearning_csv = _DEFAULT_QL_CSV
    if sarsa_csv is None:
        sarsa_csv = _DEFAULT_SARSA_CSV
    if output_path is None:
        output_path = _DEFAULT_OUTPUT

    qlearning_df = pd.read_csv(qlearning_csv)
    sarsa_df     = pd.read_csv(sarsa_csv)

    # Keep only Scan_* rows
    qlearning_data = qlearning_df[qlearning_df['Metric'].str.startswith('Scan_')].set_index('Metric')
    sarsa_data     = sarsa_df[sarsa_df['Metric'].str.startswith('Scan_')].set_index('Metric')

    qlearning_plot    = qlearning_data.T.astype(float)
    sarsa_plot        = sarsa_data.T.astype(float)
    qlearning_smoothed = qlearning_plot.rolling(window=50, center=True).mean()
    sarsa_smoothed     = sarsa_plot.rolling(window=50, center=True).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    sns.set_theme(style="darkgrid")

    # ── Box plots ────────────────────────────────────────────────────────────
    qlearning_melt = qlearning_smoothed.reset_index(drop=True).melt(
        var_name='Metric', value_name='Value')
    qlearning_melt['Algorithm'] = 'Q-Learning'

    sarsa_melt = sarsa_smoothed.reset_index(drop=True).melt(
        var_name='Metric', value_name='Value')
    sarsa_melt['Algorithm'] = 'SARSA'

    combined_data = pd.concat([qlearning_melt, sarsa_melt], ignore_index=True)
    combined_data['Group'] = combined_data['Metric'] + '\n' + combined_data['Algorithm']

    scan_metrics = sorted(
        [m for m in combined_data['Metric'].unique() if m.startswith('Scan_')],
        key=lambda x: int(x.split('_')[1])
    )
    group_order   = [f'{m}\n{alg}' for m in scan_metrics for alg in ('Q-Learning', 'SARSA')]
    group_palette = {g: '#1f77b4' if 'Q-Learning' in g else '#ff7f0e' for g in group_order}

    sns.boxplot(data=combined_data.dropna(subset=['Value']),
                x='Group', y='Value',
                order=group_order, palette=group_palette, ax=ax1)

    legend_handles = [
        mpatches.Patch(color='#1f77b4', label='Q-Learning'),
        mpatches.Patch(color='#ff7f0e', label='SARSA'),
    ]
    ax1.set_xlabel('Scan Depth Threshold', fontsize=12)
    ax1.set_ylabel('Safety Score', fontsize=12)
    ax1.set_title('Box Plot: Safety Margins Comparison (Q-Learning vs SARSA)',
                  fontsize=13, fontweight='bold')
    ax1.legend(handles=legend_handles, title='Algorithm', fontsize=10, loc='upper left')

    # ── Moving-average line plots ─────────────────────────────────────────────
    episodes_arr = np.arange(len(qlearning_smoothed))
    scan_cols    = [c for c in qlearning_smoothed.columns if c.startswith('Scan_')]

    line_data = []
    for col in scan_cols:
        for ep, val in zip(episodes_arr, qlearning_smoothed[col]):
            line_data.append({'Episode': ep, 'Safety_Score': val,
                              'Algorithm': 'Q-Learning', 'Scan_Depth': col})
        for ep, val in zip(episodes_arr, sarsa_smoothed[col]):
            line_data.append({'Episode': ep, 'Safety_Score': val,
                              'Algorithm': 'SARSA', 'Scan_Depth': col})

    line_df     = pd.DataFrame(line_data)
    dash_styles = [(None, None), (2, 2), (1, 1), (4, 2), (1, 3), (3, 1, 1, 1)]
    dashes      = {col: dash_styles[i % len(dash_styles)] for i, col in enumerate(scan_cols)}

    sns.lineplot(data=line_df, x='Episode', y='Safety_Score',
                 hue='Algorithm', style='Scan_Depth',
                 linewidth=2, ax=ax2,
                 palette=['#1f77b4', '#ff7f0e'], dashes=dashes)

    ax2.set_xlabel('Episodes', fontsize=12)
    ax2.set_ylabel('Safety Score (50-Episode Moving Average)', fontsize=12)
    ax2.set_title('Moving Average: Safety Margins Over Episodes',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='best', ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Comparison plot -> {output_path}")


if __name__ == "__main__":
    plot_comparison()
