"""
plot_safety_comparison.py
-------------------------
Generates a box-plot of safety margins per scan depth threshold.
Reads from results/figures/safety_comparison_qlearning.csv (produced by
compare_safety.py).  Can also be called programmatically with a custom path.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_SCRIPT_DIR, "..", "results", "figures",
                            "safety_comparison_qlearning.csv")
_DEFAULT_OUT = os.path.join(_SCRIPT_DIR, "..", "results", "figures",
                            "safety_comparison_boxplot.png")


def plot_safety(csv_path=None, output_path=None):
    """Box-plot safety margins across scan depth thresholds."""
    if csv_path is None:
        csv_path = _DEFAULT_CSV
    if output_path is None:
        output_path = _DEFAULT_OUT

    df = pd.read_csv(csv_path)

    # Keep only Scan_* rows
    scan_rows = df[df['Metric'].str.startswith('Scan_')].copy()
    scan_rows = scan_rows.set_index('Metric')

    plot_data = scan_rows.T.astype(float)
    plot_data_smoothed = plot_data.rolling(window=50, center=True).mean()

    plot_data_reset = plot_data_smoothed.reset_index(drop=True)
    box_plot_data = plot_data_reset.melt(var_name='Metric', value_name='Value')
    box_plot_data = box_plot_data.dropna(subset=['Value'])

    metrics = box_plot_data['Metric'].unique().tolist()
    colors = sns.color_palette('Set2', len(metrics))
    metric_palette = {m: colors[i] for i, m in enumerate(metrics)}

    plt.figure(figsize=(14, 6))
    sns.set_theme(style="darkgrid")

    scan_order = sorted(
        [m for m in metrics if m.startswith('Scan_')],
        key=lambda x: int(x.split('_')[1])
    )
    sns.boxplot(data=box_plot_data, x='Metric', y='Value',
                order=scan_order, palette=metric_palette)

    plt.xlabel('Scan Depth Threshold', fontsize=12)
    plt.ylabel('Safety Score', fontsize=12)
    plt.title('Safety Margins per Scan Depth Threshold', fontsize=14, fontweight='bold')

    ax = plt.gca()
    ax.set_xticklabels([s.split('_')[1] for s in scan_order])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] Safety comparison box-plot -> {output_path}")


if __name__ == "__main__":
    plot_safety()
