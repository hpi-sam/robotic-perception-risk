"""
hypothesis_tests.py
-------------------
Nonparametric hypothesis tests (Mann-Whitney U) comparing safety margins
across scan depth thresholds for Q-Learning and SARSA.

Reads from results/figures/safety_comparison_qlearning.csv and
results/figures/safety_comparison_sarsa.csv (produced by compare_safety.py
and compare_safety_sarsa.py).
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_QL_CSV = os.path.join(_SCRIPT_DIR, "..", "results", "figures",
                               "safety_comparison_qlearning.csv")
_DEFAULT_SARSA_CSV = os.path.join(_SCRIPT_DIR, "..", "results", "figures",
                                  "safety_comparison_sarsa.csv")


def run_tests(ql_csv=None, sarsa_csv=None):
    """Run Mann-Whitney U tests comparing scan depth thresholds."""
    if ql_csv is None:
        ql_csv = _DEFAULT_QL_CSV
    if sarsa_csv is None:
        sarsa_csv = _DEFAULT_SARSA_CSV

    qlearning_df = pd.read_csv(ql_csv)
    sarsa_df = pd.read_csv(sarsa_csv)

    def _extract_row(df, label):
        row = df[df['Metric'] == label]
        if row.empty:
            raise KeyError(f"Row '{label}' not found in CSV.")
        return row.iloc[0, 1:].astype(float).values

    qlearning_scan5  = _extract_row(qlearning_df, 'Scan_5')
    qlearning_scan10 = _extract_row(qlearning_df, 'Scan_10')
    qlearning_scan15 = _extract_row(qlearning_df, 'Scan_15')
    qlearning_scan20 = _extract_row(qlearning_df, 'Scan_20')

    sarsa_scan5  = _extract_row(sarsa_df, 'Scan_5')
    sarsa_scan10 = _extract_row(sarsa_df, 'Scan_10')
    sarsa_scan15 = _extract_row(sarsa_df, 'Scan_15')
    sarsa_scan20 = _extract_row(sarsa_df, 'Scan_20')

    print("=" * 70)
    print("NONPARAMETRIC HYPOTHESIS TESTS: Mann-Whitney U Test")
    print("=" * 70)
    print("\nNull Hypothesis (H0): The two distributions are the same")
    print("Alternative Hypothesis (H1): The two distributions are different")
    print("Significance Level: α = 0.05\n")

    def _report(label_a, data_a, label_b, data_b):
        stat, p = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        sig = 'SIGNIFICANT' if p < 0.05 else 'NOT SIGNIFICANT'
        print(f"  {label_a} vs {label_b}:")
        print(f"    U-statistic : {stat:.2f}")
        print(f"    p-value     : {p:.6f}")
        print(f"    Result      : {sig} (α=0.05)")
        print(f"    Mean {label_a}: {np.mean(data_a):.4f} ± {np.std(data_a):.4f}")
        print(f"    Mean {label_b}: {np.mean(data_b):.4f} ± {np.std(data_b):.4f}")
        return p

    print("-" * 70)
    print("Q-LEARNING ALGORITHM")
    print("-" * 70)
    p_q5_q10  = _report("Scan_5",  qlearning_scan5,  "Scan_10", qlearning_scan10)
    p_q10_q15 = _report("Scan_10", qlearning_scan10, "Scan_15", qlearning_scan15)
    p_q15_q20 = _report("Scan_15", qlearning_scan15, "Scan_20", qlearning_scan20)

    print("\n" + "-" * 70)
    print("SARSA ALGORITHM")
    print("-" * 70)
    p_s5_s10  = _report("Scan_5",  sarsa_scan5,  "Scan_10", sarsa_scan10)
    p_s10_s15 = _report("Scan_10", sarsa_scan10, "Scan_15", sarsa_scan15)
    p_s15_s20 = _report("Scan_15", sarsa_scan15, "Scan_20", sarsa_scan20)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nQ-LEARNING:")
    print(f"  Scan_5  vs Scan_10: p={p_q5_q10:.6f}  {'✓ SIGNIFICANT' if p_q5_q10 < 0.05 else '✗ Not significant'}")
    print(f"  Scan_10 vs Scan_15: p={p_q10_q15:.6f}  {'✓ SIGNIFICANT' if p_q10_q15 < 0.05 else '✗ Not significant'}")
    print(f"  Scan_15 vs Scan_20: p={p_q15_q20:.6f}  {'✓ SIGNIFICANT' if p_q15_q20 < 0.05 else '✗ Not significant'}")
    print("\nSARSA:")
    print(f"  Scan_5  vs Scan_10: p={p_s5_s10:.6f}  {'✓ SIGNIFICANT' if p_s5_s10 < 0.05 else '✗ Not significant'}")
    print(f"  Scan_10 vs Scan_15: p={p_s10_s15:.6f}  {'✓ SIGNIFICANT' if p_s10_s15 < 0.05 else '✗ Not significant'}")
    print(f"  Scan_15 vs Scan_20: p={p_s15_s20:.6f}  {'✓ SIGNIFICANT' if p_s15_s20 < 0.05 else '✗ Not significant'}")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
