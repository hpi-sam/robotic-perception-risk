"""
post_convergence_ttest.py
-------------------------
Welch t-tests comparing Avg_Safety_Margin between consecutive scan-depth
thresholds (5 vs 10, 10 vs 15, 15 vs 20) within each RL algorithm, on
two different post-convergence slices:

  * Reward-conv slice  : episodes from each scan's reward-convergence
                         episode onwards (10% rule)
  * Safety-conv slice  : episodes from each scan's safety-convergence
                         episode onwards (15% rule)

Sources
-------
  results/<Algorithm>/Scan_<N>/<ALG>_Th60_Scan<N>_History.csv
    -> rows "Reward" and "Avg_Safety_Margin"
  Convergence definition matches the paper's RQ1 table:
    smallest t such that for all u >= t,
      |s_u - mean(s_{t:})| <= tol * |mean(s_{t:})|.

Outputs
-------
  - Console: full report (sources, sample sizes, means, t, p, d).
  - results/figures/post_convergence_ttests.md : markdown table with
    the same numbers.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_SCRIPT_DIR, "..", "results")
_MD_OUT      = os.path.join(_RESULTS_DIR, "figures", "post_convergence_ttests.md")


def _convergence_episode(series, tol):
    """Forward walk: smallest t (1-indexed) such that all r[t:] are within
    +/- tol of mean(r[t:]). Returns None if never satisfied."""
    s = np.asarray(pd.Series(series).dropna().values, dtype=float)
    n = len(s)
    if n == 0:
        return None
    for t in range(n):
        tail = s[t:]
        m = float(np.mean(tail))
        if m == 0:
            continue
        if np.all(np.abs(tail - m) <= tol * abs(m)):
            return t + 1
    return None


def _load_history(alg_dir, theta, scan, data_dir):
    alg_tok = "QLEARNING" if alg_dir == "Q-Learning" else "SARSA"
    path = os.path.join(data_dir, alg_dir, f"Scan_{scan}",
                        f"{alg_tok}_Th{theta}_Scan{scan}_History.csv")
    df = pd.read_csv(path, index_col=0)
    reward = pd.to_numeric(df.loc["Reward"].values, errors="coerce")
    safety = pd.to_numeric(df.loc["Avg_Safety_Margin"].values, errors="coerce")
    return reward, safety, path


def _cohens_d(a, b):
    """Pooled-variance Cohen's d (independent samples)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    s2 = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    if s2 <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / np.sqrt(s2))


def run_tests(theta=60,
              tol_reward=0.10,
              tol_safety=0.15,
              scans=(5, 10, 15, 20),
              algs=("Q-Learning", "SARSA"),
              data_dir=None):
    if data_dir is None:
        data_dir = _RESULTS_DIR

    # --- 1. Load everything and compute convergence episodes ------------
    history = {}
    conv    = {}
    for alg in algs:
        history[alg] = {}
        conv[alg]    = {}
        for s in scans:
            r, m, path = _load_history(alg, theta, s, data_dir)
            history[alg][s] = {"reward": r, "safety": m, "path": path}
            conv[alg][s] = {
                "reward": _convergence_episode(r, tol_reward),
                "safety": _convergence_episode(m, tol_safety),
            }

    # --- 2. Build per-pair, per-source comparisons ----------------------
    pairs = [(a, b) for a, b in zip(scans, scans[1:])]
    sources = [
        ("reward", "Reward convergence (10%)"),
        ("safety", "Safety convergence (15%)"),
    ]

    rows = []
    for alg in algs:
        for src_key, src_label in sources:
            for s_a, s_b in pairs:
                ep_a = conv[alg][s_a][src_key]
                ep_b = conv[alg][s_b][src_key]
                m_a = history[alg][s_a]["safety"]
                m_b = history[alg][s_b]["safety"]

                if ep_a is None or ep_b is None:
                    rows.append({
                        "Algorithm": alg, "Convergence": src_label,
                        "Pair": f"{s_a} -> {s_b}",
                        "ep_a": ep_a, "ep_b": ep_b,
                        "n_a": None, "n_b": None,
                        "mean_a": None, "mean_b": None,
                        "delta": None, "t": None, "p": None, "d": None,
                        "sig": "missing convergence",
                    })
                    continue

                # Post-convergence slice (1-indexed -> 0-indexed)
                a = pd.Series(m_a[ep_a - 1:]).dropna().values
                b = pd.Series(m_b[ep_b - 1:]).dropna().values

                t_stat, p_val = stats.ttest_ind(b, a, equal_var=False)
                d = _cohens_d(b, a)
                rows.append({
                    "Algorithm": alg, "Convergence": src_label,
                    "Pair": f"{s_a} -> {s_b}",
                    "ep_a": ep_a, "ep_b": ep_b,
                    "n_a": int(len(a)), "n_b": int(len(b)),
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "delta": float(np.mean(b) - np.mean(a)),
                    "t": float(t_stat), "p": float(p_val), "d": d,
                    "sig": _sig_marker(p_val),
                })

    df = pd.DataFrame(rows)
    _print_report(df, conv, history, tol_reward, tol_safety)
    _write_markdown(df, tol_reward, tol_safety)
    return df


def _sig_marker(p):
    if p is None or np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _print_report(df, conv, history, tol_r, tol_s):
    print("=" * 88)
    print(" Sources")
    print("=" * 88)
    for alg, scans_dict in history.items():
        for s, h in scans_dict.items():
            rel = os.path.relpath(h["path"], _SCRIPT_DIR)
            print(f"  {alg:<11}  Scan_{s:<3}  ->  {rel}")

    print("\n" + "=" * 88)
    print(f" Convergence episodes  (reward tol = {int(tol_r*100)}%, "
          f"safety tol = {int(tol_s*100)}%)")
    print("=" * 88)
    print(f"  {'Alg':<11} {'Scan':>5} {'Reward conv ep':>16} {'Safety conv ep':>16}")
    for alg in conv:
        for s in conv[alg]:
            r = conv[alg][s]["reward"]
            sf = conv[alg][s]["safety"]
            print(f"  {alg:<11} {s:>5} {str(r):>16} {str(sf):>16}")

    print("\n" + "=" * 88)
    print(" Welch t-tests on Avg_Safety_Margin "
          "(post-convergence, two-sided, H0: mean_a == mean_b)")
    print("=" * 88)
    print(f"  {'Alg':<11} {'Conv slice':<28} {'Pair':<11} "
          f"{'n_a':>4} {'n_b':>4} "
          f"{'mean_a':>8} {'mean_b':>8} {'delta':>7} "
          f"{'t':>7} {'p':>9} {'d':>6}  sig")
    print("  " + "-" * 86)
    for _, r in df.iterrows():
        def fmt(x, w, prec):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "n/a".rjust(w)
            return f"{x:>{w}.{prec}f}" if isinstance(x, float) else f"{x:>{w}}"
        print(f"  {r['Algorithm']:<11} {r['Convergence']:<28} {r['Pair']:<11} "
              f"{fmt(r['n_a'],4,0)} {fmt(r['n_b'],4,0)} "
              f"{fmt(r['mean_a'],8,3)} {fmt(r['mean_b'],8,3)} "
              f"{fmt(r['delta'],7,3)} "
              f"{fmt(r['t'],7,3)} {fmt(r['p'],9,4)} {fmt(r['d'],6,3)}  {r['sig']}")
    print("=" * 88)
    print(" Significance: *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05")
    print(" Bonferroni alpha for 12 tests = 0.05/12 = 0.00417")
    print("=" * 88)


def _write_markdown(df, tol_r, tol_s):
    lines = [
        "# Post-convergence Welch t-tests on Avg_Safety_Margin",
        "",
        f"Reward tolerance: **{int(tol_r*100)}%**, safety tolerance: "
        f"**{int(tol_s*100)}%**. Two-sided Welch t-test on the "
        "obstacle-clearance distance (per-episode `Avg_Safety_Margin`) "
        "between consecutive scan depths, using each scan's own "
        "convergence episode as the start of its post-convergence slice.",
        "",
        "| Algorithm | Convergence slice | Pair | n₁ | n₂ | mean₁ | mean₂ | Δμ | t | p | Cohen's d | Sig |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        def f(x, prec=3):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "n/a"
            return f"{x:.{prec}f}" if isinstance(x, float) else f"{x}"
        lines.append(
            f"| {r['Algorithm']} | {r['Convergence']} | {r['Pair']} | "
            f"{f(r['n_a'], 0)} | {f(r['n_b'], 0)} | "
            f"{f(r['mean_a'])} | {f(r['mean_b'])} | {f(r['delta'])} | "
            f"{f(r['t'])} | {f(r['p'], 4)} | {f(r['d'])} | {r['sig']} |"
        )
    lines += [
        "",
        "**Legend.** *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05. "
        "Bonferroni-adjusted α for 12 tests = 0.05/12 ≈ 0.00417.",
        "",
    ]
    os.makedirs(os.path.dirname(_MD_OUT), exist_ok=True)
    with open(_MD_OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n[SAVED] {_MD_OUT}")


if __name__ == "__main__":
    run_tests()
