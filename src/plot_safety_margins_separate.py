"""
plot_safety_margins_separate.py
--------------------------------
Generates three pairs of figures (Q-Learning + SARSA per pair) showing
the per-episode obstacle-clearance distance with 10% convergence markers
sourced from different signals:

  * Reward convergence  (matches the RQ1 table in the paper)
      Safety_Margins_QLearning_RewardConv.{png,pdf}
      Safety_Margins_SARSA_RewardConv.{png,pdf}

  * Safety convergence  (on the obstacle-clearance distance itself)
      Safety_Margins_QLearning_SafetyConv.{png,pdf}
      Safety_Margins_SARSA_SafetyConv.{png,pdf}

  * Both convergence types overlaid (red = reward, blue = safety)
      Safety_Margins_QLearning_BothConv.{png,pdf}
      Safety_Margins_SARSA_BothConv.{png,pdf}

Inputs:
  results/figures/safety_comparison_qlearning.csv
  results/figures/safety_comparison_sarsa.csv
  results/<Algorithm>/Scan_<N>/<ALG>_Th<theta>_Scan<N>_History.csv

Convergence definition (matches the paper's RQ1 table):
  smallest t such that for all u >= t,
      |s_u - mean(s_{t:})| <= tol * |mean(s_{t:})|
  where s = raw reward (reward source) or raw avg safety margin
  (safety source).  Returns 1-indexed episode number.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR    = os.path.join(_SCRIPT_DIR, "..", "results", "figures")
_PDF_DIR    = os.path.join(_SCRIPT_DIR, "..", "results", "pdfs")

# Same dash styles as the original combined figure
_DASH_STYLES = {
    "Scan_5":  (None, None),
    "Scan_10": (2, 2),
    "Scan_15": (1, 1),
    "Scan_20": (4, 2),
}

# Grayscale ramp per scan depth (shared across both algorithms).
_SCAN_COLOR = {
    "Scan_5":  "#000000",
    "Scan_10": "#444444",
    "Scan_15": "#888888",
    "Scan_20": "#BBBBBB",
}

# Marker colours per convergence source
_CONV_COLOR = {
    "reward": "red",
    "safety": "#1f77b4",   # blue
}


# ───────────────────────────────────────────────────────── helpers ──────

def _smooth(df, window=50):
    # `min_periods=1` keeps the smoothed series defined at the boundaries
    # (the centered window otherwise produces NaN for the first/last
    # `window/2` samples — and those NaNs would silently hide convergence
    # markers that happen to fall in the last 25 episodes).
    return df.rolling(window=window, center=True, min_periods=1).mean()


def _convergence_episode(series, tol=0.10):
    """
    Forward walk on the raw (un-smoothed) series.  Returns the smallest
    1-indexed episode t such that

        max_{u >= t} |s_u - mean(s_{t:})| <= tol * |mean(s_{t:})|.

    Returns None if the criterion never holds.
    """
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


def _load_history_row(algorithm_dirname, theta, scan, data_dir, metric_name):
    folder = os.path.join(data_dir, algorithm_dirname, f"Scan_{scan}")
    alg_token_variants = [
        algorithm_dirname.upper(),
        algorithm_dirname,
        algorithm_dirname.replace("-", "").upper(),
        algorithm_dirname.replace("-", ""),
    ]
    for tok in alg_token_variants:
        path = os.path.join(folder, f"{tok}_Th{theta}_Scan{scan}_History.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            if metric_name in df.index:
                return pd.to_numeric(df.loc[metric_name].values, errors="coerce")
    return None


def _load_reward_series(algorithm_dirname, theta, scan, data_dir):
    return _load_history_row(algorithm_dirname, theta, scan, data_dir, "Reward")


def _load_safety_series(algorithm_dirname, theta, scan, data_dir):
    return _load_history_row(algorithm_dirname, theta, scan, data_dir, "Avg_Safety_Margin")


# ───────────────────────────────────────────────────────── plotting ──────

def _plot_one_algorithm(csv_path, algorithm_name, out_png, out_pdf,
                        convergence_sources=("reward",),
                        y_limits=None, window=50,
                        algorithm_dirname=None, theta=60,
                        data_dir=None, tol=0.10, tol_by_source=None):
    """
    `convergence_sources` is a sequence of {"reward", "safety"} that
    selects which convergence markers (and which legend entries) are
    drawn on the figure.

    `tol_by_source` (optional) overrides the global `tol` per source —
    e.g. {"reward": 0.10, "safety": 0.15} uses a 10% band for reward
    convergence and a 15% band for safety convergence.
    """
    if data_dir is None:
        data_dir = os.path.join(_SCRIPT_DIR, "..", "results")
    if algorithm_dirname is None:
        algorithm_dirname = "Q-Learning" if algorithm_name == "Q-Learning" else "SARSA"
    if tol_by_source is None:
        tol_by_source = {}
    def _tol(src):
        return tol_by_source.get(src, tol)

    df = pd.read_csv(csv_path)
    scan_rows = df[df["Metric"].str.startswith("Scan_")].set_index("Metric")
    data = scan_rows.T.astype(float)
    data_smoothed = _smooth(data, window=window)

    scan_cols = sorted(
        [c for c in data_smoothed.columns if c.startswith("Scan_")],
        key=lambda x: int(x.split("_")[1]),
    )

    # Per-scan-depth convergence episodes for every requested source
    conv = {src: {} for src in convergence_sources}
    for col in scan_cols:
        scan = int(col.split("_")[1])
        if "reward" in conv:
            r = _load_reward_series(algorithm_dirname, theta, scan, data_dir)
            conv["reward"][col] = (
                _convergence_episode(r, tol=_tol("reward")) if r is not None else None
            )
        if "safety" in conv:
            s = _load_safety_series(algorithm_dirname, theta, scan, data_dir)
            conv["safety"][col] = (
                _convergence_episode(s, tol=_tol("safety")) if s is not None else None
            )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    episodes = np.arange(1, len(data_smoothed) + 1)

    # ── Curves ───────────────────────────────────────────────────────
    for col in scan_cols:
        depth = int(col.split("_")[1])
        dash  = _DASH_STYLES.get(col, (None, None))
        color = _SCAN_COLOR.get(col, "#000000")
        linestyle = "-" if dash == (None, None) else (0, dash)

        # Build legend label.  If a single source is shown, append its
        # convergence episode; if both are shown, append both (R=..,S=..).
        suffix = ""
        if len(convergence_sources) == 1:
            src = convergence_sources[0]
            ep = conv[src].get(col)
            if ep is not None:
                suffix = f"  (ep {ep})"
        else:
            parts = []
            if "reward" in conv and conv["reward"].get(col) is not None:
                parts.append(f"R={conv['reward'][col]}")
            if "safety" in conv and conv["safety"].get(col) is not None:
                parts.append(f"S={conv['safety'][col]}")
            if parts:
                suffix = "  (" + ", ".join(parts) + ")"

        ax.plot(
            episodes, data_smoothed[col].values,
            color=color, linewidth=2, linestyle=linestyle,
            label=f"Scan_{depth}{suffix}",
        )

    ax.set_title(
        f"Obstacle-Clearance Distance Over Episodes ({algorithm_name})",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Episodes", fontsize=12)
    ax.set_ylabel(f"Obstacle-Clearance Distance (cells, {window}-Ep Moving Avg)",
                  fontsize=12)
    if y_limits is not None:
        ax.set_ylim(y_limits)

    # ── Convergence markers on each curve ───────────────────────────
    y_lo, y_hi = ax.get_ylim()
    yspan = y_hi - y_lo

    for src in convergence_sources:
        marker_color = _CONV_COLOR[src]
        for col in scan_cols:
            conv_ep = conv[src].get(col)
            if conv_ep is None:
                continue
            idx = conv_ep - 1
            if idx < 0 or idx >= len(data_smoothed[col]):
                continue
            y_val = float(data_smoothed[col].iloc[idx])
            if pd.isna(y_val):
                continue
            ax.plot(
                conv_ep, y_val,
                marker="o", color=marker_color,
                markersize=5, markeredgecolor=marker_color,
                markerfacecolor=marker_color, linestyle="none",
                zorder=5,
            )

    # ── x-axis convergence-range bracket(s) ─────────────────────────
    bracket_offsets = {  # vertical stacking when both brackets are drawn
        "reward": 0.02,
        "safety": 0.06,
    }
    x_lo, x_hi = ax.get_xlim()
    xspan = x_hi - x_lo
    # If the bracket spans less than this fraction of the x-axis, collapse
    # the two endpoint labels into a single "min--max" label centred on
    # the bracket (avoids the "38090" overlap when the range is narrow).
    narrow_frac = 0.06

    for src in convergence_sources:
        eps = [v for v in conv[src].values() if v is not None]
        if not eps:
            continue
        eps_min, eps_max = min(eps), max(eps)
        c = _CONV_COLOR[src]
        y_b = y_lo + bracket_offsets[src] * yspan
        tick_h = 0.012 * yspan
        ax.plot([eps_min, eps_max], [y_b, y_b],
                color=c, linewidth=1.5, zorder=4, clip_on=False)
        ax.plot([eps_min, eps_min], [y_b - tick_h, y_b + tick_h],
                color=c, linewidth=1.5, zorder=4, clip_on=False)
        ax.plot([eps_max, eps_max], [y_b - tick_h, y_b + tick_h],
                color=c, linewidth=1.5, zorder=4, clip_on=False)

        if eps_min == eps_max:
            ax.annotate(f"{eps_min}",
                        xy=(eps_min, y_b - tick_h),
                        xytext=(0, -2), textcoords="offset points",
                        ha="center", va="top",
                        color=c, fontsize=8, fontweight="bold")
        elif (eps_max - eps_min) <= narrow_frac * xspan:
            # Endpoints too close to label separately — show one combined
            # "min--max" label at the bracket midpoint.
            mid = 0.5 * (eps_min + eps_max)
            ax.annotate(f"{eps_min}–{eps_max}",
                        xy=(mid, y_b - tick_h),
                        xytext=(0, -2), textcoords="offset points",
                        ha="center", va="top",
                        color=c, fontsize=8, fontweight="bold")
        else:
            ax.annotate(f"{eps_min}",
                        xy=(eps_min, y_b - tick_h),
                        xytext=(0, -2), textcoords="offset points",
                        ha="center", va="top",
                        color=c, fontsize=8, fontweight="bold")
            ax.annotate(f"{eps_max}",
                        xy=(eps_max, y_b - tick_h),
                        xytext=(0, -2), textcoords="offset points",
                        ha="center", va="top",
                        color=c, fontsize=8, fontweight="bold")

    # ── Legend ──────────────────────────────────────────────────────
    def _pct(src):
        return f"{int(round(_tol(src) * 100))}%"

    if len(convergence_sources) == 1:
        src = convergence_sources[0]
        legend_title = (
            f"Scan_Depth (red marker = reward convergence, {_pct(src)})"
            if src == "reward"
            else f"Scan_Depth (blue marker = safety convergence, {_pct(src)})"
        )
        ax.legend(title=legend_title, loc="best", fontsize=9)
    else:
        scan_legend = ax.legend(
            title="Scan_Depth (R = reward convergence ep, S = safety convergence ep)",
            loc="upper right", fontsize=8,
        )
        ax.add_artist(scan_legend)
        proxies = [
            mlines.Line2D([], [], color=_CONV_COLOR["reward"], marker="o",
                          markersize=6, linestyle="none",
                          label=f"Reward convergence ({_pct('reward')})"),
            mlines.Line2D([], [], color=_CONV_COLOR["safety"], marker="o",
                          markersize=6, linestyle="none",
                          label=f"Safety convergence ({_pct('safety')})"),
        ]
        ax.legend(handles=proxies, loc="lower right", fontsize=9,
                  title="Marker")

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf,            bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_png}")
    print(f"[SAVED] {out_pdf}")

    return conv


# ───────────────────────────────────────────────────────── driver ──────

def _shared_y_limits(qlearning_csv, sarsa_csv, window):
    def _rng(csv_path):
        df = pd.read_csv(csv_path)
        sr = df[df["Metric"].str.startswith("Scan_")].set_index("Metric").T.astype(float)
        sm = _smooth(sr, window=window)
        return np.nanmin(sm.values), np.nanmax(sm.values)
    lo_q, hi_q = _rng(qlearning_csv)
    lo_s, hi_s = _rng(sarsa_csv)
    lo, hi = min(lo_q, lo_s), max(hi_q, hi_s)
    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)


def plot_all(qlearning_csv=None, sarsa_csv=None,
             output_dir=None, pdf_dir=None,
             window=50, share_y=True,
             data_dir=None, theta=60, tol=0.10,
             tol_by_source=None):
    if tol_by_source is None:
        # Default: 10% band for reward, 15% band for safety
        # (safety is intrinsically noisier).
        tol_by_source = {"reward": 0.10, "safety": 0.15}
    if qlearning_csv is None:
        qlearning_csv = os.path.join(_FIG_DIR, "safety_comparison_qlearning.csv")
    if sarsa_csv is None:
        sarsa_csv = os.path.join(_FIG_DIR, "safety_comparison_sarsa.csv")
    if output_dir is None:
        output_dir = _FIG_DIR
    if pdf_dir is None:
        pdf_dir = _PDF_DIR

    y_limits = _shared_y_limits(qlearning_csv, sarsa_csv, window) if share_y else None

    pairs = [
        ("RewardConv", ("reward",)),
        ("SafetyConv", ("safety",)),
        ("BothConv",   ("reward", "safety")),
    ]

    all_conv = {"Q-Learning": {}, "SARSA": {}}

    for suffix, sources in pairs:
        ql_conv = _plot_one_algorithm(
            qlearning_csv, "Q-Learning",
            os.path.join(output_dir, f"Safety_Margins_QLearning_{suffix}.png"),
            os.path.join(pdf_dir,    f"Safety_Margins_QLearning_{suffix}.pdf"),
            convergence_sources=sources,
            y_limits=y_limits, window=window,
            algorithm_dirname="Q-Learning",
            theta=theta, data_dir=data_dir, tol=tol,
            tol_by_source=tol_by_source,
        )
        sarsa_conv = _plot_one_algorithm(
            sarsa_csv, "SARSA",
            os.path.join(output_dir, f"Safety_Margins_SARSA_{suffix}.png"),
            os.path.join(pdf_dir,    f"Safety_Margins_SARSA_{suffix}.pdf"),
            convergence_sources=sources,
            y_limits=y_limits, window=window,
            algorithm_dirname="SARSA",
            theta=theta, data_dir=data_dir, tol=tol,
            tol_by_source=tol_by_source,
        )
        if suffix == "BothConv":
            all_conv["Q-Learning"] = ql_conv
            all_conv["SARSA"]      = sarsa_conv

    # Print the reward-vs-safety comparison table that the user asked for.
    r_pct = int(round(tol_by_source.get("reward", tol) * 100))
    s_pct = int(round(tol_by_source.get("safety", tol) * 100))
    print("\n" + "=" * 78)
    print(f"  Convergence episode comparison  (reward tol = {r_pct}%, "
          f"safety tol = {s_pct}%, 1-indexed)")
    print("=" * 78)
    header = f"  {'Algorithm':<11} {'Scan':<6} {'Reward ep':>10} {'Safety ep':>10}   {'Reward first?'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for alg in ("Q-Learning", "SARSA"):
        for col in ("Scan_5", "Scan_10", "Scan_15", "Scan_20"):
            r_ep = all_conv[alg].get("reward", {}).get(col)
            s_ep = all_conv[alg].get("safety", {}).get(col)
            order = ""
            if r_ep is not None and s_ep is not None:
                if r_ep < s_ep:
                    order = f"yes  (reward {s_ep - r_ep} ep earlier)"
                elif r_ep > s_ep:
                    order = f"no   (safety {r_ep - s_ep} ep earlier)"
                else:
                    order = "tied"
            print(f"  {alg:<11} {col:<6} {str(r_ep):>10} {str(s_ep):>10}   {order}")
    print("=" * 72)


if __name__ == "__main__":
    plot_all()
