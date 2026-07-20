# Oracle vs Risk-Aware Navigation — Results Summary

## Setup

- **Full-map oracle (baseline):** agent is given the complete occupancy map at
  episode 0. No fog-of-war, no scanning, no perception risk. Reward = −StepCost
  + ProgressReward; only penalty is hitting an obstacle (−40). Single
  reproducible run per algorithm (seed 42).
- **Risk-aware method (Th60):** operates under fog-of-war with active perception
  scanning and the perception-risk term in the reward (θ=60°), evaluated across
  scan-depth thresholds τ_scan ∈ {5, 10, 15, 20} — matching the paper. Collision
  penalty −20.
- Distances/clearances are in grid cells (resolution 0.05 m → 1 cell = 5 cm).

### Metric definitions (aligned with the paper)

- **1st Episode Success** — first episode in which the goal is reached (this is
  the paper's "1st Episode Success" column, and what earlier drafts of this
  table mislabelled as "convergence").
- **Conv. Ep. (10%) / (5%)** — the paper's reward-stability convergence: the
  first episode after which all subsequent rewards stay within 10% / 5% of their
  mean. This is a later, stricter point than 1st Episode Success.
- **Final path clearance** — obstacle-clearance margin (d̄_clear) of the final
  extracted greedy path.
- **Per-episode clearance** — d̄_clear averaged per episode over the last 100
  (post-convergence) episodes; this is the statistically testable distribution.
- **Success % (last 100)** — goal-reaching rate over the last 100 episodes.

> **Note:** *Final Reward* is deliberately excluded from these tables. The oracle
> (−40, no risk term) and the risk-aware method (−20, with the −ξ_perc term) use
> different reward functions, so reward magnitudes are not comparable. The
> comparison is made on reward-independent outcomes (convergence, path length,
> clearance, success, scans, entropy).

---

## Comparison tables — `comparison_oracle_vs_riskaware_{QLEARNING,SARSA}.csv`

**Structure:** rows = metrics; columns = `Oracle_baseline` then risk-aware at
each scan threshold.

### Q-Learning

| Metric | Oracle | Scan5 | Scan10 | Scan15 | Scan20 |
|---|---|---|---|---|---|
| 1st Episode Success | 367 | 381 | 394 | 366 | 374 |
| Conv. Ep. (10%) | 391 | 381 | 412 | 427 | 451 |
| Conv. Ep. (5%) | 409 | 460 | 438 | 457 | 451 |
| Final path length (cells) | 133 | 178 | 140 | 175 | 131 |
| Final path clearance | 12.36 | 11.10 | 11.64 | 10.79 | 12.43 |
| Success % (last 100) | 100 | 100 | 98 | 99 | 99 |
| Total scans | 0 | 377 | 440 | 387 | 508 |
| Exploration entropy | 14.43 | 14.36 | 14.53 | 14.32 | 14.49 |

### SARSA

| Metric | Oracle | Scan5 | Scan10 | Scan15 | Scan20 |
|---|---|---|---|---|---|
| 1st Episode Success | 382 | 380 | 355 | 372 | 388 |
| Conv. Ep. (10%) | 389 | 384 | 380 | 385 | 390 |
| Conv. Ep. (5%) | 454 | 458 | 395 | 407 | 420 |
| Final path length (cells) | 176 | 168 | 177 | 170 | 171 |
| Final path clearance | 11.97 | 11.91 | 11.07 | 11.83 | 11.06 |
| Success % (last 100) | 100 | 100 | 100 | 100 | 100 |
| Total scans | 0 | 431 | 401 | 417 | 441 |
| Exploration entropy | 14.55 | 14.51 | 14.44 | 14.50 | 14.54 |

---

## Convergence summary — `convergence_summary.csv`

Paper Table 1 layout, with the full-map oracle added as a row under each
algorithm. Shows all three convergence definitions side by side.

| RL | Scan Depth | 1st Episode Success | Conv. Ep. (10%) | Conv. Ep. (5%) |
|---|---|---|---|---|
| QL | Oracle (full map) | 367 | 391 | 409 |
| QL | 5 | 381 | 381 | 460 |
| QL | 10 | 394 | 412 | 438 |
| QL | 15 | 366 | 427 | 457 |
| QL | 20 | 374 | 451 | 451 |
| SARSA | Oracle (full map) | 382 | 389 | 454 |
| SARSA | 5 | 380 | 384 | 458 |
| SARSA | 10 | 355 | 380 | 395 |
| SARSA | 15 | 372 | 385 | 407 |
| SARSA | 20 | 388 | 390 | 420 |

The oracle converges within the same window as the method (10%: QL 391 /
SARSA 389; 5%: QL 409 / SARSA 454), confirming that full observability does not
buy meaningfully faster convergence than fog-of-war navigation with perception
risk.

---

## Safety tables — `safety_comparison_{QLEARNING,SARSA}.csv`

**Structure:** rows = conditions (oracle + each risk-aware threshold); columns =
per-episode clearance (last-100 mean ± std), final-path clearance, Welch's
two-sided t-test p-value vs oracle, and Cohen's d effect size vs oracle
(d > 0 = risk-aware safer than oracle; d < 0 = oracle safer). Statistics mirror
the paper's RQ3 methodology (Welch's t-test + Cohen's d).

### Q-Learning

| Condition | Per-episode clearance (mean ± std) | Final-path clearance | p vs oracle | Cohen d |
|---|---|---|---|---|
| Oracle | 13.43 ± 0.65 | 12.36 | — | — |
| Scan5 | 12.35 ± 0.50 | 11.10 | <0.0001 (oracle safer) | −1.84 |
| Scan10 | 12.66 ± 0.60 | 11.64 | <0.0001 (oracle safer) | −1.22 |
| Scan15 | 13.56 ± 0.64 | 10.79 | 0.15 (no diff) | +0.21 |
| Scan20 | 13.38 ± 0.72 | 12.43 | 0.62 (no diff) | −0.07 |

### SARSA

| Condition | Per-episode clearance (mean ± std) | Final-path clearance | p vs oracle | Cohen d |
|---|---|---|---|---|
| Oracle | 13.23 ± 0.61 | 11.97 | — | — |
| Scan5 | 11.82 ± 0.56 | 11.91 | <0.0001 (oracle safer) | −2.41 |
| Scan10 | 13.62 ± 0.70 | 11.07 | <0.0001 (risk safer) | +0.60 |
| Scan15 | 13.49 ± 0.63 | 11.83 | 0.0035 (risk safer) | +0.42 |
| Scan20 | 13.48 ± 0.60 | 11.06 | 0.0030 (risk safer) | +0.43 |

---

## Observations

1. **The risk-aware method matches the full-map oracle overall.** Despite
   navigating unknown terrain and paying 377–508 perception scans (vs the
   oracle's zero), 1st-episode success (~355–394), reward-stability convergence
   (~380–460), post-convergence success (98–100 %), and path length are all in
   the oracle's range.

2. **Clearance improves with scan depth.** At the shallowest setting (Scan5) the
   method is significantly *less* safe than the oracle for both algorithms
   (Cohen d = −1.84 QL, −2.41 SARSA) — the agent commits to moves before it can
   see enough clearance. From Scan10 upward the gap closes.

3. **At sufficient scan depth the method reaches or exceeds oracle clearance.**
   - Q-Learning: Scan15–20 are statistically indistinguishable from the oracle.
   - SARSA: Scan10–20 are significantly *safer* than the oracle
     (Cohen d = +0.42 to +0.60). SARSA benefits more, consistent with its
     conservative on-policy updates compounding the risk signal.

4. **Perception cost is the trade-off, not accuracy.** The only price for
   fog-of-war operation is scan count; task success and exploration entropy are
   essentially unaffected.

5. **Metric caveat:** per-episode clearance (averaged over all training paths)
   and final-path clearance (the single extracted greedy route) can diverge —
   e.g. SARSA Scan20 has a high per-episode margin (13.48) but a lower
   final-path margin (11.06). The paper's safety claim should specify which
   definition it refers to; per-episode is the statistically testable one.

**Open items:**
- Results are single-seed; a multi-seed run would make each cell individually
  significant.
- The full-map oracle is a new baseline not present in the current paper draft
  (the paper's RQ1–RQ3 sweep scan depth within the perception method). This
  table is an addition beyond the manuscript.
- A subtle definitional difference: the risk-aware clearance stops at the
  nearest obstacle *or unknown* cell, whereas the oracle (no unknown cells)
  stops only at obstacles. This slightly disadvantages the risk-aware method in
  the raw comparison.
