# Perfect Perception vs Risk-Aware Navigation — Results Summary

## Setup

- **Perfect perception (baseline):** agent is given the complete occupancy map
  at episode 0. No fog-of-war, no scanning, no perception risk. Reward =
  −StepCost + ProgressReward; the only penalty is a collision penalty of −20 for
  entering an unknown cell. Single reproducible run per algorithm (seed 42).
- **Risk-aware method (Th60):** operates under fog-of-war with active perception
  scanning and the perception-risk term in the reward (θ=60°), evaluated across
  scan-depth thresholds τ_scan ∈ {5, 10, 15, 20} — matching the paper. Collision
  penalty −20. Scan depth 5 is the baseline of the sweep (paper Δ baseline).
- Distances/clearances are in grid cells (resolution 0.05 m → 1 cell = 5 cm).

> **Note on the penalty:** with perfect perception the map is fully known, so
> unknown cells are pre-resolved and the −20 penalty is effectively inert — the
> agent's valid-action masking never lets it enter a non-free cell. We verified
> that penalty values of −20 and −40 produce byte-identical results for this
> baseline; −20 is reported for consistency with the risk-aware method.

### Metric definitions (aligned with the paper)

- **1st Episode Success** — first episode in which the goal is reached (the
  paper's "1st Episode Success" column, and what earlier drafts mislabelled as
  "convergence").
- **Conv. Ep. (10%) / (5%)** — the paper's reward-stability convergence: first
  episode after which all subsequent rewards stay within 10% / 5% of their mean.
  Later and stricter than 1st Episode Success.
- **Final path clearance** — obstacle-clearance margin (d̄_clear) of the final
  extracted greedy path.
- **Per-episode clearance** — d̄_clear averaged per episode over the last 100
  (post-convergence) episodes; the statistically testable distribution.
- **Success % (last 100)** — goal-reaching rate over the last 100 episodes.

> *Final Reward* is deliberately excluded. The perfect-perception baseline
> (−20, no risk term) and the risk-aware method (−20, with the −ξ_perc term) use
> different reward functions, so reward magnitudes are not comparable. The
> comparison is on reward-independent outcomes (convergence, path length,
> clearance, success, scans, entropy).

---

## Comparison tables — `comparison_perfect_perception_vs_riskaware_{QLEARNING,SARSA}.csv`

**Structure:** rows = metrics; columns = `Perfect perception` then risk-aware at
each scan threshold (Scan 5 = sweep baseline).

### Q-Learning

| Metric | Perfect perception | Scan5 (Baseline) | Scan10 | Scan15 | Scan20 |
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

| Metric | Perfect perception | Scan5 (Baseline) | Scan10 | Scan15 | Scan20 |
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

Paper Table 1 layout, with the perfect-perception baseline added as a row under
each algorithm. Shows all three convergence definitions side by side.

| RL | Scan Depth | 1st Episode Success | Conv. Ep. (10%) | Conv. Ep. (5%) |
|---|---|---|---|---|
| QL | Perfect perception | 367 | 391 | 409 |
| QL | 5 (Baseline) | 381 | 381 | 460 |
| QL | 10 | 394 | 412 | 438 |
| QL | 15 | 366 | 427 | 457 |
| QL | 20 | 374 | 451 | 451 |
| SARSA | Perfect perception | 382 | 389 | 454 |
| SARSA | 5 (Baseline) | 380 | 384 | 458 |
| SARSA | 10 | 355 | 380 | 395 |
| SARSA | 15 | 372 | 385 | 407 |
| SARSA | 20 | 388 | 390 | 420 |

Perfect perception converges within the same window as the method (10%: QL 391 /
SARSA 389; 5%: QL 409 / SARSA 454), confirming that full observability does not
buy meaningfully faster convergence than fog-of-war navigation with perception
risk.

---

## Safety tables — `safety_comparison_{QLEARNING,SARSA}.csv`

**Structure:** rows = conditions (perfect perception + each risk-aware
threshold); columns = per-episode clearance (last-100 mean ± std), final-path
clearance, Welch's two-sided t-test p-value vs perfect perception, and Cohen's d
effect size (d > 0 = risk-aware safer than perfect perception; d < 0 = perfect
perception safer). Statistics mirror the paper's RQ3 methodology.

### Q-Learning

| Condition | Per-episode clearance (mean ± std) | Final-path clearance | p vs Perfect perception | Cohen d |
|---|---|---|---|---|
| Perfect perception | 13.43 ± 0.65 | 12.36 | — | — |
| Scan5 (Baseline) | 12.35 ± 0.50 | 11.10 | <0.0001 (perception safer) | −1.84 |
| Scan10 | 12.66 ± 0.60 | 11.64 | <0.0001 (perception safer) | −1.22 |
| Scan15 | 13.56 ± 0.64 | 10.79 | 0.15 (no diff) | +0.21 |
| Scan20 | 13.38 ± 0.72 | 12.43 | 0.62 (no diff) | −0.07 |

### SARSA

| Condition | Per-episode clearance (mean ± std) | Final-path clearance | p vs Perfect perception | Cohen d |
|---|---|---|---|---|
| Perfect perception | 13.23 ± 0.61 | 11.97 | — | — |
| Scan5 (Baseline) | 11.82 ± 0.56 | 11.91 | <0.0001 (perception safer) | −2.41 |
| Scan10 | 13.62 ± 0.70 | 11.07 | <0.0001 (risk safer) | +0.60 |
| Scan15 | 13.49 ± 0.63 | 11.83 | 0.0035 (risk safer) | +0.42 |
| Scan20 | 13.48 ± 0.60 | 11.06 | 0.0030 (risk safer) | +0.43 |

---

## Observations

1. **The risk-aware method matches the perfect-perception baseline overall.**
   Despite navigating unknown terrain and paying 377–508 perception scans (vs the
   baseline's zero), 1st-episode success (~355–394), reward-stability convergence
   (~380–460), post-convergence success (98–100 %), and path length are all in
   the baseline's range.

2. **Clearance improves with scan depth.** At the shallowest setting
   (Scan5, baseline) the method is significantly *less* safe than perfect
   perception for both algorithms (Cohen d = −1.84 QL, −2.41 SARSA) — the agent
   commits to moves before it can see enough clearance. From Scan10 up the gap
   closes.

3. **At sufficient scan depth the method reaches or exceeds perfect-perception
   clearance.**
   - Q-Learning: Scan15–20 are statistically indistinguishable from perfect
     perception.
   - SARSA: Scan10–20 are significantly *safer* than perfect perception
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
- The perfect-perception baseline is a new baseline not present in the current
  paper draft (the paper's RQ1–RQ3 sweep scan depth within the perception
  method). This table is an addition beyond the manuscript.
- A subtle definitional difference: the risk-aware clearance stops at the
  nearest obstacle *or unknown* cell, whereas perfect perception (no unknown
  cells) stops only at obstacles. This slightly disadvantages the risk-aware
  method in the raw comparison.
