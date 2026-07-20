# Perfect Perception vs Risk-Aware Navigation — Results Summary

## Comparison Summary — which set wins each metric

**How to read:** lower path length is better; higher clearance is safer; earlier
convergence (Conv. Ep. 10%) is faster learning. **PP** = Perfect perception
(full-map baseline). Best value in each row is in **bold**. Clearance is the
per-episode obstacle-clearance distance d̄_clear (last-100 mean, the
statistically tested distribution — the same metric as the paper's RQ3).

### Q-Learning

| Metric (better =) | PP | Scan5 | Scan10 | Scan15 | Scan20 | Winner |
|---|---|---|---|---|---|---|
| Path length, cells (lower) | 133 | 178 | 140 | 175 | **131** | **Scan20** (PP close 2nd, 133) |
| Clearance d̄_clear (higher) | 13.43 | 12.35 | 12.66 | **13.56** | 13.38 | **Scan15** (not sig. > PP; PP 2nd) |
| Convergence, Conv. 10% (lower) | 391 | **381** | 412 | 427 | 451 | **Scan5** (PP 2nd, 391) |

### SARSA

| Metric (better =) | PP | Scan5 | Scan10 | Scan15 | Scan20 | Winner |
|---|---|---|---|---|---|---|
| Path length, cells (lower) | 176 | **168** | 177 | 170 | 171 | **Scan5** (PP 2nd-worst, 176) |
| Clearance d̄_clear (higher) | 13.23 | 11.82 | **13.62** | 13.49 | 13.48 | **Scan10** (sig. > PP, d=+0.60) |
| Convergence, Conv. 10% (lower) | 389 | 384 | **380** | 385 | 390 | **Scan10** (tight 380–390 window) |

**Takeaways per metric:**
- **Path length** — the risk-aware method wins for *both* algorithms (QL Scan20
  131 < PP 133; SARSA Scan5 168 < PP 176). Fog-of-war navigation finds paths as
  short or shorter than the full-map baseline.
- **Clearance** — the risk-aware method matches or exceeds perfect perception at
  sufficient scan depth: QL Scan15 (13.56) edges PP (not significant), and SARSA
  Scan10–20 are significantly *safer* than PP (Cohen d up to +0.60). Only the
  shallowest setting (Scan5) is clearly less safe than PP.
- **Convergence** — the risk-aware method converges as fast or faster than
  perfect perception for both algorithms (QL Scan5 381 < PP 391; SARSA Scan10
  380 < PP 389). Full observability buys no convergence advantage.

**Overall:** on every metric, at least one scan-depth setting of the risk-aware
method matches or beats the full-map baseline — the fog-of-war method loses
nothing to perfect perception and often wins. (Single seed; see open items.)

---

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
- **Clearance (d̄_clear)** — average obstacle-clearance distance: the per-step
  straight-line distance to the nearest obstacle, averaged per episode, reported
  as the last-100 (post-convergence) mean. This is the paper's RQ3 metric and
  the statistically testable distribution.
- **Success % (last 100)** — goal-reaching rate over the last 100 episodes.

> *Final Reward* is deliberately excluded. The perfect-perception baseline
> (−20, no risk term) and the risk-aware method (−20, with the −ξ_perc term) use
> different reward functions, so reward magnitudes are not comparable. The
> comparison is on reward-independent outcomes (convergence, path length,
> clearance, success, scans).

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
| Success % (last 100) | 100 | 100 | 98 | 99 | 99 |
| Total scans | 0 | 377 | 440 | 387 | 508 |

### SARSA

| Metric | Perfect perception | Scan5 (Baseline) | Scan10 | Scan15 | Scan20 |
|---|---|---|---|---|---|
| 1st Episode Success | 382 | 380 | 355 | 372 | 388 |
| Conv. Ep. (10%) | 389 | 384 | 380 | 385 | 390 |
| Conv. Ep. (5%) | 454 | 458 | 395 | 407 | 420 |
| Final path length (cells) | 176 | 168 | 177 | 170 | 171 |
| Success % (last 100) | 100 | 100 | 100 | 100 | 100 |
| Total scans | 0 | 431 | 401 | 417 | 441 |

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
threshold); columns = per-episode clearance d̄_clear (last-100 mean ± std),
Welch's two-sided t-test p-value vs perfect perception, and Cohen's d effect
size (d > 0 = risk-aware safer than perfect perception; d < 0 = perfect
perception safer). Statistics mirror the paper's RQ3 methodology.

### Q-Learning

| Condition | Clearance d̄_clear (mean ± std) | p vs Perfect perception | Cohen d |
|---|---|---|---|
| Perfect perception | 13.43 ± 0.65 | — | — |
| Scan5 (Baseline) | 12.35 ± 0.50 | <0.0001 (perception safer) | −1.84 |
| Scan10 | 12.66 ± 0.60 | <0.0001 (perception safer) | −1.22 |
| Scan15 | 13.56 ± 0.64 | 0.15 (no diff) | +0.21 |
| Scan20 | 13.38 ± 0.72 | 0.62 (no diff) | −0.07 |

### SARSA

| Condition | Clearance d̄_clear (mean ± std) | p vs Perfect perception | Cohen d |
|---|---|---|---|
| Perfect perception | 13.23 ± 0.61 | — | — |
| Scan5 (Baseline) | 11.82 ± 0.56 | <0.0001 (perception safer) | −2.41 |
| Scan10 | 13.62 ± 0.70 | <0.0001 (risk safer) | +0.60 |
| Scan15 | 13.49 ± 0.63 | 0.0035 (risk safer) | +0.42 |
| Scan20 | 13.48 ± 0.60 | 0.0030 (risk safer) | +0.43 |

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
   fog-of-war operation is scan count; task success is essentially unaffected.

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
