# Nonparametric Hypothesis Tests — BASELINE (No Risk Perception)

## Test Overview
- **Test Type**: Mann-Whitney U Test (non-parametric alternative to t-test)
- **Metric**: Avg_Safety_Margin per episode
- **Null Hypothesis (H0)**: The two distributions are the same
- **Alternative Hypothesis (H1)**: The two distributions are different
- **Significance Level**: alpha = 0.05

---

## QLEARNING Algorithm

### 1. Scan_5 vs Scan_10

| Metric | Value |
|--------|-------|
| U-statistic | 77582.50 |
| p-value | 0.000000 |
| Result | **SIGNIFICANT** |
| Mean Scan_5 | 10.3603 ± 2.8068 |
| Mean Scan_10 | 12.0951 ± 2.7673 |

### 2. Scan_10 vs Scan_15

| Metric | Value |
|--------|-------|
| U-statistic | 122533.00 |
| p-value | 0.589119 |
| Result | **NOT SIGNIFICANT** |
| Mean Scan_10 | 12.0951 ± 2.7673 |
| Mean Scan_15 | 12.0623 ± 2.8221 |

### 3. Scan_15 vs Scan_20

| Metric | Value |
|--------|-------|
| U-statistic | 122967.50 |
| p-value | 0.656344 |
| Result | **NOT SIGNIFICANT** |
| Mean Scan_15 | 12.0623 ± 2.8221 |
| Mean Scan_20 | 12.1269 ± 2.8723 |

### 4. Scan_20 vs Scan_25

| Metric | Value |
|--------|-------|
| U-statistic | 126778.00 |
| p-value | 0.697101 |
| Result | **NOT SIGNIFICANT** |
| Mean Scan_20 | 12.1269 ± 2.8723 |
| Mean Scan_25 | 12.1758 ± 2.8260 |

---

## SARSA Algorithm

### 1. Scan_5 vs Scan_10

| Metric | Value |
|--------|-------|
| U-statistic | 103443.50 |
| p-value | 0.000002 |
| Result | **SIGNIFICANT** |
| Mean Scan_5 | 10.8622 ± 2.3883 |
| Mean Scan_10 | 11.6500 ± 2.7481 |

### 2. Scan_10 vs Scan_15

| Metric | Value |
|--------|-------|
| U-statistic | 104941.00 |
| p-value | 0.000011 |
| Result | **SIGNIFICANT** |
| Mean Scan_10 | 11.6500 ± 2.7481 |
| Mean Scan_15 | 12.2783 ± 2.8068 |

### 3. Scan_15 vs Scan_20

| Metric | Value |
|--------|-------|
| U-statistic | 121993.00 |
| p-value | 0.510306 |
| Result | **NOT SIGNIFICANT** |
| Mean Scan_15 | 12.2783 ± 2.8068 |
| Mean Scan_20 | 12.2872 ± 2.6556 |

### 4. Scan_20 vs Scan_25

| Metric | Value |
|--------|-------|
| U-statistic | 135267.50 |
| p-value | 0.024559 |
| Result | **SIGNIFICANT** |
| Mean Scan_20 | 12.2872 ± 2.6556 |
| Mean Scan_25 | 12.1440 ± 2.7425 |

---

## Summary Table

| Algorithm | Comparison | p-value | Result |
|-----------|------------|---------|--------|
| QLEARNING | Scan_5 vs Scan_10 | 0.000000 | SIGNIFICANT |
| QLEARNING | Scan_10 vs Scan_15 | 0.589119 | Not significant |
| QLEARNING | Scan_15 vs Scan_20 | 0.656344 | Not significant |
| QLEARNING | Scan_20 vs Scan_25 | 0.697101 | Not significant |
| SARSA | Scan_5 vs Scan_10 | 0.000002 | SIGNIFICANT |
| SARSA | Scan_10 vs Scan_15 | 0.000011 | SIGNIFICANT |
| SARSA | Scan_15 vs Scan_20 | 0.510306 | Not significant |
| SARSA | Scan_20 vs Scan_25 | 0.024559 | SIGNIFICANT |
