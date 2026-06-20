# Nonparametric Hypothesis Tests Results

## Test Overview
- **Test Type**: Mann-Whitney U Test (non-parametric alternative to t-test)
- **Null Hypothesis (H₀)**: The two distributions are the same
- **Alternative Hypothesis (H₁)**: The two distributions are different
- **Significance Level**: α = 0.05

---

## Q-Learning Algorithm

### 1. Scan_5 vs Scan_15

| Metric | Value |
|--------|-------|
| U-statistic | 84691.50 |
| p-value | 0.000000 |
| Result | **SIGNIFICANT** ✓ |
| Mean Scan_5 | 11.0811 ± 2.2059 |
| Mean Scan_15 | 12.5830 ± 2.8885 |

**Interpretation**: There is a statistically significant difference between Scan_5 and Scan_15 safety margins in Q-Learning. Increasing the scan depth from 5 to 15 significantly improves safety.

### 2. Scan_15 vs Scan_25

| Metric | Value |
|--------|-------|
| U-statistic | 125634.00 |
| p-value | 0.889669 |
| Result | **NOT SIGNIFICANT** ✗ |
| Mean Scan_15 | 12.5830 ± 2.8885 |
| Mean Scan_25 | 12.5496 ± 2.9286 |

**Interpretation**: There is NO statistically significant difference between Scan_15 and Scan_25. Further increasing the scan depth from 15 to 25 does not provide additional safety benefit (diminishing returns).

---

## SARSA Algorithm

### 1. Scan_5 vs Scan_15

| Metric | Value |
|--------|-------|
| U-statistic | 77704.50 |
| p-value | 0.000000 |
| Result | **SIGNIFICANT** ✓ |
| Mean Scan_5 | 10.7570 ± 2.6770 |
| Mean Scan_15 | 12.3944 ± 2.7406 |

**Interpretation**: There is a statistically significant difference between Scan_5 and Scan_15 safety margins in SARSA. Similar to Q-Learning, increasing the scan depth from 5 to 15 significantly improves safety.

### 2. Scan_15 vs Scan_25

| Metric | Value |
|--------|-------|
| U-statistic | 124189.00 |
| p-value | 0.859129 |
| Result | **NOT SIGNIFICANT** ✗ |
| Mean Scan_15 | 12.3944 ± 2.7406 |
| Mean Scan_25 | 12.4860 ± 2.8210 |

**Interpretation**: There is NO statistically significant difference between Scan_15 and Scan_25. Again, diminishing returns are observed when increasing from 15 to 25.

---

## Summary Table

| Comparison | Q-Learning | SARSA |
|------------|-----------|-------|
| Scan_5 vs Scan_15 | p = 0.000000 ✓ **SIGNIFICANT** | p = 0.000000 ✓ **SIGNIFICANT** |
| Scan_15 vs Scan_25 | p = 0.889669 ✗ Not significant | p = 0.859129 ✗ Not significant |

---

## Key Findings

1. **Both algorithms show identical patterns**:
   - ✓ **Significant improvement** from Scan_5 to Scan_15
   - ✗ **No significant improvement** from Scan_15 to Scan_25

2. **Practical Implications**:
   - Scan depth of 15 is sufficient for safety improvements
   - Increasing to 25 does not provide additional statistical benefit
   - Recommendation: Use Scan_15 as an optimal trade-off between performance and safety

3. **Algorithm Consistency**:
   - Both Q-Learning and SARSA demonstrate the same safety characteristics
   - The choice between algorithms doesn't affect the scan depth optimization conclusion
