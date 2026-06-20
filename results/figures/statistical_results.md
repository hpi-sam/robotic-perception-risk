# Statistical Results: Active Perception RL Comparison

This document contains the consolidated statistical results for the BSc Thesis paper, based on the **Threshold 60** experimental data (Scan Depths: 5, 10, 15, 20).

---

## 1. Correlation Analysis: Perception vs. Performance
To quantify the relationship between internal uncertainty (**Perception Entropy $H$**) and mission performance (**Episode Reward $R$**).

*   **Test**: Spearman's Rank Correlation Coefficient ($\rho$)
*   **Correction**: Bonferroni correction applied (8 tests)

| Agent | Scan Depth | Spearman ($\rho$) | Original $p$ | Adjusted $p$ (Bonferroni) |
| :--- | :--- | :--- | :--- | :--- |
| **Q-Learning** | 5 | 0.6074 | $5.56 \times 10^{-52}$ | $4.45 \times 10^{-51}$ |
| | 10 | 0.5134 | $1.73 \times 10^{-35}$ | $1.38 \times 10^{-34}$ |
| | 15 | 0.4907 | $1.99 \times 10^{-32}$ | $1.59 \times 10^{-31}$ |
| | **20** | **0.6322** | **$3.62 \times 10^{-57}$** | **$2.89 \times 10^{-56}$** |
| **SARSA** | 5 | 0.6139 | $7.85 \times 10^{-53}$ | $6.28 \times 10^{-52}$ |
| | 10 | 0.7077 | $3.59 \times 10^{-76}$ | $2.87 \times 10^{-75}$ |
| | 15 | 0.7042 | $5.52 \times 10^{-75}$ | $4.42 \times 10^{-74}$ |
| | **20** | **0.7539** | **$6.45 \times 10^{-93}$** | **$5.16 \times 10^{-92}$** |

---

## 2. Saturation Point Analysis: Safety Margins
To evaluate significant differences in safety margins across scan depths and identify the point of diminishing returns.

*   **Test**: Mann-Whitney U Test (Nonparametric)
*   **Comparison**: 500 episodes per configuration

### Agent: Q-Learning
| Comparison | U-Statistic | p-value | Mean Diff ($\Delta\mu$) | Result |
| :--- | :--- | :--- | :--- | :--- |
| Scan 5 vs Scan 10 | 120,238.00 | 0.297 | +0.17 | Not Significant |
| **Scan 10 vs Scan 15** | **99,461.50** | **$2.24 \times 10^{-8}$** | **+0.85** | **Significant** |
| Scan 15 vs Scan 20 | 121,912.00 | 0.499 | +0.23 | Not Significant |

### Agent: SARSA
| Comparison | U-Statistic | p-value | Mean Diff ($\Delta\mu$) | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Scan 5 vs Scan 10** | **86,407.50** | **$2.89 \times 10^{-17}$** | **+1.33** | **Significant** |
| Scan 10 vs Scan 15 | 128,815.50 | 0.403 | -0.06 | Not Significant |
| Scan 15 vs Scan 20 | 124,288.50 | 0.876 | +0.04 | Not Significant |

---

## Summary Findings
1.  **Correlation**: Highly significant positive correlation between perception entropy and reward across all settings ($p_{adj} < 0.001$).
2.  **Saturation**: Q-Learning safety performance saturates at **Scan 15**, while SARSA reaches its peak safety performance earlier at **Scan 10**.
