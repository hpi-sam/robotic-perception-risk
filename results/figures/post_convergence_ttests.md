# Post-convergence Welch t-tests on Avg_Safety_Margin

Reward tolerance: **10%**, safety tolerance: **15%**. Two-sided Welch t-test on the obstacle-clearance distance (per-episode `Avg_Safety_Margin`) between consecutive scan depths, using each scan's own convergence episode as the start of its post-convergence slice.

| Algorithm | Convergence slice | Pair | n₁ | n₂ | mean₁ | mean₂ | Δμ | t | p | Cohen's d | Sig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Q-Learning | Reward convergence (10%) | 5 -> 10 | 120 | 89 | 12.287 | 12.745 | 0.458 | 6.322 | 0.0000 | 0.873 | *** |
| Q-Learning | Reward convergence (10%) | 10 -> 15 | 89 | 74 | 12.745 | 13.626 | 0.881 | 10.048 | 0.0000 | 1.609 | *** |
| Q-Learning | Reward convergence (10%) | 15 -> 20 | 74 | 50 | 13.626 | 13.507 | -0.118 | -0.939 | 0.3500 | -0.179 | ns |
| Q-Learning | Safety convergence (15%) | 5 -> 10 | 139 | 89 | 12.227 | 12.745 | 0.518 | 7.116 | 0.0000 | 0.931 | *** |
| Q-Learning | Safety convergence (15%) | 10 -> 15 | 89 | 149 | 12.745 | 13.395 | 0.650 | 8.437 | 0.0000 | 1.045 | *** |
| Q-Learning | Safety convergence (15%) | 15 -> 20 | 149 | 50 | 13.395 | 13.507 | 0.112 | 0.945 | 0.3476 | 0.161 | ns |
| SARSA | Reward convergence (10%) | 5 -> 10 | 117 | 121 | 11.768 | 13.574 | 1.806 | 21.745 | 0.0000 | 2.810 | *** |
| SARSA | Reward convergence (10%) | 10 -> 15 | 121 | 116 | 13.574 | 13.439 | -0.135 | -1.553 | 0.1218 | -0.201 | ns |
| SARSA | Reward convergence (10%) | 15 -> 20 | 116 | 111 | 13.439 | 13.515 | 0.076 | 0.929 | 0.3539 | 0.123 | ns |
| SARSA | Safety convergence (15%) | 5 -> 10 | 122 | 60 | 11.741 | 13.621 | 1.880 | 18.163 | 0.0000 | 3.024 | *** |
| SARSA | Safety convergence (15%) | 10 -> 15 | 60 | 136 | 13.621 | 13.355 | -0.266 | -2.517 | 0.0133 | -0.395 | * |
| SARSA | Safety convergence (15%) | 15 -> 20 | 136 | 123 | 13.355 | 13.499 | 0.143 | 1.813 | 0.0710 | 0.225 | ns |

**Legend.** *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05. Bonferroni-adjusted α for 12 tests = 0.05/12 ≈ 0.00417.
