"""
rl_agent.py
-----------
Tabular RL agent implementing:
  - ε-greedy action selection
  - SARSA (on-policy, conservative)
  - Q-Learning (off-policy, aggressive)

The Q-table is a defaultdict keyed by (state, action_idx) → float.
It persists across all replanning events so the agent accumulates knowledge
as the map grows frame by frame.
"""

import random
import math
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional


class RLAgent:
    """
    Tabular RL agent with ε-greedy exploration.

    Parameters
    ----------
    alpha   : float   Learning rate α (how fast new info overwrites old).
    gamma   : float   Discount factor γ (how much future rewards matter).
    n_actions : int   Number of discrete actions (8 compass directions).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        n_actions: int = 8,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.n_actions = n_actions

        # Q-table: (state_tuple, action_int) → float, default 0.0
        self.Q: defaultdict = defaultdict(float)

        # Tracking statistics
        self.total_updates = 0
        self.episode_rewards: List[float] = []
        self.td_errors: List[float] = []          # |TD error| per update
        self.visit_counts: defaultdict = defaultdict(int) # (state) -> count

        # --- CSV export extras (isolated, do not affect any training logic) ---
        self.sa_visit_counts: defaultdict = defaultdict(int)    # (state, action) -> int
        self.sa_tensions: dict = {}                              # (state, action) -> last tension value

    # ─────────────────────────────────────────────────────────────────
    def select_action(
        self,
        state: tuple,
        epsilon: float,
        valid_actions: List[int],
    ) -> int:
        """
        ε-greedy policy:
          - With probability epsilon → random valid action (explore)
          - Otherwise              → greedy best action (exploit)
        """
        if random.random() < epsilon:
            return random.choice(valid_actions)
        return self.greedy_action(state, valid_actions)

    # ─────────────────────────────────────────────────────────────────
    def greedy_action(self, state: tuple, valid_actions: List[int]) -> int:
        """Return the action with the highest Q-value (pure exploitation)."""
        best_a = valid_actions[0]
        best_q = self.Q[(state, best_a)]
        for a in valid_actions[1:]:
            q = self.Q[(state, a)]
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    # ─────────────────────────────────────────────────────────────────
    def update_sarsa(
        self,
        s: tuple,
        a: int,
        r: float,
        s_prime: tuple,
        a_prime: int,
        done: bool,
    ):
        """
        SARSA on-policy update (conservative):
            Q(s,a) ← Q(s,a) + α [ r + γ·Q(s',a') − Q(s,a) ]

        Note: a_prime is the ACTUAL next action chosen by ε-greedy (not greedy max).
        This makes SARSA sensitive to the exploration policy and thus more cautious
        about risky actions that were accidentally discovered during exploration.
        """
        td_target = r if done else r + self.gamma * self.Q[(s_prime, a_prime)]
        td_error  = td_target - self.Q[(s, a)]
        self.Q[(s, a)] += self.alpha * td_error
        self.total_updates += 1
        self.td_errors.append(abs(td_error))
        self.visit_counts[s] += 1

    # ─────────────────────────────────────────────────────────────────
    def update_qlearning(
        self,
        s: tuple,
        a: int,
        r: float,
        s_prime: tuple,
        valid_actions_prime: List[int],
        done: bool,
    ):
        """
        Q-Learning off-policy update (aggressive):
            Q(s,a) ← Q(s,a) + α [ r + γ·max_a Q(s',a) − Q(s,a) ]

        Note: uses the greedy max over ALL valid actions at s', regardless of
        what the ε-greedy policy would actually choose. This leads to faster
        convergence but more risk-seeking behaviour when penalties are infrequent.
        """
        if done:
            td_target = r
        else:
            max_q_next = max(self.Q[(s_prime, a)] for a in valid_actions_prime)
            td_target = r + self.gamma * max_q_next
        td_error = td_target - self.Q[(s, a)]
        self.Q[(s, a)] += self.alpha * td_error
        self.total_updates += 1
        self.td_errors.append(abs(td_error))
        self.visit_counts[s] += 1

    # ─────────────────────────────────────────────────────────────────
    def q_table_size(self) -> int:
        """Number of (state, action) entries learned so far."""
        return len(self.Q)

    # ─────────────────────────────────────────────────────────────────
    def print_summary(self, name: str = "RL Agent"):
        """Print a short summary of the agent's learned state."""
        print()
        print(f"  {'='*66}")
        print(f"  {name} Summary")
        print(f"  {'='*66}")
        print(f"  Q-table entries   : {self.q_table_size()}")
        print(f"  Total Q updates   : {self.total_updates}")
        if self.episode_rewards:
            recent = self.episode_rewards[-min(20, len(self.episode_rewards)):]
            print(f"  Mean reward (last 20 eps) : {sum(recent)/len(recent):.2f}")
        print(f"  {'='*66}")
        print()

    # ─────────────────────────────────────────────────────────────────
    def record_sa_visit(self, s: tuple, a: int):
        """Record a visit to (state, action) for CSV export only."""
        self.sa_visit_counts[(s, a)] += 1

    def record_sa_tension(self, s: tuple, a: int, tension: float):
        """Store the most recent tension value for (state, action) for CSV export only."""
        self.sa_tensions[(s, a)] = tension

    def get_sa_tension(self, s: tuple, a: int) -> float:
        """Return the last recorded tension for (state, action).
        Never scanned -> 0.0. Scanned but tension was exactly 0 -> 0.000001."""
        key = (s, a)
        if key not in self.sa_tensions:
            return 0.0
        val = float(self.sa_tensions[key])
        return val if val != 0.0 else 0.000001

    # ─────────────────────────────────────────────────────────────────
    def compute_exploration_entropy(self) -> float:
        """
        Computes Shannon entropy of the visitation distribution.
        Higher entropy = more uniform exploration.
        """
        if not self.visit_counts:
            return 0.0
        
        counts = list(self.visit_counts.values())
        total = sum(counts)
        probs = [c / total for c in counts]
        
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return float(entropy)
