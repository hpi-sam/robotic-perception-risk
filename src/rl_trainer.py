"""
rl_trainer.py
-------------
Online training loop for SARSA and Q-Learning agents.

Called at every replanning event (Mode 2: every frame, Mode 3: every N frames)
with the CURRENT partial occupancy map. The Q-table PERSISTS across calls,
so the agent accumulates knowledge as the map grows frame by frame.

Usage in main.py:
    from rl_trainer import train_online

    sarsa_agent  = RLAgent(alpha=RL_ALPHA, gamma=RL_GAMMA)
    ql_agent     = RLAgent(alpha=RL_ALPHA, gamma=RL_GAMMA)
    sarsa_eps    = RL_EPSILON_START
    ql_eps       = RL_EPSILON_START

    # At each replan event:
    path_sarsa, sarsa_eps = train_online(
        sarsa_agent, env_sarsa, start, goal, 
        n_episodes=RL_EPISODES_PER_REPLAN,
        mode='sarsa', epsilon=sarsa_eps,
        epsilon_min=RL_EPSILON_END,
        epsilon_decay=rl_epsilon_decay,
    )
"""

import math
import numpy as np
from typing import Tuple, List, Optional

from rl_environment import GridEnv, ACTIONS, N_ACTIONS
from rl_agent import RLAgent


def train_online(
    agent: RLAgent,
    env: GridEnv,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    n_episodes: int,
    mode: str,                  # 'sarsa' or 'qlearning'
    epsilon: float,             # current exploration rate (persists across calls)
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    verbose: bool = False,
    start_dir: int = 0,
    # In-episode scanning parameters
    scan_callback = None,       # function(agent_pos, direction) -> updated env data
    scan_depth_threshold: int = 30,
    episode_offset: int = 0,    # Offset for global episode printing
    total_episodes: Optional[int] = None, # Total episodes for global printing
    reset_scan_history = None,  # optional callable — resets per-episode scan history
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], float, float, int, int, int, float, dict]:
    """
    Run `n_episodes` mini-episodes of RL training on the current partial map.

    Parameters
    ----------
    agent         : RLAgent — shared agent (Q-table persists across calls)
    env           : GridEnv — environment (updated map each call)
    start         : (row, col) source cell for this replan event
    goal          : (row, col) target cell
    n_episodes    : how many episodes to run this call
    mode          : 'sarsa' or 'qlearning'
    epsilon       : current eps (passed in, updated, returned)
    epsilon_min   : floor for eps decay
    epsilon_decay : multiplicative decay per episode
    scan_callback : optional function for mid-episode scanning
    scan_depth_threshold : d_dir threshold to trigger mid-episode scan

    Returns
    -------
    path    : the greedy path extracted after training
    epsilon : the updated epsilon value
    """
    env.goal = goal

    first_goal_ep = -1
    first_goal_steps = -1
    goal_reach_count = 0
    
    max_reward = -float('inf')
    best_path_found = []

    history = {
        'reward': [],
        'length': [],
        'td_error': [],
        'success': [],
        'safety': [],
        'entropy': [],
        'tension': []
    }

    for ep in range(n_episodes):
        # Reset per-episode scan history so every episode can trigger fresh scans
        if reset_scan_history is not None:
            reset_scan_history()

        # Track tensions triggered in this episode
        ep_tensions = []

        def _internal_scan_callback(pos, direction):
            if scan_callback:
                t_val = scan_callback(pos, direction)
                if t_val is not None:
                    ep_tensions.append(t_val)
                    # CSV-only: record tension for the (state, action) being evaluated
                    agent.record_sa_tension(state, action, t_val)

        if (ep + 1) % 50 == 0:
            print(f"      [Episode {ep+1}/{n_episodes}] ... running ...")
        state = env.reset(start, start_dir)
        episode_reward = 0.0
        done = False
        steps = 0
        current_episode_path = [start]
        
        # Track TD errors for this episode
        td_start_idx = len(agent.td_errors)

        if mode == 'sarsa':
            # SARSA: choose first action here (on-policy)
            valid = env.valid_actions()
            action = agent.select_action(state, epsilon, valid)

        while not done:
            valid = env.valid_actions()
            steps += 1

            if mode == 'sarsa':
                # --- In-Episode Scan Check (before stepping) ---
                r_c, c_c, _ = env._decode_pos(state)
                if env.dist is not None and env.dist.ndim == 3:
                    d_ahead = float(env.dist[r_c, c_c, action])
                    # Only scan if the path ahead leads to UNKNOWN (-1)
                    if env.stop_type is not None:
                        st = env.stop_type[r_c, c_c, action]
                        if st == -1 and d_ahead < scan_depth_threshold:
                            _internal_scan_callback((r_c, c_c), action)

                # Real-time progress print
                if steps % 50 == 0:
                    current_ep_global = ep + 1 + episode_offset
                    total_to_show = total_episodes if total_episodes else n_episodes
                    print(f"      [Episode {current_ep_global}/{total_to_show} | Step {steps}/{env.max_steps}] Exploring...", end='\r')

                # Action is pre-selected (from previous step or episode start)
                next_state, reward, done = env.step(action)
                episode_reward += reward
                
                r_c, c_c, _ = env._decode_pos(next_state)
                current_episode_path.append((r_c, c_c))

                if not done:
                    valid_next = env.valid_actions()
                    next_action = agent.select_action(next_state, epsilon, valid_next)
                else:
                    next_action = 0   # doesn't matter, done=True

                agent.update_sarsa(state, action, reward, next_state, next_action, done)
                agent.record_sa_visit(state, action)  # CSV-only tracking
                state = next_state
                action = next_action

            elif mode == 'qlearning':
                action = agent.select_action(state, epsilon, valid)
                
                # --- In-Episode Scan Check (before stepping) ---
                r_c, c_c, _ = env._decode_pos(state)
                if env.dist is not None and env.dist.ndim == 3:
                    d_ahead = float(env.dist[r_c, c_c, action])
                    if env.stop_type is not None:
                        st = env.stop_type[r_c, c_c, action]
                        if st == -1 and d_ahead < scan_depth_threshold:
                            _internal_scan_callback((r_c, c_c), action)
                
                # Real-time progress print
                if steps % 50 == 0:
                    current_ep_global = ep + 1 + episode_offset
                    total_to_show = total_episodes if total_episodes else n_episodes
                    print(f"      [Episode {current_ep_global}/{total_to_show} | Step {steps}/{env.max_steps}] Exploring...", end='\r')

                next_state, reward, done = env.step(action)
                episode_reward += reward
                
                r_c, c_c, _ = env._decode_pos(next_state)
                current_episode_path.append((r_c, c_c))

                valid_next = env.valid_actions()
                agent.update_qlearning(state, action, reward, next_state, valid_next, done)
                agent.record_sa_visit(state, action)  # CSV-only tracking
                state = next_state

            # Check if goal reached to record stats
            r_c, c_c, _ = env._decode_pos(state)
            if (r_c, c_c) == env.goal:
                if first_goal_ep == -1:
                    first_goal_ep = ep + 1
                    first_goal_steps = steps
                goal_reach_count += 1
        
        # --- End of Episode Metrics ---
        # 1. TD Error Avg for this episode
        td_end_idx = len(agent.td_errors)
        if td_end_idx > td_start_idx:
            avg_td = float(np.mean(agent.td_errors[td_start_idx:td_end_idx]))
        else:
            avg_td = 0.0
            
        # 2. Safety Margin (Avg distance to nearest obstacle along the path)
        ep_safety = 0.0
        if env.dist is not None and env.dist.ndim == 3:
            for (r_p, c_p) in current_episode_path:
                ep_safety += float(np.min(env.dist[r_p, c_p, :]))
        avg_safety = ep_safety / len(current_episode_path)
        
        # 3. Success Flag
        r_fin, c_fin, _ = env._decode_pos(state)
        success_flag = 1 if (r_fin, c_fin) == env.goal else 0

        # Append to history
        history['reward'].append(episode_reward)
        history['length'].append(steps)
        history['td_error'].append(avg_td)
        history['success'].append(success_flag)
        history['safety'].append(avg_safety)
        history['entropy'].append(agent.compute_exploration_entropy())
        history['tension'].append(np.mean(ep_tensions) if ep_tensions else 0.0)
        # Track the absolute best path by reward
        if episode_reward > max_reward:
            max_reward = episode_reward
            best_path_found = current_episode_path

        agent.episode_rewards.append(episode_reward)
        
        # Periodic status print
        if (ep + 1) % 10 == 0:
            current_ep_global = ep + 1 + episode_offset
            total_to_show = total_episodes if total_episodes else n_episodes
            print(f"      [Finished Episode {current_ep_global}/{total_to_show}]")

        # Decay epsilon after each episode (floor at epsilon_min)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Extract the greedy path (no exploration) after training
    greedy_path, greedy_reward = env.extract_greedy_path(agent, start, start_dir, max_steps=400)

    if verbose:
        print(f"    [{mode.upper()} | {n_episodes} eps] eps={epsilon:.3f} "
              f"-> path={len(greedy_path)} cells  Q-table={agent.q_table_size()}  Goal Hits={goal_reach_count}")

    return greedy_path, best_path_found, max_reward, epsilon, first_goal_ep, first_goal_steps, goal_reach_count, greedy_reward, history


def compute_epsilon_decay(
    epsilon_start: float,
    epsilon_end: float,
    total_replan_events: int,
    episodes_per_event: int,
) -> float:
    """
    Pre-compute the per-episode epsilon decay factor so that epsilon
    reaches epsilon_end by the end of the full run.

    total_replan_events  : estimated number of replanning events in the run
    episodes_per_event   : RL_EPISODES_PER_REPLAN
    """
    total_episodes = total_replan_events * episodes_per_event
    if total_episodes <= 0 or epsilon_start <= epsilon_end:
        return 1.0
    decay = (epsilon_end / epsilon_start) ** (1.0 / total_episodes)
    return float(decay)
