"""Training / evaluation loops for the PPO agent.

Mirrors `src/loops.py` (DQN): same per-episode bookkeeping, same periodic
greedy evaluation, same best-checkpoint selection, so the runner scripts and
plotting utilities are interchangeable between the two algorithms.
"""
import json
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_ppo(
    agent,
    env,
    n_episodes: int,
    log_dir: Optional[Path] = None,
    eval_fn: Optional[Callable] = None,
    freq_episodes_eval: int = 100,
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> List[float]:

    logging = log_dir is not None
    if logging:
        writer = SummaryWriter(log_dir)

    reward_per_episode = []
    operation_cost_per_episode = []
    avg_action_per_episode = []
    eval_episodes = []
    eval_rewards = []
    profit_per_episode = []
    safety_violation_per_episode = []

    best_eval_reward = -float("inf")
    best_eval_episode = None

    for i in tqdm(range(0, n_episodes)):

        state = env.reset()
        rewards = 0
        op_cost = 0
        ep_profit = 0.0
        ep_violation = 0.0
        actions_taken = []
        done = False
        while not done:

            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            o_co = env.render()
            agent.observe(state, action, reward, next_state, done)
            # PPO update fires automatically once the rollout buffer is full.
            update_metrics = agent.update()

            rewards += reward
            op_cost += o_co
            actions_taken.append(float(np.asarray(action).reshape(-1)[0]))
            ep_profit += float(getattr(env, 'last_profit', 0.0))
            ep_violation = max(ep_violation, float(getattr(env, 'last_safety_violation', 0.0)))

            state = next_state

        if logging:
            writer.add_scalar('train/rewards', rewards, i)
            writer.add_scalar('train/operation_cost', op_cost, i)
            writer.add_scalar('train/avg_action', np.mean(actions_taken), i)
            if agent.last_update_metrics:
                writer.add_scalar('train/policy_log_std', agent.last_update_metrics.get('policy_log_std', 0.0), i)
                writer.add_scalar('train/policy_loss', agent.last_update_metrics.get('policy_loss', 0.0), i)
                writer.add_scalar('train/value_loss', agent.last_update_metrics.get('value_loss', 0.0), i)

        reward_per_episode.append(rewards)
        operation_cost_per_episode.append(op_cost)
        avg_action_per_episode.append(float(np.mean(actions_taken)))
        profit_per_episode.append(ep_profit)
        safety_violation_per_episode.append(ep_violation)

        # Periodic greedy evaluation (deterministic mean action) to separate
        # policy quality from exploration noise.
        if eval_fn is not None and (i + 1) % freq_episodes_eval == 0:
            eval_reward = float(eval_fn(agent, env))
            eval_episodes.append(i + 1)
            eval_rewards.append(eval_reward)
            if logging:
                writer.add_scalar('eval/mean_reward', eval_reward, i)
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_eval_episode = i + 1
                if checkpoint_dir is not None:
                    ckpt_path = Path(checkpoint_dir)
                    ckpt_path.mkdir(parents=True, exist_ok=True)
                    agent.save_to_disk(ckpt_path)
                    with open(ckpt_path / 'best_eval.json', 'w') as f:
                        json.dump(
                            {'best_eval_reward': best_eval_reward,
                             'best_eval_episode': best_eval_episode},
                            f,
                        )
                    print(f"Best greedy eval improved: episode {best_eval_episode}, "
                          f"reward={best_eval_reward:.4f} -> checkpoint saved to {ckpt_path}")

    # Make training diagnostics available after training without changing the
    # return signature (same attributes the DQN loop exposes).
    agent.avg_action_per_episode = avg_action_per_episode

    # Make periodic greedy-eval results available after training.
    agent.eval_episodes = eval_episodes
    agent.eval_rewards = eval_rewards

    # Make per-episode profit/safety metrics available.
    agent.profit_per_episode = profit_per_episode
    agent.safety_violation_per_episode = safety_violation_per_episode

    # Make best-eval checkpoint info available after training.
    agent.best_eval_reward = best_eval_reward if eval_episodes else None
    agent.best_eval_episode = best_eval_episode

    return reward_per_episode, operation_cost_per_episode
