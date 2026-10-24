from typing import Tuple, List, Callable, Union, Optional
import random
import json
from pathlib import Path
from collections import deque
from pdb import set_trace as stop

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt


def train(
    agent,
    env,
    n_episodes: int,
    log_dir: Optional[Path] = None,
    max_steps: Optional[int] = float("inf"),
    n_episodes_evaluate_agent: Optional[int] = 100,
    freq_episodes_evaluate_agent: int = 200,
    eval_fn: Optional[Callable] = None,
    freq_episodes_eval: int = 100,
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> None:

    # Tensorborad log writer
    logging = False
    if log_dir is not None:
        writer = SummaryWriter(log_dir)
        logging = True

    reward_per_episode = []
    # r_unbalance_per_episode = []
    operation_cost_per_episode = []
    avg_q_per_episode = []
    avg_max_q_per_episode = []
    steps_per_episode = []
    eval_episodes = []
    eval_rewards = []
    profit_per_episode = []
    safety_violation_per_episode = []
    #global_step_counter = 0

    best_eval_reward = -float("inf")
    best_eval_episode = None

    for i in tqdm(range(0, n_episodes)):

        state = env.reset()
        rewards = 0
        # r_unbal = 0
        op_cost = 0
        ep_profit = 0.0
        ep_violation = 0.0
        #steps = 0
        q_means = []
        q_maxes = []
        done = False
        while not done:

            q_mean, q_max = agent.get_state_q_stats(state)
            q_means.append(q_mean)
            q_maxes.append(q_max)

            action = agent.act(state)

            # agents takes a step and the environment throws out a new state and
            # a reward
            next_state, reward, done, info = env.step(action)

            # r_ub, o_co = env.render()
            o_co = env.render()
            # agent observes transition and stores it for later use
            agent.observe(state, action, reward, next_state, done)
            #env.render()
            # learning happens here, through experience replay
            agent.replay()

            #global_step_counter += 1
            #steps += 1
            rewards += reward
            # r_unbal += r_ub
            op_cost += o_co
            ep_profit += float(getattr(env, 'last_profit', 0.0))
            ep_violation = max(ep_violation, float(getattr(env, 'last_safety_violation', 0.0)))

            state = next_state
        
        # log to Tensorboard
        if logging:
            writer.add_scalar('train/rewards', rewards, i)
            #writer.add_scalar('train/steps', steps, i)
            writer.add_scalar('train/epsilon', agent.epsilon, i)
            writer.add_scalar('train/replay_memory_size', len(agent.memory), i)
            writer.add_scalar('train/avg_q', np.mean(q_means), i)
            writer.add_scalar('train/avg_max_q', np.mean(q_maxes), i)

        reward_per_episode.append(rewards)
        # r_unbalance_per_episode.append(r_unbal)
        operation_cost_per_episode.append(op_cost)
        avg_q_per_episode.append(float(np.mean(q_means)))
        avg_max_q_per_episode.append(float(np.mean(q_maxes)))
        profit_per_episode.append(ep_profit)
        safety_violation_per_episode.append(ep_violation)

        # Periodic greedy evaluation (epsilon forced to 0) to separate policy
        # quality from exploration noise.
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

        #steps_per_episode.append(steps)

        # if (i > 0) and (i % freq_episodes_evaluate_agent) == 0:
        # if (i + 1) % freq_episodes_evaluate_agent == 0:
        #     # evaluate agent
        #     eval_rewards, eval_steps = evaluate(agent, env,
        #                                         n_episodes=n_episodes_evaluate_agent,
        #                                         epsilon=0.01)

        #     # from src.utils import get_success_rate_from_n_steps
        #     # success_rate = get_success_rate_from_n_steps(env, eval_steps)
        #     print(f'Reward mean: {np.mean(eval_rewards):.5f}, std: {np.std(eval_rewards):.5f}')
        #     print(f'Num steps mean: {np.mean(eval_steps):.2f}, std: {np.std(eval_steps):.2f}')
        #     # print(f'Success rate: {success_rate:.2%}')
        #     if logging:
        #         writer.add_scalar('eval/avg_reward', np.mean(eval_rewards), i)
        #         writer.add_scalar('eval/avg_steps', np.mean(eval_steps), i)
            # writer.add_scalar('eval/success_rate', success_rate, i)

        # if global_step_counter > max_steps:
        #     break

    # Make Q-value trends available after training without changing return signature.
    agent.avg_q_per_episode = avg_q_per_episode
    agent.avg_max_q_per_episode = avg_max_q_per_episode

    # Make periodic greedy-eval results available after training.
    agent.eval_episodes = eval_episodes
    agent.eval_rewards = eval_rewards

    # Make per-episode profit/safety metrics available (used by profit_safety reward mode).
    agent.profit_per_episode = profit_per_episode
    agent.safety_violation_per_episode = safety_violation_per_episode

    # Make best-eval checkpoint info available after training.
    agent.best_eval_reward = best_eval_reward if eval_episodes else None
    agent.best_eval_episode = best_eval_episode

    # return reward_per_episode, r_unbalance_per_episode, operation_cost_per_episode
    return reward_per_episode, operation_cost_per_episode
        


def evaluate(
    agent,
    env,
    day0, dayn,
    epsilon: Optional[float] = None,
    seed: Optional[int] = 0,
) -> Tuple[List, List]:

    from src.utils import set_seed
    set_seed(env, seed)

    # output metrics
    #reward_per_episode = []
    #r_unbalance_per_episode = []
    # steps_per_episode = []

    eval_days = []
    eval_rewards_per_day = []
    eval_costs_per_day = []
    eval_profits_per_day = []
    eval_violations_per_day = []

    for day in tqdm(range(day0,dayn)):

        state = env.reset(day=day)
        ep_reward = 0
        # r_unbal = 0
        ep_cost = 0
        ep_profit = 0.0
        ep_violation = 0.0
        #steps = 0
        done = False
        while not done:

            action = agent.act(state, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)

            # r_ub, op_co = env.render(mode='Eval')
            op_co = env.render(mode='Eval')

            ep_reward += reward
            # r_unbal += r_ub
            ep_cost += op_co
            ep_profit += float(getattr(env, 'last_profit', 0.0))
            ep_violation = max(ep_violation, float(getattr(env, 'last_safety_violation', 0.0)))
            #steps += 1
            state = next_state

        eval_days.append(day)
        eval_rewards_per_day.append(ep_reward)
        eval_costs_per_day.append(ep_cost)
        eval_profits_per_day.append(ep_profit)
        eval_violations_per_day.append(ep_violation)

    n_days = max(1, len(eval_days))

    # Expose per-day metrics on the agent so callers can cite the greedy
    # performance (profit vs. safety) without changing the return signature.
    agent.eval_days = eval_days
    agent.eval_rewards_per_day = eval_rewards_per_day
    agent.eval_costs_per_day = eval_costs_per_day
    agent.eval_profits_per_day = eval_profits_per_day
    agent.eval_violations_per_day = eval_violations_per_day
    agent.eval_total_reward = float(np.sum(eval_rewards_per_day))
    agent.eval_total_cost = float(np.sum(eval_costs_per_day))
    agent.eval_total_profit = float(np.sum(eval_profits_per_day))
    agent.eval_mean_profit = float(np.mean(eval_profits_per_day))
    agent.eval_peak_violation = float(np.max(eval_violations_per_day))
    agent.eval_mean_violation = float(np.mean(eval_violations_per_day))

    rewards = float(np.sum(eval_rewards_per_day))
    op_cos = float(np.sum(eval_costs_per_day))

    # return rewards, r_unbal, op_cos
    return rewards, op_cos


def plot_reward_cost_curves(
    reward_per_episode: List[float],
    operation_cost_per_episode: List[float],
    moving_avg_window: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    data_label: Optional[str] = None,
):
    """Plot cumulative reward and cumulative cost per episode with moving averages."""

    rewards = np.asarray(reward_per_episode, dtype=np.float64)
    costs = np.asarray(operation_cost_per_episode, dtype=np.float64)

    if rewards.size == 0 or costs.size == 0:
        raise ValueError("reward_per_episode and operation_cost_per_episode must be non-empty")

    if rewards.size != costs.size:
        raise ValueError("reward_per_episode and operation_cost_per_episode must have the same length")

    n = rewards.size
    episodes = np.arange(1, n + 1)

    if moving_avg_window < 1:
        moving_avg_window = 1

    def _moving_avg(x: np.ndarray, w: int) -> np.ndarray:
        if x.size < w:
            return np.array([])
        return np.convolve(x, np.ones(w) / w, mode="valid")

    reward_ma = _moving_avg(rewards, moving_avg_window)
    cost_ma = _moving_avg(costs, moving_avg_window)
    ma_x = np.arange(moving_avg_window, n + 1)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    if data_label:
        fig.suptitle(f"Training data source: {data_label}", fontsize=12, fontweight="bold")

    axs[0].plot(episodes, rewards, color="deepskyblue", alpha=0.45, linewidth=2)
    if reward_ma.size > 0:
        axs[0].plot(ma_x, reward_ma, color="salmon", linewidth=3, label="Moving average")
        axs[0].legend(loc="upper left", frameon=False)
    axs[0].set_ylabel("Cumulative Reward")
    axs[0].grid(True, alpha=0.25)

    axs[1].plot(episodes, costs, color="deepskyblue", alpha=0.45, linewidth=2)
    if cost_ma.size > 0:
        axs[1].plot(ma_x, cost_ma, color="orange", linewidth=3, label="Moving average")
        axs[1].legend(loc="upper right", frameon=False)
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Cumulative Cost")
    axs[1].grid(True, alpha=0.25)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220)

    return fig, axs