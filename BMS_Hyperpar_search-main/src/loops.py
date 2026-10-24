from typing import Tuple, List, Callable, Union, Optional
import random
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
    #global_step_counter = 0

    for i in tqdm(range(0, n_episodes)):

        state = env.reset()
        rewards = 0
        # r_unbal = 0
        op_cost = 0
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

    for day in tqdm(range(day0,dayn)):

        state = env.reset(day=day)
        rewards = 0
        # r_unbal = 0
        op_cos = 0
        #steps = 0
        done = False
        while not done:

            action = agent.act(state, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)

            # r_ub, op_co = env.render(mode='Eval')
            op_co = env.render(mode='Eval')

            rewards += reward
            # r_unbal += r_ub
            op_cos += op_co
            #steps += 1
            state = next_state
        
        #print('R: ', rewards)
        #reward_per_episode.append(rewards)
        #r_unbalance_per_episode.append(r_unbal)

    # return rewards, r_unbal, op_cos
    return rewards, op_cos


def plot_reward_cost_curves(
    reward_per_episode: List[float],
    operation_cost_per_episode: List[float],
    moving_avg_window: int = 10,
    save_path: Optional[Union[str, Path]] = None,
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