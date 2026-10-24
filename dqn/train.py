"""DQN training entry point (Double DQN, ``pure_profit`` objective).

Run from the project root::

    python -m dqn.train
    python dqn/train.py --episodes 200

The environment, reward and DQN algorithm are the same as the reference
experiment (``runner_train.py``); only the output locations and configuration
loading have been standardized (``configs/dqn.yaml``, ``models/dqn``,
``results/dqn``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the project root importable whether run as ``python dqn/train.py`` or
# ``python -m dqn.train``.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dqn.agent import QAgent
from evaluation.metrics import load_config, make_env
from src.loops import train, plot_reward_cost_curves
from src.utils import set_seed


def greedy_eval(agent, env, n_episodes=10, seed=0):
    """Greedy evaluation callback (epsilon forced to 0)."""
    set_seed(env, seed)
    ep_rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            action = agent.act(state, epsilon=0.0)
            next_state, reward, done, _ = env.step(action)
            ep_r += float(reward)
            state = next_state
        ep_rewards.append(ep_r)
    return float(np.mean(ep_rewards))


def build_agent(env, agent_cfg):
    return QAgent(
        env,
        learning_rate=agent_cfg["learning_rate"],
        discount_factor=agent_cfg["discount_factor"],
        batch_size=agent_cfg["batch_size"],
        memory_size=agent_cfg["memory_size"],
        freq_steps_update_target=agent_cfg["freq_steps_update_target"],
        n_steps_warm_up_memory=agent_cfg["n_steps_warm_up_memory"],
        freq_steps_train=agent_cfg["freq_steps_train"],
        n_gradient_steps=agent_cfg["n_gradient_steps"],
        nn_hidden_layers=agent_cfg["nn_hidden_layers"],
        max_grad_norm=agent_cfg["max_grad_norm"],
        normalize_state=agent_cfg["normalize_state"],
        use_double_dqn=agent_cfg["use_double_dqn"],
        epsilon_start=agent_cfg["epsilon_start"],
        epsilon_end=agent_cfg["epsilon_end"],
        steps_epsilon_decay=agent_cfg["steps_epsilon_decay"],
    )


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)


def main():
    parser = argparse.ArgumentParser(description="Train DQN with pure_profit reward")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "dqn.yaml"))
    parser.add_argument("--episodes", type=int, default=None,
                        help="override n_episodes from the config")
    parser.add_argument("--eval-freq", type=int, default=None,
                        help="override greedy-eval frequency (episodes)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    train_cfg = cfg["train"]
    model_dir = resolve(cfg["paths"]["model_dir"])
    result_dir = resolve(cfg["paths"]["result_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = args.episodes if args.episodes is not None else train_cfg["n_episodes"]
    eval_freq = args.eval_freq if args.eval_freq is not None else train_cfg["eval_freq"]

    # Shared environment, single standardized objective (explicit, not default).
    env = make_env(day0=env_cfg["day0"], dayn=env_cfg["dayn"],
                   reward_mode=env_cfg["reward_mode"])
    set_seed(env, train_cfg["seed"])

    agent = build_agent(env, cfg["agent"])

    ckpt_dir = model_dir / "best_checkpoint"
    print(f"Training DQN for {n_episodes} episodes "
          f"(reward_mode={env_cfg['reward_mode']}, normalize_state={agent.normalize_state})")
    print(f"  best checkpoint dir: {ckpt_dir}")

    rewards, costs = train(
        agent, env, n_episodes=n_episodes, log_dir=None,
        eval_fn=greedy_eval, freq_episodes_eval=eval_freq,
        checkpoint_dir=ckpt_dir,
    )

    # Persist the final agent as well as the best-eval checkpoint.
    final_dir = model_dir / "final"
    agent.save_to_disk(final_dir)

    out_path = result_dir / f"reward_cost_{n_episodes}eps.png"
    fig, axs = plot_reward_cost_curves(rewards, costs, moving_avg_window=50,
                                       save_path=str(out_path),
                                       data_label="src/Prices.csv")
    if len(agent.eval_episodes) > 0:
        axs[0].plot(agent.eval_episodes, agent.eval_rewards, marker="o",
                    markersize=5, color="black", linewidth=2,
                    label="Greedy eval (eps=0)")
        axs[0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=220)
    plt.close(fig)

    print(f"Training plot saved to {out_path}")
    print(f"Final agent saved to {final_dir}")
    print(f"Best greedy eval: episode {agent.best_eval_episode}, "
          f"reward={agent.best_eval_reward}")


if __name__ == "__main__":
    main()
