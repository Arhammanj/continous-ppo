"""PPO training entry point (Stable-Baselines3, ``pure_profit`` objective).

Run from the project root::

    python -m ppo.train
    python ppo/train.py --timesteps 50000

The environment dynamics, safety shield, price data, action semantics and
pure-profit reward are *identical* to the DQN baseline; PPO only interacts with
the env through the thin :class:`ppo.adapter.GymnasiumAdapter`. Economic profit
and operation cost are logged by a callback (the env's ``render`` is not used
for metrics).
"""
from __future__ import annotations

import argparse
import csv
import json
import ast
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from evaluation.metrics import load_config, make_env
from ppo.adapter import GymnasiumAdapter


class EconomicMetricsCallback(BaseCallback):
    """Log *actual* economic profit and operation cost per episode.

    Values are read directly from the underlying ``MicroGridEnv`` (not from the
    normalized training reward and not from ``env.render()``).
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_profits: list[float] = []
        self.episode_costs: list[float] = []
        self.episode_rewards: list[float] = []
        self._ep_profit = 0.0
        self._ep_cost = 0.0
        self._ep_reward = 0.0

    @staticmethod
    def _inner_env(vec_env):
        env = vec_env
        if hasattr(env, "envs") and len(env.envs) > 0:
            env = env.envs[0]
        guard = 0
        while hasattr(env, "env") and not hasattr(env, "last_profit") and guard < 20:
            env = env.env
            guard += 1
        return env

    def _on_step(self) -> bool:
        env = self._inner_env(self.training_env)
        self._ep_profit += float(getattr(env, "last_profit", 0.0))
        self._ep_cost += float(getattr(env, "operation_cost", 0.0))
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self._ep_reward += float(rewards[0])

        dones = self.locals.get("dones")
        done = bool(dones[0]) if dones is not None else False
        if done:
            self.episode_profits.append(self._ep_profit)
            self.episode_costs.append(self._ep_cost)
            self.episode_rewards.append(self._ep_reward)
            if self.logger is not None:
                self.logger.record("train/episode_economic_profit", self._ep_profit)
                self.logger.record("train/episode_operation_cost", self._ep_cost)
                self.logger.record("train/episode_reward", self._ep_reward)
            self._ep_profit = 0.0
            self._ep_cost = 0.0
            self._ep_reward = 0.0
        return True


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)


def main():
    parser = argparse.ArgumentParser(description="Train PPO with pure_profit reward")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "ppo.yaml"))
    parser.add_argument("--timesteps", type=int, default=None,
                        help="override total_timesteps from the config")
    parser.add_argument("--seed", type=int, default=None,
                        help="override seed from the config")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--log-std-init", type=float, default=None)
    parser.add_argument("--net-arch", type=str, default=None,
                        help="Python literal list, e.g. '[128,128]'")
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--result-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    ppo_cfg = cfg["ppo"]
    model_dir = resolve(args.model_dir or cfg["paths"]["model_dir"])
    result_dir = resolve(args.result_dir or cfg["paths"]["result_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = args.timesteps if args.timesteps is not None else ppo_cfg["total_timesteps"]
    seed = args.seed if args.seed is not None else ppo_cfg["seed"]
    parameters = {
        "policy": ppo_cfg["policy"],
        "learning_rate": args.learning_rate if args.learning_rate is not None else ppo_cfg["learning_rate"],
        "n_steps": args.n_steps if args.n_steps is not None else ppo_cfg["n_steps"],
        "batch_size": args.batch_size if args.batch_size is not None else ppo_cfg["batch_size"],
        "n_epochs": args.n_epochs if args.n_epochs is not None else ppo_cfg["n_epochs"],
        "gamma": args.gamma if args.gamma is not None else ppo_cfg["gamma"],
        "gae_lambda": args.gae_lambda if args.gae_lambda is not None else ppo_cfg["gae_lambda"],
        "ent_coef": args.ent_coef if args.ent_coef is not None else ppo_cfg["ent_coef"],
        "log_std_init": args.log_std_init if args.log_std_init is not None else ppo_cfg.get("log_std_init", 0.0),
        "net_arch": ast.literal_eval(args.net_arch) if args.net_arch is not None else ppo_cfg.get("net_arch", [64, 64]),
        "vf_coef": args.vf_coef if args.vf_coef is not None else ppo_cfg.get("vf_coef", 0.5),
        "clip_range": args.clip_range if args.clip_range is not None else ppo_cfg.get("clip_range", 0.2),
        "max_grad_norm": args.max_grad_norm if args.max_grad_norm is not None else ppo_cfg.get("max_grad_norm", 0.5),
        "seed": seed,
        "total_timesteps": total_timesteps,
    }

    # Shared environment, single standardized objective (explicit, not default).
    env = make_env(day0=env_cfg["day0"], dayn=env_cfg["dayn"],
                   reward_mode=env_cfg["reward_mode"])
    adapter = GymnasiumAdapter(env)

    model = PPO(
        ppo_cfg["policy"],
        adapter,
        learning_rate=parameters["learning_rate"],
        n_steps=parameters["n_steps"],
        batch_size=parameters["batch_size"],
        n_epochs=parameters["n_epochs"],
        gamma=parameters["gamma"],
        gae_lambda=parameters["gae_lambda"],
        ent_coef=parameters["ent_coef"],
        policy_kwargs={
            "log_std_init": parameters["log_std_init"],
            "net_arch": parameters["net_arch"],
        },
        vf_coef=parameters["vf_coef"],
        clip_range=parameters["clip_range"],
        max_grad_norm=parameters["max_grad_norm"],
        seed=seed,
        verbose=1,
    )

    callback = EconomicMetricsCallback()
    print(f"Training PPO for {total_timesteps} timesteps "
          f"(reward_mode={env_cfg['reward_mode']}, seed={seed})")
    started_at = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    runtime_seconds = time.time() - started_at

    save_path = model_dir / "ppo_model"
    model.save(save_path)
    print(f"PPO model saved to {save_path}.zip")

    csv_path = result_dir / "ppo_training_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "economic_profit", "operation_cost", "reward"])
        for i, (p, c, r) in enumerate(zip(callback.episode_profits,
                                          callback.episode_costs,
                                          callback.episode_rewards)):
            writer.writerow([i + 1, p, c, r])
    print(f"Training metrics saved to {csv_path} "
          f"({len(callback.episode_profits)} episodes logged)")
    with open(result_dir / "experiment_metadata.json", "w") as f:
        json.dump({
            "parameters": parameters,
            "environment": env_cfg,
            "runtime_seconds": runtime_seconds,
            "actual_environment_steps": model.num_timesteps,
            "ppo_updates": model.num_timesteps // parameters["n_steps"],
        }, f, indent=2)


if __name__ == "__main__":
    main()
