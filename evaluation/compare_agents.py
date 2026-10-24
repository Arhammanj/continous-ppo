"""Compare trained DQN and PPO agents on the same held-out evaluation days.

Run from the project root::

    python -m evaluation.compare_agents

Both agents are evaluated with deterministic policies on identical held-out
days (default 300-364). Economic profit comes from the environment's actual
profit accounting, not from summing normalized training rewards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import (
    evaluate_agent, load_config, make_env, print_evaluation, summary_rows,
)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)


def build_dqn_action_fn(cfg):
    from dqn.agent import QAgent
    env_cfg = cfg["env"]
    model_dir = resolve(cfg["paths"]["model_dir"])
    ckpt = model_dir / "best_checkpoint"
    if not ckpt.exists():
        raise FileNotFoundError(f"DQN checkpoint not found: {ckpt}")
    env = make_env(day0=env_cfg["day0"], dayn=env_cfg["dayn"],
                   reward_mode=env_cfg["reward_mode"])
    agent = QAgent.load_from_disk(env, ckpt)
    return env, (lambda state: agent.act(state, epsilon=0.0))


def build_ppo_action_fn(cfg):
    from stable_baselines3 import PPO
    from ppo.adapter import GymnasiumAdapter
    env_cfg = cfg["env"]
    model_dir = resolve(cfg["paths"]["model_dir"])
    model_path = model_dir / "ppo_model"
    if not Path(str(model_path) + ".zip").exists():
        raise FileNotFoundError(f"PPO model not found: {model_path}.zip")
    load_env = GymnasiumAdapter(make_env(day0=env_cfg["day0"], dayn=env_cfg["dayn"],
                                         reward_mode=env_cfg["reward_mode"]))
    model = PPO.load(str(model_path), env=load_env)
    env = make_env(day0=env_cfg["day0"], dayn=env_cfg["dayn"],
                   reward_mode=env_cfg["reward_mode"])

    def action_fn(state):
        obs = np.asarray(state, dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(1)

    return env, action_fn


def main():
    parser = argparse.ArgumentParser(description="Compare DQN and PPO agents")
    parser.add_argument("--dqn-config", type=str, default=str(ROOT / "configs" / "dqn.yaml"))
    parser.add_argument("--ppo-config", type=str, default=str(ROOT / "configs" / "ppo.yaml"))
    args = parser.parse_args()

    dqn_cfg = load_config(args.dqn_config)
    ppo_cfg = load_config(args.ppo_config)
    day0, dayn = dqn_cfg["eval"]["day0"], dqn_cfg["eval"]["dayn"]

    results = {}
    errors = {}
    for name, builder, cfg in (("DQN", build_dqn_action_fn, dqn_cfg),
                               ("PPO", build_ppo_action_fn, ppo_cfg)):
        try:
            env, action_fn = builder(cfg)
            results[name] = evaluate_agent(env, action_fn, day0=day0, dayn=dayn)
        except FileNotFoundError as e:
            errors[name] = str(e)

    for name, res in results.items():
        print_evaluation(name, res)
    for name, err in errors.items():
        print(f"\n[{name}] skipped: {err}")

    if len(results) == 2:
        dqn = results["DQN"]
        ppo = results["PPO"]
        print("\n=== Comparison (same held-out days) ===")
        print(f"  {'metric':>24} | {'DQN':>12} | {'PPO':>12}")
        for label, _ in summary_rows(dqn):
            dqn_val = dict(summary_rows(dqn))[label]
            ppo_val = dict(summary_rows(ppo))[label]
            print(f"  {label:>24} | {str(dqn_val):>12} | {str(ppo_val):>12}")

        out_dir = ROOT / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {name: {
            "summary": dict(summary_rows(res)),
            "days": [d["day"] for d in res["per_day"]],
            "daily_profit": res["daily_profit"].tolist(),
        } for name, res in results.items()}
        with open(out_dir / "comparison.json", "w") as f:
            json.dump(payload, f, indent=2)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(13, 4.5))
        width = 0.4
        days = np.asarray([d["day"] for d in dqn["per_day"]])
        ax.bar(days - width / 2, dqn["daily_profit"], width, label="DQN",
               color="steelblue")
        ax.bar(days + width / 2, ppo["daily_profit"], width, label="PPO",
               color="darkorange")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Held-out day")
        ax.set_ylabel("Daily economic profit")
        ax.set_title(f"DQN vs PPO daily profit (days {day0}-{dayn - 1})")
        ax.legend(frameon=False)
        ax.grid(True, linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(str(out_dir / "comparison.png"), dpi=200)
        plt.close(fig)
        print(f"\nSaved {out_dir / 'comparison.json'} and {out_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
