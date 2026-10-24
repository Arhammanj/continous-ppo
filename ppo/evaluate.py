"""PPO evaluation on the shared held-out days (default 300-364).

Run from the project root::

    python -m ppo.evaluate
    python ppo/evaluate.py --model models/ppo/ppo_model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from evaluation.metrics import (
    evaluate_agent, load_config, make_env, print_evaluation, save_evaluation,
)
from ppo.adapter import GymnasiumAdapter


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "ppo.yaml"))
    parser.add_argument("--model", type=str, default=None,
                        help="model path (default: <model_dir>/ppo_model)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    model_dir = resolve(cfg["paths"]["model_dir"])
    result_dir = resolve(cfg["paths"]["result_dir"])

    model_path = resolve(args.model) if args.model else model_dir / "ppo_model"
    if not Path(str(model_path) + ".zip").exists() and not Path(model_path).exists():
        raise FileNotFoundError(f"PPO model not found: {model_path}.zip")

    # The adapter is only used to give PPO the correct spaces at load time.
    load_env = GymnasiumAdapter(make_env(day0=env_cfg["day0"], dayn=env_cfg["dayn"],
                                         reward_mode=env_cfg["reward_mode"]))
    model = PPO.load(str(model_path), env=load_env)

    # Evaluation uses the shared inner environment directly (same held-out days).
    env = make_env(day0=env_cfg["day0"], dayn=env_cfg["dayn"],
                   reward_mode=env_cfg["reward_mode"])

    def action_fn(state):
        obs = np.asarray(state, dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(1)

    day0, dayn = cfg["eval"]["day0"], cfg["eval"]["dayn"]
    results = evaluate_agent(env, action_fn, day0=day0, dayn=dayn)
    print_evaluation("PPO", results)
    save_evaluation(result_dir, "PPO", results, checkpoint=str(model_path),
                    reward_mode=env_cfg["reward_mode"])
    return results


if __name__ == "__main__":
    main()
