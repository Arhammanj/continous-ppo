"""Evaluate isolated PPO experiments without changing training artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import evaluate_agent, make_env
from ppo.adapter import GymnasiumAdapter


def evaluate_seed(seed_dir: Path, day0: int, dayn: int) -> dict:
    model_path = seed_dir / "model" / "ppo_model"
    metadata_path = seed_dir / "training" / "experiment_metadata.json"
    if not Path(str(model_path) + ".zip").exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}.zip")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    env = make_env(day0=0, dayn=300, reward_mode="pure_profit")
    load_env = GymnasiumAdapter(env)
    model = PPO.load(str(model_path), env=load_env)

    def action_fn(state):
        action, _ = model.predict(np.asarray(state, dtype=np.float32), deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(1)

    results = evaluate_agent(env, action_fn, day0=day0, dayn=dayn)
    per_day = results["per_day"]
    actions = [action for day in per_day for action in day["actions"]]
    return {
        "seed": metadata.get("parameters", {}).get("seed", seed_dir.name),
        "day0": day0,
        "dayn": dayn,
        "n_days": results["n_days"],
        "mean_daily_profit": results["mean_daily_profit"],
        "std_daily_profit": results["std_daily_profit"],
        "total_profit": results["total_profit"],
        "max_violation": results["max_violation"],
        "total_energy_bought": results["total_energy_bought"],
        "total_energy_sold": results["total_energy_sold"],
        "total_throughput": results["total_energy_bought"] + results["total_energy_sold"],
        "charge_hours": sum(action > 0 for action in actions),
        "discharge_hours": sum(action < 0 for action in actions),
        "idle_hours": sum(action == 0 for action in actions),
        "actual_environment_steps": metadata.get("actual_environment_steps"),
        "ppo_updates": metadata.get("ppo_updates"),
        "runtime_seconds": metadata.get("runtime_seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--day0", type=int, required=True)
    parser.add_argument("--dayn", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.experiment_dir)
    rows = [
        evaluate_seed(seed_dir, args.day0, args.dayn)
        for seed_dir in sorted(root.glob("seed_*"))
    ]
    if not rows:
        raise FileNotFoundError(f"No seed directories found under {root}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
