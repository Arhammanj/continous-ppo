"""Shared economic metrics and the common evaluation protocol.

Both the DQN and PPO pipelines use this module so that held-out evaluation is
performed identically for the two algorithms.

Key points:
- Economic profit is taken from the environment's *actual* profit accounting
  (``env.last_profit`` = ``sell_benefit - buy_cost``), NOT from summing the
  normalized training reward.
- Only held-out days are evaluated (default 300-364); training never touches
  those days.
- Actions are always produced by a deterministic policy (``epsilon = 0`` for
  DQN, ``deterministic = True`` for PPO).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import yaml

from src.mg_env import MicroGridEnv
from src.config import REWARD_MODE, TRAIN_DAY0, TRAIN_DAYN, EVAL_DAY0, EVAL_DAYN


def load_config(path) -> Dict:
    """Load a YAML experiment configuration."""
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)


def make_env(day0: int = TRAIN_DAY0, dayn: int = TRAIN_DAYN,
             reward_mode: str = REWARD_MODE, **kwargs) -> MicroGridEnv:
    """Build the shared microgrid environment.

    The reward objective is always ``pure_profit`` (the only supported training
    reward); callers may still override environment geometry such as ``day0``
    and ``dayn``.
    """
    return MicroGridEnv(day0=day0, dayn=dayn, reward_mode=reward_mode, **kwargs)


def run_episode(env: MicroGridEnv, action_fn: Callable[[np.ndarray], object],
                day: int) -> Dict:
    """Run one deterministic episode (one simulated day) and collect metrics."""
    state = env.reset(day=day)
    done = False

    total_reward = 0.0
    profit = 0.0
    op_cost = 0.0
    energy_bought = 0.0
    energy_sold = 0.0
    soc_values: List[float] = []
    ch_dch_values: List[float] = []
    max_violation = 0.0
    n_steps = 0

    while not done:
        action = action_fn(state)
        state, reward, done, info = env.step(action)

        total_reward += float(reward)
        # Economic profit in currency units (not the normalized reward).
        profit += float(getattr(env, "last_profit", 0.0))
        op_cost += float(getattr(env, "operation_cost", 0.0))

        grid_flow = float(getattr(env, "total_current_output", 0.0))
        if grid_flow >= 0:
            energy_bought += grid_flow
        else:
            energy_sold += -grid_flow

        soc_values.append(float(env.battery.SOC))
        ch_dch_values.append(float(env.battery.energy_change))
        max_violation = max(
            max_violation, float(getattr(env, "last_safety_violation", 0.0))
        )
        n_steps += 1

    return {
        "day": int(day),
        "total_reward": total_reward,
        "profit": profit,
        "operation_cost": op_cost,
        "energy_bought": energy_bought,
        "energy_sold": energy_sold,
        "soc_min": float(np.min(soc_values)) if soc_values else float("nan"),
        "soc_max": float(np.max(soc_values)) if soc_values else float("nan"),
        "max_violation": max_violation,
        "n_steps": n_steps,
        "actions": ch_dch_values,
    }


def evaluate_agent(env: MicroGridEnv,
                   action_fn: Callable[[np.ndarray], object],
                   day0: int = EVAL_DAY0,
                   dayn: int = EVAL_DAYN) -> Dict:
    """Evaluate a deterministic policy over a range of held-out days."""
    per_day = [run_episode(env, action_fn, day) for day in range(day0, dayn)]

    daily_profit = np.asarray([d["profit"] for d in per_day], dtype=np.float64)
    daily_cost = np.asarray([d["operation_cost"] for d in per_day], dtype=np.float64)
    daily_reward = np.asarray([d["total_reward"] for d in per_day], dtype=np.float64)

    return {
        "day0": int(day0),
        "dayn": int(dayn),
        "n_days": len(per_day),
        "per_day": per_day,
        "daily_profit": daily_profit,
        "daily_operation_cost": daily_cost,
        "daily_reward": daily_reward,
        "total_profit": float(daily_profit.sum()),
        "mean_daily_profit": float(daily_profit.mean()),
        "std_daily_profit": float(daily_profit.std()),
        "median_daily_profit": float(np.median(daily_profit)),
        "total_operation_cost": float(daily_cost.sum()),
        "total_energy_bought": float(np.sum([d["energy_bought"] for d in per_day])),
        "total_energy_sold": float(np.sum([d["energy_sold"] for d in per_day])),
        "max_violation": float(np.max([d["max_violation"] for d in per_day])),
        "days_with_positive_profit": int(np.sum(daily_profit > 0)),
        "mean_daily_reward": float(daily_reward.mean()),
    }


def summary_rows(results: Dict) -> List[List]:
    """Return rows suitable for printing a comparison table."""
    return [
        ["days evaluated", results["n_days"]],
        ["total economic profit", f"{results['total_profit']:.2f}"],
        ["mean daily profit", f"{results['mean_daily_profit']:.3f}"],
        ["std daily profit", f"{results['std_daily_profit']:.3f}"],
        ["median daily profit", f"{results['median_daily_profit']:.3f}"],
        ["days with profit > 0", f"{results['days_with_positive_profit']}/{results['n_days']}"],
        ["total operation cost", f"{results['total_operation_cost']:.2f}"],
        ["total energy bought", f"{results['total_energy_bought']:.1f}"],
        ["total energy sold", f"{results['total_energy_sold']:.1f}"],
        ["max safety violation", f"{results['max_violation']:.4f}"],
    ]


def print_evaluation(name: str, results: Dict) -> None:
    print(f"\n=== {name} evaluation (days {results['day0']}-{results['dayn'] - 1}) ===")
    for label, value in summary_rows(results):
        print(f"  {label:>24}: {value}")


def save_evaluation(result_dir, name: str, results: Dict,
                    checkpoint: str = None, reward_mode: str = REWARD_MODE) -> None:
    """Persist evaluation results (JSON + per-day CSV + daily-profit plot)."""
    import csv
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    days = [d["day"] for d in results["per_day"]]

    out = {
        "algorithm": name.lower(),
        "checkpoint": checkpoint,
        "reward_mode": reward_mode,
        "summary": {label: value for label, value in summary_rows(results)},
        "days": days,
        "daily_profit": results["daily_profit"].tolist(),
    }
    with open(result_dir / "eval.json", "w") as f:
        json.dump(out, f, indent=2)

    with open(result_dir / "eval_daily.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["day", "profit", "operation_cost", "total_reward",
                         "energy_bought", "energy_sold", "soc_min", "soc_max",
                         "max_violation", "n_steps"])
        for d in results["per_day"]:
            writer.writerow([d["day"], d["profit"], d["operation_cost"],
                             d["total_reward"], d["energy_bought"],
                             d["energy_sold"], d["soc_min"], d["soc_max"],
                             d["max_violation"], d["n_steps"]])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(days, results["daily_profit"], color="mediumseagreen")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Held-out day")
    ax.set_ylabel("Daily economic profit")
    ax.set_title(f"{name} daily profit on held-out days "
                 f"{results['day0']}-{results['dayn'] - 1} "
                 f"(mean {results['mean_daily_profit']:.2f})")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(str(result_dir / "eval_daily_profit.png"), dpi=200)
    plt.close(fig)

    print(f"Saved {result_dir / 'eval.json'}, {result_dir / 'eval_daily.csv'}, "
          f"{result_dir / 'eval_daily_profit.png'}")
