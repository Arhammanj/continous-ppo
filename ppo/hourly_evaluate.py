"""Export an hourly PPO trace with continuous SOC across consecutive days."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import load_config, make_env
from ppo.adapter import GymnasiumAdapter
from src.mg_env import ACTIONS


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export hourly PPO results with continuous SOC"
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "ppo.yaml"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--day0", type=int, default=None)
    parser.add_argument("--dayn", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    day0 = args.day0 if args.day0 is not None else cfg["eval"]["day0"]
    dayn = args.dayn if args.dayn is not None else cfg["eval"]["dayn"]
    model_path = resolve(args.model) if args.model else (
        resolve(cfg["paths"]["model_dir"]) / "ppo_model"
    )
    output = resolve(args.output) if args.output else (
        resolve(cfg["paths"]["result_dir"]) / "ppo_hourly_eval.csv"
    )

    load_env = GymnasiumAdapter(make_env(
        day0=env_cfg["day0"],
        dayn=env_cfg["dayn"],
        reward_mode=env_cfg["reward_mode"],
    ))
    model = PPO.load(str(model_path), env=load_env)

    # One environment is intentionally reused so reset(day + 1) carries the
    # previous day's terminal SOC into the next day.
    env = make_env(
        day0=day0,
        dayn=dayn,
        reward_mode=env_cfg["reward_mode"],
    )
    profit_scale = env._get_profit_norm_scale()
    rows = []
    global_cumulative_profit = 0.0

    for day in range(day0, dayn):
        state = env.reset(day=day)
        daily_cumulative_profit = 0.0
        day_start_soc = float(env.battery.SOC)

        for hour in range(env.iterations):
            start_soc = float(env.battery.SOC)
            index = (day * env.iterations + hour) % len(env.grid.buy_prices)
            action, _ = model.predict(
                np.asarray(state, dtype=np.float32), deterministic=True
            )
            normalized_action = float(np.asarray(action).reshape(-1)[0])
            physical_action = normalized_action * float(max(abs(value) for value in ACTIONS))
            state, reward, done, _ = env.step(
                physical_action
            )
            profit = float(env.last_profit)
            daily_cumulative_profit += profit
            global_cumulative_profit += profit
            profit_reward = profit / profit_scale

            rows.append({
                "day": day,
                "hour": hour,
                "buy_price": float(env.grid.buy_prices[index]),
                "sell_price": float(env.grid.sell_prices[index]),
                "normalized_action": normalized_action,
                "battery_energy_change": float(env.battery.energy_change),
                "start_soc": start_soc,
                "end_soc": float(env.battery.SOC),
                "hourly_profit": profit,
                "profit_reward": profit_reward,
                "price_opportunity_adjustment": (
                    float(env.last_price_opportunity_adjustment)
                ),
                "terminal_market_value": (
                    env._terminal_market_value() if done else 0.0
                ),
                "hourly_reward": float(reward),
                "daily_cumulative_profit": daily_cumulative_profit,
                "global_cumulative_profit": global_cumulative_profit,
                "terminal": bool(done),
                "day_start_soc": day_start_soc,
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} hourly rows to {output}")


if __name__ == "__main__":
    main()
