"""Export hourly battery operation results (with electricity prices) to CSV.

Runs the trained PPO agent greedily on held-out test days and logs, for every
timestep: day, hour, buy/sell price, charge power, discharge power, net power,
SOC at start and end of the step, the pure_profit reward, and the raw
trading profit.  Uses the real environment (src/mg_env.py) and the prices
loaded from src/Prices.csv.  No dummy data.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.mg_env import MicroGridEnv
from src.ppo_agent import PPOAgent

CKPT = Path('results') / 'v1new_ppo' / 'best_checkpoint'
DAY0 = 300
DAYN = 365


def export_hourly(env, agent, day0=DAY0, dayn=DAYN):
    rows = []
    n = len(env.grid.buy_prices)
    # Deterministic initial SOC for the FIRST day only; every following day
    # continues from the previous day's ending SOC (no re-randomization).
    np.random.seed(0)
    for day in range(day0, dayn):
        state = env.reset(day=day, carry_soc=(day != day0))
        for h in range(24):
            idx = (day * 24 + h) % n
            buy_price = float(env.grid.buy_prices[idx])
            sell_price = float(env.grid.sell_prices[idx])

            soc_start = float(env.battery.SOC)
            action = agent.act(state, epsilon=0.0)  # PPO: deterministic mean action
            state, reward, done, _ = env.step(action)

            energy = float(env.battery.energy_change)
            soc_end = float(env.battery.SOC)

            pch = energy if energy > 0 else 0.0        # charging power (kW, >=0)
            pdch = -energy if energy < 0 else 0.0      # discharging power (kW, >=0)
            net_kw = energy                             # net battery power (+ch / -dch)

            buy_cost = buy_price * energy if energy > 0 else 0.0
            sell_benefit = sell_price * (-energy) if energy < 0 else 0.0
            trading_profit = sell_benefit - buy_cost   # == env.last_profit

            rows.append({
                'day': day,
                'hour': h,
                'buy_price_eur_per_kwh': buy_price,
                'sell_price_eur_per_kwh': sell_price,
                'P_charge_kW': pch,
                'P_discharge_kW': pdch,
                'P_net_kW': net_kw,
                'SOC_start': soc_start,
                'SOC_end': soc_end,
                'reward_pure_profit': float(reward),
                'trading_profit_eur': trading_profit,
            })
            if done:
                break
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=str(CKPT))
    parser.add_argument('--day0', type=int, default=DAY0)
    parser.add_argument('--dayn', type=int, default=DAYN)
    parser.add_argument('--outdir', type=str, default='results/v1new_ppo')
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    env = MicroGridEnv(reward_mode='pure_profit')
    agent = PPOAgent.load_from_disk(env, ckpt)

    df = export_hourly(env, agent, args.day0, args.dayn)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'battery_hourly_with_price_{args.day0}_{args.dayn - 1}.csv'
    df.to_csv(out_csv, index=False)

    # ---- verification against the simulation internals ----
    e_chg = df['P_charge_kW'].sum()
    e_dch = df['P_discharge_kW'].sum()
    profit = df['trading_profit_eur'].sum()
    reward = df['reward_pure_profit'].sum()
    # pure_profit reward = profit / norm_scale, recomputed independently
    scale = env._get_profit_norm_scale()
    print('=' * 70)
    print(f'Checkpoint          : {ckpt}')
    print(f'Reward mode         : {env.reward_mode}')
    print(f'Test window         : days {args.day0}..{args.dayn - 1}  ({len(df)} hourly steps)')
    print(f'Energy charged      : {e_chg:.2f} kWh')
    print(f'Energy discharged   : {e_dch:.2f} kWh')
    print(f'Trading profit      : {profit:.2f} EUR')
    print(f'Sum of rewards      : {reward:.4f}  (profit/scale, scale={scale:.4f})')
    print(f'profit / scale      : {profit / scale:.4f}   <- must match sum of rewards')
    assert np.isclose(reward, profit / scale, rtol=1e-4), 'reward/profit mismatch'
    # SOC consistency: SOC_end - SOC_start == net_kW / capacity
    cap = env.battery.capacity
    soc_delta = df['SOC_end'] - df['SOC_start']
    assert np.allclose(soc_delta, df['P_net_kW'] / cap, atol=1e-9), 'SOC/power mismatch'
    # charge and discharge are mutually exclusive
    assert (df['P_charge_kW'] * df['P_discharge_kW'] == 0).all()
    # cross-day SOC continuity: each day starts exactly where the previous ended
    daily_end = df.groupby('day')['SOC_end'].last()
    daily_start = df.groupby('day')['SOC_start'].first()
    assert np.allclose(daily_start.iloc[1:].values, daily_end.iloc[:-1].values,
                       atol=1e-9), 'SOC not continuous across day boundaries'
    print('Verification passed: reward = profit/scale, SOC delta = P_net/capacity,')
    print('                     charge & discharge mutually exclusive,')
    print('                     SOC continuous across day boundaries.')
    print(f'Saved CSV           : {out_csv.resolve()}')
    print('=' * 70)


if __name__ == '__main__':
    main()
