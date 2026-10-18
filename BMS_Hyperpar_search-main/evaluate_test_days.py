"""Greedy evaluation of the trained agent on held-out test days (default 300-364).

Computes, for every hour and aggregated per day / over the whole window:

  buy_cost      = buy_price * energy           (energy > 0  -> charging/importing)
  sell_benefit  = sell_price * |energy|        (energy < 0  -> discharging/exporting)
  battery_cost  = energy^2 * degradation
  trading_profit = sell_benefit - buy_cost      (net arbitrage revenue, == env.last_profit)
  op_cost (no degradation) = buy_cost - sell_benefit
  op_cost (with degradation) = buy_cost - sell_benefit + battery_cost

Outputs a per-hour CSV, a per-day CSV and a summary plot under results/.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from src.mg_env import MicroGridEnv
from src.q_agent import QAgent

CKPT = Path('results') / 'best_checkpoint'
DAY0 = 300
DAYN = 365


def evaluate_days(env, agent, day0=DAY0, dayn=DAYN):
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

            action = agent.act(state, epsilon=0.0)
            state, reward, done, _ = env.step(action)

            energy = float(env.battery.energy_change)
            soc = float(env.battery.SOC)

            buy_cost = buy_price * energy if energy > 0 else 0.0
            sell_benefit = sell_price * (-energy) if energy < 0 else 0.0
            battery_cost = energy ** 2 * env.deg
            trading_profit = sell_benefit - buy_cost

            rows.append({
                'day': day,
                'hour': h,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'energy_change': energy,
                'SOC': soc,
                'buy_cost': buy_cost,
                'sell_benefit': sell_benefit,
                'battery_cost': battery_cost,
                'trading_profit': trading_profit,
                'op_cost_no_degradation': buy_cost - sell_benefit,
                'op_cost': buy_cost - sell_benefit + battery_cost,
            })
            if done:
                break

    df = pd.DataFrame(rows)

    daily = df.groupby('day').agg(
        energy_charged=('energy_change', lambda e: e[e > 0].sum()),
        energy_discharged=('energy_change', lambda e: -e[e < 0].sum()),
        buy_cost=('buy_cost', 'sum'),
        sell_benefit=('sell_benefit', 'sum'),
        battery_cost=('battery_cost', 'sum'),
        trading_profit=('trading_profit', 'sum'),
        op_cost_no_degradation=('op_cost_no_degradation', 'sum'),
        op_cost=('op_cost', 'sum'),
    ).reset_index()
    daily['net_profit'] = daily['sell_benefit'] - daily['buy_cost'] - daily['battery_cost']

    return df, daily


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=str(CKPT))
    parser.add_argument('--day0', type=int, default=DAY0)
    parser.add_argument('--dayn', type=int, default=DAYN)
    parser.add_argument('--outdir', type=str, default='results')
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    env = MicroGridEnv(reward_mode='pure_profit')
    agent = QAgent.load_from_disk(env, ckpt)

    df, daily = evaluate_days(env, agent, args.day0, args.dayn)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hourly_csv = out_dir / f'eval_hourly_{args.day0}_{args.dayn - 1}.csv'
    daily_csv = out_dir / f'eval_daily_{args.day0}_{args.dayn - 1}.csv'
    df.to_csv(hourly_csv, index=False)
    daily.to_csv(daily_csv, index=False)

    total = daily.sum(numeric_only=True)

    print('=' * 68)
    print(f'Evaluation of {ckpt} on test days {args.day0}..{args.dayn - 1}')
    print(f'({len(daily)} days, {len(df)} hourly steps)')
    print('=' * 68)
    print(f'Energy charged     : {total["energy_charged"]:12.1f} kWh')
    print(f'Energy discharged  : {total["energy_discharged"]:12.1f} kWh')
    print(f'Buy cost           : {total["buy_cost"]:12.2f}  (energy bought from grid)')
    print(f'Sell benefit       : {total["sell_benefit"]:12.2f}  (energy sold to grid)')
    print(f'Battery deg. cost  : {total["battery_cost"]:12.4f}')
    print('-' * 68)
    print(f'Trading profit     : {total["trading_profit"]:12.2f}  (sell_benefit - buy_cost)')
    print(f'Net profit         : {total["net_profit"]:12.2f}  (incl. degradation cost)')
    print(f'Op cost (no deg.)  : {total["op_cost_no_degradation"]:12.2f}')
    print(f'Op cost (with deg.): {total["op_cost"]:12.2f}')
    print('-' * 68)
    print(f'Mean daily trading profit: {total["trading_profit"] / len(daily):.2f} +/- '
          f'{daily["trading_profit"].std():.2f}')
    print(f'Best day            : day {int(daily.loc[daily["trading_profit"].idxmax(), "day"])} '
          f'({daily["trading_profit"].max():.2f})')
    print(f'Worst day           : day {int(daily.loc[daily["trading_profit"].idxmin(), "day"])} '
          f'({daily["trading_profit"].min():.2f})')
    print(f'Saved hourly csv   : {hourly_csv}')
    print(f'Saved daily csv    : {daily_csv}')

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    days = daily['day'].values

    axs[0].bar(days, daily['trading_profit'].values, color='seagreen')
    axs[0].axhline(0.0, color='black', linewidth=0.8)
    axs[0].set_ylabel('Daily trading profit')
    axs[0].set_title(f'Trained agent ({ckpt}) on test days {args.day0}..{args.dayn - 1}')
    axs[0].grid(True, linestyle='--', alpha=0.35)

    axs[1].bar(days, daily['buy_cost'].values, color='tomato', label='Buy cost')
    axs[1].bar(days, daily['sell_benefit'].values, color='steelblue', label='Sell benefit')
    axs[1].set_ylabel('Cost / benefit')
    axs[1].legend(frameon=False)
    axs[1].grid(True, linestyle='--', alpha=0.35)

    axs[2].plot(days, daily['trading_profit'].cumsum().values, color='purple', linewidth=2)
    axs[2].axhline(0.0, color='black', linewidth=0.8)
    axs[2].set_ylabel('Cumulative trading profit')
    axs[2].set_xlabel('Day')
    axs[2].grid(True, linestyle='--', alpha=0.35)

    fig.tight_layout()
    png_path = out_dir / f'eval_test_days_{args.day0}_{args.dayn - 1}.png'
    fig.savefig(str(png_path), dpi=200)
    print(f'Saved plot         : {png_path}')


if __name__ == '__main__':
    main()
