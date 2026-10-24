"""Single-day greedy trace of the best checkpoint.

4 panels: price / SOC / charge-discharge / cumulative profit,
so you can see directly whether the agent times charge/discharge around
the real price peak instead of a fixed hour-of-day schedule.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.mg_env import MicroGridEnv
from src.q_agent import QAgent

CKPT = Path('results') / 'best_checkpoint'
DEFAULT_DAY = 300


def run_trace(day: int):
    env = MicroGridEnv(reward_mode='profit_safety')
    agent = QAgent.load_from_disk(env, CKPT)

    hours = np.arange(24)
    n = len(env.grid.buy_prices)

    buy = []
    sell = []
    soc = []
    ch_dch = []
    profits = []
    rewards = []

    state = env.reset(day=day)
    for h in range(24):
        idx = (day * 24 + h) % n
        buy.append(float(env.grid.buy_prices[idx]))
        sell.append(float(env.grid.sell_prices[idx]))

        action = agent.act(state, epsilon=0.0)
        state, reward, done, _ = env.step(action)
        soc.append(float(env.battery.SOC))
        ch_dch.append(float(env.battery.energy_change))
        profits.append(float(env.last_profit))
        rewards.append(float(reward))
        if done:
            break

    hours = hours[:len(soc)]
    buy = np.array(buy); sell = np.array(sell)
    soc = np.array(soc); ch_dch = np.array(ch_dch)
    profits = np.array(profits); rewards = np.array(rewards)
    cum_profit = np.cumsum(profits)

    peak_hour = int(np.argmax(sell))

    fig, axs = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

    # 1) Price
    axs[0].plot(hours, buy, color='dodgerblue', linewidth=2, label='Buy price')
    axs[0].plot(hours, sell, color='crimson', linewidth=2, label='Sell price')
    axs[0].axvline(peak_hour, color='black', linestyle='--', linewidth=1, alpha=0.6,
                   label=f'Peak sell hour {peak_hour}')
    axs[0].set_ylabel('Price')
    axs[0].set_title(f'Best-checkpoint greedy trace — day {day} (trained on Prices (3).csv)')
    axs[0].legend(frameon=False)
    axs[0].grid(True, linestyle='--', alpha=0.4)

    # 2) SOC
    axs[1].plot(hours, soc, color='seagreen', linewidth=2, marker='o', markersize=3)
    axs[1].axhline(0.7, color='tomato', linestyle='--', linewidth=1, alpha=0.7, label='Boundary high (0.7)')
    axs[1].axhline(0.3, color='tomato', linestyle='--', linewidth=1, alpha=0.7, label='Boundary low (0.3)')
    axs[1].axhline(0.8, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Hard max (0.8)')
    axs[1].axhline(0.2, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Hard min (0.2)')
    axs[1].set_ylabel('SOC')
    axs[1].set_ylim(0.1, 0.9)
    axs[1].legend(frameon=False, fontsize=8)
    axs[1].grid(True, linestyle='--', alpha=0.4)

    # 3) Charge / discharge
    charge = np.clip(ch_dch, 0, None)
    discharge = np.clip(ch_dch, None, 0)
    axs[2].bar(hours, charge, color='seagreen', label='Charge')
    axs[2].bar(hours, discharge, color='tomato', label='Discharge')
    axs[2].axhline(0.0, color='black', linewidth=0.8)
    axs[2].set_ylabel('Power (kW)')
    axs[2].legend(frameon=False)
    axs[2].grid(True, linestyle='--', alpha=0.4)

    # 4) Cumulative profit
    axs[3].plot(hours, cum_profit, color='purple', linewidth=2, marker='o', markersize=3)
    axs[3].axhline(0.0, color='black', linewidth=0.8)
    axs[3].set_ylabel('Cumulative profit')
    axs[3].set_xlabel('Hour')
    axs[3].grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()

    out_path = Path('results') / f'trace_day{day}_best.png'
    fig.savefig(str(out_path), dpi=200)
    print(f'Trace saved to {out_path}')

    print(f'--- Summary day {day} ---')
    print(f'Total profit     : {profits.sum():.2f}')
    print(f'Peak SOC         : {soc.max():.3f} | Min SOC: {soc.min():.3f}')
    print(f'Peak violation   : {max(0.0, soc.max() - 0.7) / 0.1: .2f} (if >0, broke boundary high)')
    print(f'Charge hours     : {[int(h) for h in hours[ch_dch > 0]]}')
    print(f'Discharge hours  : {[int(h) for h in hours[ch_dch < 0]]}')
    print(f'Peak sell hour   : {peak_hour}')

    return profits, soc, ch_dch, buy, sell


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=int, default=DEFAULT_DAY,
                        help='day index (0-364) to trace')
    parser.add_argument('--ckpt', type=str, default=str(CKPT))
    args = parser.parse_args()
    CKPT = Path(args.ckpt)
    run_trace(args.day)
