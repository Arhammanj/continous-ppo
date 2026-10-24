"""7-day greedy trace of the best checkpoint on HELD-OUT testing days.

Each panel = one day (24h): charge/discharge bars + sell-price overlay,
so timing around the real price peak is visible. Days are the first 7
held-out days (301-307); training used days 0-300.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.mg_env import MicroGridEnv
from src.q_agent import QAgent

CKPT = Path('results') / 'best_checkpoint'
TEST_DAYS = [301, 302, 303, 304, 305, 306, 307]


def trace_day(env, agent, day: int):
    n = len(env.grid.buy_prices)
    hours, sell, ch_dch, profits, soc = [], [], [], [], []
    np.random.seed(0)
    state = env.reset(day=day)
    for h in range(24):
        idx = (day * 24 + h) % n
        hours.append(h)
        sell.append(float(env.grid.sell_prices[idx]))
        action = agent.act(state, epsilon=0.0)
        state, reward, done, _ = env.step(action)
        ch_dch.append(float(env.battery.energy_change))
        profits.append(float(env.last_profit))
        soc.append(float(env.battery.SOC))
        if done:
            break
    return (np.array(hours), np.array(sell), np.array(ch_dch),
            np.array(profits), np.array(soc))


env = MicroGridEnv(reward_mode='pure_profit')
agent = QAgent.load_from_disk(env, CKPT)

fig, axs = plt.subplots(len(TEST_DAYS), 1, figsize=(11, 2.6 * len(TEST_DAYS)),
                        sharex=True)
for ax, day in zip(axs, TEST_DAYS):
    hours, sell, ch_dch, profits, soc = trace_day(env, agent, day)

    charge = np.clip(ch_dch, 0, None)
    discharge = np.clip(ch_dch, None, 0)
    ax.bar(hours, charge, color='seagreen', label='Charge')
    ax.bar(hours, discharge, color='tomato', label='Discharge')
    ax.axhline(0.0, color='black', linewidth=0.8)

    ax2 = ax.twinx()
    ax2.plot(hours, sell, color='steelblue', linestyle='--', linewidth=1.4,
             marker='o', markersize=2.5, label='Sell price')
    ax2.set_ylabel('Price', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.set_ylim(0, None)

    peak_hour = int(np.argmax(sell))
    ax.axvline(peak_hour, color='black', linestyle=':', linewidth=1, alpha=0.6)
    ax.text(peak_hour + 0.2, ax.get_ylim()[1] * 0.85, f'peak {peak_hour}h',
            fontsize=8, color='black')

    ax.set_ylabel('Power (kW)')
    ax.set_title(f'Day {day} (held-out test) — total profit {profits.sum():+.1f} | '
                 f'SOC {soc.min():.2f}–{soc.max():.2f} | charge {int((charge > 0).sum())}h, '
                 f'discharge {int((discharge < 0).sum())}h')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(frameon=False, loc='upper left', fontsize=8)

axs[-1].set_xlabel('Hour')
axs[0].legend(frameon=False, loc='upper left', fontsize=8)
fig.suptitle('7 held-out test days — battery charge/discharge vs price (best checkpoint, pure_profit + shield)',
             fontsize=13)
fig.tight_layout()

out_path = Path('results') / 'trace_test7days.png'
fig.savefig(str(out_path), dpi=200)
print(f'Plot saved to {out_path}')

print('\n--- Summary (test days) ---')
for day in TEST_DAYS:
    hours, sell, ch_dch, profits, soc = trace_day(env, agent, day)
    peak = int(np.argmax(sell))
    ch_h = [int(h) for h in hours[ch_dch > 0]]
    ds_h = [int(h) for h in hours[ch_dch < 0]]
    print(f'day {day}: profit {profits.sum():+8.2f} | peak sell {peak}h | '
          f'charge@{ch_h} | discharge@{ds_h}')
