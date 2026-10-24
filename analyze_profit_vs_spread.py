from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from src.mg_env import MicroGridEnv, DEFAULT_DAY0_TRAIN, DEFAULT_DAYN_TRAIN
from src.q_agent import QAgent

env = MicroGridEnv(reward_mode='profit_safety')
agent = QAgent.load_from_disk(env, Path('results') / 'best_checkpoint')
buy_p = env.grid.buy_prices
sell_p = env.grid.sell_prices
n = len(buy_p)

def one_day(day):
    state = env.reset(day=day)
    profit, sell_at_disch, sell_at_charge = 0.0, [], []
    for h in range(24):
        idx = (day * 24 + h) % n
        a = agent.act(state, epsilon=0.0)
        state, r, done, _ = env.step(a)
        ec = float(env.battery.energy_change)
        profit += float(env.last_profit)
        if ec > 0:
            sell_at_charge.append(float(sell_p[idx]))
        elif ec < 0:
            sell_at_disch.append(float(sell_p[idx]))
        if done:
            break
    m_d = float(np.mean(sell_at_disch)) if sell_at_disch else float('nan')
    m_c = float(np.mean(sell_at_charge)) if sell_at_charge else float('nan')
    return profit, m_d, m_c

days = list(range(DEFAULT_DAY0_TRAIN, DEFAULT_DAYN_TRAIN + 1))
profits, buy_spreads, sell_spreads, buy_stds, sell_stds, deltas = [], [], [], [], [], []
for day in days:
    s0, s24 = day * 24, (day + 1) * 24
    bd = buy_p[s0:s24]
    sd = sell_p[s0:s24]
    p, md, mc = one_day(day)
    profits.append(p)
    buy_spreads.append(float(np.ptp(bd)))
    sell_spreads.append(float(np.ptp(sd)))
    buy_stds.append(float(np.std(bd)))
    sell_stds.append(float(np.std(sd)))
    deltas.append((md - mc) if not (np.isnan(md) or np.isnan(mc)) else 0.0)

profits = np.asarray(profits)
buy_spreads = np.asarray(buy_spreads)
sell_spreads = np.asarray(sell_spreads)
buy_stds = np.asarray(buy_stds)
sell_stds = np.asarray(sell_stds)
deltas = np.asarray(deltas)

mask = ~np.isnan(buy_spreads) & ~np.isnan(profits)
r_buy, p_buy = stats.pearsonr(buy_spreads[mask], profits[mask])
r_sell, p_sell = stats.pearsonr(sell_spreads[mask], profits[mask])
r_std, p_std = stats.pearsonr(sell_stds[mask], profits[mask])

n_pos = int(np.sum(np.array(deltas) > 0))
print(f"Days swept: {len(days)} ({DEFAULT_DAY0_TRAIN}-{DEFAULT_DAYN_TRAIN})")
print(f"Mean per-day profit : {profits.mean():.2f}   median: {np.median(profits):.2f}")
print(f"Days with profit>0  : {int(np.sum(profits > 0))}/{len(days)}")
print(f"Good timing (delta>0): {n_pos}/{len(days)}  (mean delta {deltas.mean():.3f})")
print(f"corr profit vs buy spread : r={r_buy:.3f}  p={p_buy:.3e}")
print(f"corr profit vs sell spread: r={r_sell:.3f}  p={p_sell:.3e}")
print(f"corr profit vs sell std   : r={r_std:.3f}  p={p_std:.3e}")

fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, x, xlab, r, p in [
    (axs[0], buy_spreads, "Daily buy-price spread (max-min)", r_buy, p_buy),
    (axs[1], sell_spreads, "Daily sell-price spread (max-min)", r_sell, p_sell),
    (axs[2], sell_stds, "Daily sell-price std", r_std, p_std),
]:
    ax.scatter(x, profits, s=28, alpha=0.75, edgecolor='k', linewidth=0.4)
    if r is not None:
        xline = np.linspace(x.min(), x.max(), 50)
        slope, intercept, *_ = np.polyfit(x, profits, 1)
        ax.plot(xline, slope * xline + intercept, color='tab:red', linewidth=1.5)
        ax.set_title(f"r = {r:.3f}  (p = {p:.2e})")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Per-day profit")

fig.suptitle("Per-day profit vs daily price volatility (full train eval range)", fontsize=13)
fig.tight_layout()
out = Path('results') / 'profit_vs_spread.png'
fig.savefig(str(out), dpi=220)
print(f"Plot saved to {out}")

np.savez(Path('results') / 'profit_vs_spread.npz',
         day=np.asarray(days), profit=profits, buy_spread=buy_spreads,
         sell_spread=sell_spreads, sell_std=sell_stds, delta=deltas)
print("Data saved to results/profit_vs_spread.npz")
