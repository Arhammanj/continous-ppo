"""Regenerate 4 figures for Reward v1 (Pure-Profit, results/best_checkpoint 256x256)
under a fixed shared style spec, to match a parallel Reward v2 figure set.

No retraining, no reward/environment changes -- plotting only. Loads the existing
checkpoint and evaluates it on test days 300-364.

Methodology note: continuous evaluation across the whole 65-day window. The RNG
is seeded once before the day loop, the FIRST day starts at the seeded initial
SOC, and every following day starts at exactly the SOC left at the end of the
previous day (env.reset(day=day, carry_soc=...)) -- the battery is never
re-randomized at day boundaries. The rollout reads
env.operation_cost/env.step()'s reward directly instead of calling
env.render(mode='Eval') (which only prints debug text and opens throwaway
matplotlib figures -- it has no effect on any number reported here).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

from src.mg_env import MicroGridEnv
from src.q_agent import QAgent

CKPT_DIR = Path('results') / 'best_checkpoint'
TRAIN_HISTORY_CSV = Path('results') / 'profit_safety_5000eps.csv'
REWARD_MODE = 'pure_profit'
DAY0, DAYN = 300, 365  # 65 days, 300..364 inclusive
OUT_DIR = Path('final_plots_styled')

# --- shared style spec ---
GREEN = "#2E7D32"   # profit / charge / positive
RED = "#C62828"     # loss / discharge / negative
BLUE = "#1565C0"    # average / reference line
RAW = "#B0BEC5"     # neutral light gray for raw/noisy series (not specified by
                     # the spec; used only where the spec leaves a color undefined)

TITLE_FS = 14
LABEL_FS = 11
TICK_FS = 9
LEGEND_FS = 10

SINGLE_FIGSIZE = (10, 5)
STACKED_FIGSIZE = (10, 8)


def style_axes(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=LABEL_FS)
    ax.set_ylabel(ax.get_ylabel(), fontsize=LABEL_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)


def add_legend(ax):
    ax.legend(loc='upper right', frameon=True, fontsize=LEGEND_FS)


def rollout(agent, env, day0, dayn):
    """Continuous greedy rollout across [day0, dayn): the first day uses the
    seeded initial SOC and every following day starts at exactly the SOC left
    at the end of the previous day. Reads env.operation_cost / the reward
    returned by env.step() directly -- equivalent to loops.evaluate() but
    without calling env.render(mode='Eval'), which only prints debug text /
    opens unused matplotlib figures and has no effect on any of these numbers."""
    rows = []
    n = len(env.grid.buy_prices)
    np.random.seed(0)  # deterministic initial SOC for the FIRST day only
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
            rows.append({
                'day': day, 'hour': h,
                'buy_price': buy_price, 'sell_price': sell_price,
                'SOC': soc, 'energy_change': energy,
                'reward': float(reward),
                'trading_profit': sell_benefit - buy_cost,
            })
            if done:
                break
    hourly = pd.DataFrame(rows)
    daily = hourly.groupby('day', as_index=False).agg(
        reward=('reward', 'sum'),
        trading_profit=('trading_profit', 'sum'),
    )
    return hourly, daily


def fig1_daily_profit(daily_df):
    days = daily_df['day'].values
    profit = daily_df['trading_profit'].values
    avg = float(np.mean(profit))

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    colors = [GREEN if p >= 0 else RED for p in profit]
    ax.bar(days, profit, color=colors)
    ax.axhline(avg, color=BLUE, linestyle='--', linewidth=2,
               label=f'Average = £{avg:.2f}/day')
    ax.axhline(0.0, color='black', linewidth=0.8)

    ax.set_title("Fig. 8. Daily trading profit on unseen test days 300-364 — Reward v1 (Pure-Profit)",
                 fontsize=TITLE_FS, fontweight='bold')
    ax.set_xlabel('Test day')
    ax.set_ylabel('Daily trading profit (£)')
    style_axes(ax)
    add_legend(ax)

    fig.tight_layout()
    out_path = OUT_DIR / 'v1_daily_profit_STYLED.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[fig1] saved {out_path}  (avg=£{avg:.2f}/day, total=£{profit.sum():.2f})")
    return out_path


def fig2_day310(hourly_df, env, day=310):
    d = hourly_df[hourly_df['day'] == day].sort_values('hour')
    hours = np.arange(len(d))
    buy = d['buy_price'].values
    sell = d['sell_price'].values
    soc = d['SOC'].values
    energy = d['energy_change'].values
    charge = np.clip(energy, 0, None)
    discharge = np.clip(energy, None, 0)

    fig, axs = plt.subplots(3, 1, figsize=STACKED_FIGSIZE, sharex=True)

    axs[0].plot(hours, buy, color=BLUE, linewidth=2, label='Buy price')
    axs[0].plot(hours, sell, color="#424242", linewidth=2, label='Sell price')
    axs[0].set_ylabel('Price (£/kWh)')
    add_legend(axs[0])
    style_axes(axs[0])

    axs[1].plot(hours, soc, color=GREEN, linewidth=2, marker='o', markersize=3, label='SOC')
    axs[1].axhline(env.soc_max, color=BLUE, linestyle='--', linewidth=2, alpha=0.4,
                   label=f'Hard max ({env.soc_max:.1f})')
    axs[1].axhline(env.soc_min, color=BLUE, linestyle='--', linewidth=2, alpha=0.4,
                   label=f'Hard min ({env.soc_min:.1f})')
    axs[1].axhline(env.boundary_soc_high, color=BLUE, linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Boundary high ({env.boundary_soc_high:.1f})')
    axs[1].axhline(env.boundary_soc_low, color=BLUE, linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Boundary low ({env.boundary_soc_low:.1f})')
    axs[1].axhline(env.soc_target, color=BLUE, linestyle='--', linewidth=2, alpha=1.0,
                   label=f'Target ({env.soc_target:.1f})')
    axs[1].set_ylabel('SOC')
    axs[1].set_ylim(0.1, 0.9)
    add_legend(axs[1])
    style_axes(axs[1])

    axs[2].bar(hours, charge, color=GREEN, label='Charge')
    axs[2].bar(hours, discharge, color=RED, label='Discharge')
    axs[2].axhline(0.0, color='black', linewidth=0.8)
    axs[2].set_ylabel('Power (kW)')
    axs[2].set_xlabel('Hour')
    add_legend(axs[2])
    style_axes(axs[2])

    fig.suptitle(f"Fig. 5. Battery operation on Day {day} — Reward v1 (Pure-Profit)",
                 fontsize=TITLE_FS, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / 'v1_day310_STYLED.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[fig2] saved {out_path}")
    return out_path


def fig3_training_convergence():
    df = pd.read_csv(TRAIN_HISTORY_CSV)
    episodes = df['episode'].values
    rewards = df['reward'].values
    costs = df['operation_cost'].values

    w = 50
    def moving_average(x, w):
        if len(x) < w:
            return np.array([]), np.array([])
        ma = np.convolve(x, np.ones(w) / w, mode='valid')
        idx = np.arange(w, len(x) + 1)
        return idx, ma
    ma_idx_r, ma_r = moving_average(rewards, w)
    ma_idx_c, ma_c = moving_average(costs, w)

    with open(CKPT_DIR / 'best_eval.json') as f:
        best_ep = json.load(f)['best_eval_episode']

    fig, axs = plt.subplots(2, 1, figsize=STACKED_FIGSIZE)

    axs[0].plot(episodes, rewards, color=RAW, linewidth=1, alpha=0.8, label='Episode reward')
    axs[0].plot(ma_idx_r, ma_r, color=BLUE, linestyle='--', linewidth=2, label=f'{w}-episode moving avg')
    axs[0].axvline(best_ep, color=RED, linestyle='--', linewidth=2, label=f'Saved checkpoint (episode {best_ep})')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Episode reward')
    add_legend(axs[0])
    style_axes(axs[0])

    axs[1].plot(episodes, costs, color=RAW, linewidth=1, alpha=0.8, label='Episode operating cost')
    axs[1].plot(ma_idx_c, ma_c, color=BLUE, linestyle='--', linewidth=2, label=f'{w}-episode moving avg')
    axs[1].axvline(best_ep, color=RED, linestyle='--', linewidth=2, label=f'Saved checkpoint (episode {best_ep})')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Operating cost (£)')
    add_legend(axs[1])
    style_axes(axs[1])

    fig.suptitle("Fig. 1. Training reward and operating cost convergence — Reward v1 (Pure-Profit)",
                 fontsize=TITLE_FS, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / 'v1_train_convergence_STYLED.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[fig3] saved {out_path}  (source: {TRAIN_HISTORY_CSV}, {len(df)} episodes)")
    return out_path


def fig4_eval_convergence(daily_df):
    days = daily_df['day'].values
    reward = daily_df['reward'].values
    profit = daily_df['trading_profit'].values

    w = 7
    def moving_average(x, w):
        if len(x) < w:
            return np.array([]), np.array([])
        ma = np.convolve(x, np.ones(w) / w, mode='valid')
        idx = np.arange(w, len(x) + 1)
        return idx, ma
    ma_idx_r, ma_r = moving_average(reward, w)
    ma_idx_p, ma_p = moving_average(profit, w)
    ma_days_r = days[ma_idx_r - 1] if ma_r.size else np.array([])
    ma_days_p = days[ma_idx_p - 1] if ma_p.size else np.array([])

    fig, axs = plt.subplots(2, 1, figsize=STACKED_FIGSIZE, sharex=True)

    axs[0].plot(days, reward, color=RAW, linewidth=1, marker='o', markersize=3,
                alpha=0.9, label='Per-day eval reward')
    axs[0].plot(ma_days_r, ma_r, color=BLUE, linestyle='--', linewidth=2, label=f'{w}-day moving avg')
    axs[0].set_ylabel('Eval reward')
    add_legend(axs[0])
    style_axes(axs[0])

    profit_colors = [GREEN if p >= 0 else RED for p in profit]
    axs[1].scatter(days, profit, color=profit_colors, s=22, zorder=3, label='Per-day eval profit')
    axs[1].plot(ma_days_p, ma_p, color=BLUE, linestyle='--', linewidth=2, label=f'{w}-day moving avg')
    axs[1].axhline(0.0, color='black', linewidth=0.8)
    axs[1].set_xlabel('Test day')
    axs[1].set_ylabel('Eval profit (£)')
    add_legend(axs[1])
    style_axes(axs[1])

    fig.suptitle("Fig. 3. Evaluation reward and profit per test day — Reward v1 (Pure-Profit)",
                 fontsize=TITLE_FS, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / 'v1_eval_convergence_STYLED.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[fig4] saved {out_path}")
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = MicroGridEnv(reward_mode=REWARD_MODE)
    agent = QAgent.load_from_disk(env, CKPT_DIR)

    hourly_df, daily_df = rollout(agent, env, DAY0, DAYN)

    saved = []
    saved.append(fig3_training_convergence())
    saved.append(fig1_daily_profit(daily_df))
    saved.append(fig2_day310(hourly_df, env, day=310))
    saved.append(fig4_eval_convergence(daily_df))

    print("=" * 60)
    print("Saved figures:")
    for p in saved:
        print(" -", p.resolve())


if __name__ == '__main__':
    main()
