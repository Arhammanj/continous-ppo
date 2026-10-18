"""Generate the 5 requested figures for Reward v1 (Stability-Enhanced).

Reward v1 = mg_env.MicroGridEnv with reward_mode="simple" (the environment
default), shielded Battery (hard SOC safety band [0.3, 0.7]).

This script only trains (if no v1 checkpoint exists yet) and evaluates using
the existing `train()` / `evaluate()` functions in src/loops.py — it does not
modify the reward function or environment logic.

Note on trading profit: `env.last_profit` is only populated by the
"profit_safety" / "pure_profit" reward branches in mg_env.py, not by
"simple". So for the profit-based figures (3, 4, 5) this script recomputes
trading profit itself from the same public quantities the environment
already exposes (grid prices + battery.energy_change), exactly like the
existing evaluate_test_days.py script does for its own reward mode.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.mg_env import MicroGridEnv, DEFAULT_DAY0_TRAIN, DEFAULT_DAYN_TRAIN
from src.q_agent import QAgent
from src.loops import train, evaluate
from src.utils import set_seed

REWARD_MODE = "simple"
N_EPISODES = 2000
DAY0_EVAL, DAYN_EVAL = 300, 365  # 65 test days: 300..364 inclusive
TITLE_TAG = "Reward v1 (Stability-Enhanced)"

ROOT = Path(__file__).parent
CKPT_DIR = ROOT / 'results' / 'best_checkpoint_v1_simple'
HIST_PATH = CKPT_DIR / 'training_history.npz'
OUT_DIR = ROOT / 'final_plots'


def moving_average(x, w):
    x = np.asarray(x, dtype=np.float64)
    if x.size < w:
        return np.array([]), np.array([])
    ma = np.convolve(x, np.ones(w) / w, mode='valid')
    idx = np.arange(w, x.size + 1)  # 1-based end-of-window position
    return idx, ma


def get_or_train_agent():
    """Reuse the v1 checkpoint + training curves if they already exist,
    otherwise train a fresh agent for N_EPISODES under reward_mode='simple'."""

    train_env = MicroGridEnv(
        day0=DEFAULT_DAY0_TRAIN, dayn=DEFAULT_DAYN_TRAIN, reward_mode=REWARD_MODE,
    )

    if (CKPT_DIR / 'model').exists() and HIST_PATH.exists():
        print(f"Found existing v1 checkpoint at {CKPT_DIR} — skipping training.")
        agent = QAgent.load_from_disk(train_env, CKPT_DIR)
        hist = np.load(HIST_PATH)
        return agent, train_env, hist['rewards'].tolist(), hist['costs'].tolist()

    print(f"No v1 checkpoint found at {CKPT_DIR} — training for {N_EPISODES} "
          f"episodes (reward_mode='{REWARD_MODE}')...")
    set_seed(train_env, 0)
    agent = QAgent(
        train_env, steps_epsilon_decay=36000, normalize_state=True,
        nn_hidden_layers=[256, 256],
    )

    def greedy_eval(agent, env, n_episodes=10, seed=0):
        set_seed(env, seed)
        ep_rewards = []
        for _ in range(n_episodes):
            state = env.reset()
            ep_r = 0.0
            done = False
            while not done:
                action = agent.act(state, epsilon=0.0)
                state, reward, done, _ = env.step(action)
                ep_r += float(reward)
            ep_rewards.append(ep_r)
        return float(np.mean(ep_rewards))

    rewards, costs = train(
        agent, train_env, n_episodes=N_EPISODES, log_dir=None,
        eval_fn=greedy_eval, freq_episodes_eval=100, checkpoint_dir=CKPT_DIR,
    )

    if not (CKPT_DIR / 'model').exists():
        # Greedy eval never improved on -inf (shouldn't happen), fall back to
        # persisting the final trained weights so a checkpoint always exists.
        agent.save_to_disk(CKPT_DIR)

    np.savez(HIST_PATH, rewards=np.array(rewards), costs=np.array(costs))

    return agent, train_env, rewards, costs


def trace_days(agent, env, day0, dayn):
    """Greedy hourly trace over [day0, dayn), recomputing trading profit
    (sell_benefit - buy_cost) independently of reward_mode."""
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

            rows.append({
                'day': day, 'hour': h,
                'buy_price': buy_price, 'sell_price': sell_price,
                'SOC': soc, 'energy_change': energy,
                'trading_profit': sell_benefit - buy_cost,
            })
            if done:
                break

    hourly = pd.DataFrame(rows)
    daily = hourly.groupby('day', as_index=False).agg(trading_profit=('trading_profit', 'sum'))
    return hourly, daily


def fig1_training_convergence(rewards, costs):
    rewards = np.asarray(rewards, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    episodes = np.arange(1, len(rewards) + 1)
    w = 50
    ma_idx_r, ma_r = moving_average(rewards, w)
    ma_idx_c, ma_c = moving_average(costs, w)

    fig, axs = plt.subplots(2, 1, figsize=(12, 9))

    axs[0].plot(episodes, rewards, color='lightskyblue', linewidth=1, alpha=0.6, label='Episode reward')
    if ma_r.size:
        axs[0].plot(ma_idx_r, ma_r, color='navy', linewidth=2, label=f'{w}-episode moving avg')
    axs[0].set_title(f"Training Reward — {TITLE_TAG}")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Episode reward")
    axs[0].legend(frameon=False)
    axs[0].grid(True, linestyle='--', alpha=0.35)

    axs[1].plot(episodes, costs, color='navajowhite', linewidth=1, alpha=0.7, label='Episode operating cost')
    if ma_c.size:
        axs[1].plot(ma_idx_c, ma_c, color='darkorange', linewidth=2, label=f'{w}-episode moving avg')
    axs[1].set_title(f"Training Operating Cost — {TITLE_TAG}")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Episode operating cost")
    axs[1].legend(frameon=False)
    axs[1].grid(True, linestyle='--', alpha=0.35)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'v1_training_convergence.png', dpi=200)
    plt.close(fig)


def fig2_eval_convergence(days, rewards_per_day, costs_per_day):
    days = np.asarray(days)
    rewards_per_day = np.asarray(rewards_per_day, dtype=np.float64)
    costs_per_day = np.asarray(costs_per_day, dtype=np.float64)
    w = 7
    ma_idx_r, ma_r = moving_average(rewards_per_day, w)
    ma_idx_c, ma_c = moving_average(costs_per_day, w)
    ma_days_r = days[ma_idx_r - 1] if ma_r.size else np.array([])
    ma_days_c = days[ma_idx_c - 1] if ma_c.size else np.array([])

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(days, rewards_per_day, color='steelblue', marker='o', markersize=3,
                linewidth=1, alpha=0.6, label='Per-day eval reward')
    if ma_r.size:
        axs[0].plot(ma_days_r, ma_r, color='navy', linewidth=2, label=f'{w}-day moving avg')
    axs[0].set_xlabel('Test day')
    axs[0].set_ylabel('Eval reward')
    axs[0].legend(frameon=False)
    axs[0].grid(True, linestyle='--', alpha=0.35)

    axs[1].plot(days, costs_per_day, color='darkorange', marker='o', markersize=3,
                linewidth=1, alpha=0.6, label='Per-day eval cost')
    if ma_c.size:
        axs[1].plot(ma_days_c, ma_c, color='chocolate', linewidth=2, label=f'{w}-day moving avg')
    axs[1].set_xlabel('Test day')
    axs[1].set_ylabel('Eval operating cost')
    axs[1].legend(frameon=False)
    axs[1].grid(True, linestyle='--', alpha=0.35)

    fig.suptitle("Figure — Evaluation Convergence on Unseen Days (Reward v1)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / 'v1_eval_convergence.png', dpi=200)
    plt.close(fig)


def fig3_multiday_operation(hourly_df, days):
    fig, axs = plt.subplots(3, len(days), figsize=(3.2 * len(days), 9), sharex=True)
    hours = np.arange(24)

    for col, day in enumerate(days):
        d = hourly_df[hourly_df['day'] == day].sort_values('hour')
        price = d['sell_price'].values
        soc = d['SOC'].values
        energy = d['energy_change'].values
        charge = np.clip(energy, 0, None)
        discharge = np.clip(energy, None, 0)
        h = hours[:len(price)]

        ax_price = axs[0, col]
        ax_price.plot(h, price, color='dodgerblue', linewidth=1.6)
        ax_price.set_title(f"Day {day}")
        if col == 0:
            ax_price.set_ylabel("Grid price")
        ax_price.grid(True, linestyle='--', alpha=0.35)

        ax_soc = axs[1, col]
        ax_soc.plot(h, soc, color='seagreen', linewidth=1.6, marker='o', markersize=2)
        ax_soc.axhline(0.8, color='black', linestyle=':', linewidth=1, alpha=0.6)
        ax_soc.axhline(0.2, color='black', linestyle=':', linewidth=1, alpha=0.6)
        ax_soc.axhline(0.7, color='tomato', linestyle='--', linewidth=1, alpha=0.7)
        ax_soc.axhline(0.3, color='tomato', linestyle='--', linewidth=1, alpha=0.7)
        ax_soc.axhline(0.6, color='gray', linestyle='-.', linewidth=1, alpha=0.7)
        ax_soc.set_ylim(0.1, 0.9)
        if col == 0:
            ax_soc.set_ylabel("SOC")
        ax_soc.grid(True, linestyle='--', alpha=0.35)

        ax_pow = axs[2, col]
        ax_pow.bar(h, charge, color='seagreen', width=0.8)
        ax_pow.bar(h[:len(discharge)], discharge, color='tomato', width=0.8)
        ax_pow.axhline(0.0, color='black', linewidth=0.8)
        ax_pow.set_xlabel("Hour")
        if col == 0:
            ax_pow.set_ylabel("Charge/Discharge")
        ax_pow.grid(True, linestyle='--', alpha=0.35)

    legend_elems = [
        Line2D([0], [0], color='black', linestyle=':', label='SOC hard max/min (0.8/0.2)'),
        Line2D([0], [0], color='tomato', linestyle='--', label='SOC boundary (0.7/0.3)'),
        Line2D([0], [0], color='gray', linestyle='-.', label='SOC target (0.6)'),
        Line2D([0], [0], color='seagreen', lw=6, label='Charge'),
        Line2D([0], [0], color='tomato', lw=6, label='Discharge'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Figure — Battery Operation on Test Days 300-306 (Reward v1)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUT_DIR / 'v1_multiday_operation.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig4_singleday_snapshot(hourly_df, day):
    d = hourly_df[hourly_df['day'] == day].sort_values('hour')
    hours = np.arange(len(d))
    buy = d['buy_price'].values
    sell = d['sell_price'].values
    soc = d['SOC'].values
    energy = d['energy_change'].values
    charge = np.clip(energy, 0, None)
    discharge = np.clip(energy, None, 0)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(hours, buy, color='dodgerblue', linewidth=2, label='Buy price')
    axs[0].plot(hours, sell, color='crimson', linewidth=2, label='Sell price')
    axs[0].set_ylabel('Price')
    axs[0].legend(frameon=False)
    axs[0].grid(True, linestyle='--', alpha=0.4)

    axs[1].plot(hours, soc, color='seagreen', linewidth=2, marker='o', markersize=3)
    axs[1].axhline(0.7, color='tomato', linestyle='--', linewidth=1, alpha=0.7, label='Boundary high (0.7)')
    axs[1].axhline(0.3, color='tomato', linestyle='--', linewidth=1, alpha=0.7, label='Boundary low (0.3)')
    axs[1].axhline(0.6, color='gray', linestyle='-.', linewidth=1, alpha=0.7, label='Target (0.6)')
    axs[1].axhline(0.8, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Hard max (0.8)')
    axs[1].axhline(0.2, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Hard min (0.2)')
    axs[1].set_ylabel('SOC')
    axs[1].set_ylim(0.1, 0.9)
    axs[1].legend(frameon=False, fontsize=8)
    axs[1].grid(True, linestyle='--', alpha=0.4)

    axs[2].bar(hours, charge, color='seagreen', label='Charge')
    axs[2].bar(hours, discharge, color='tomato', label='Discharge')
    axs[2].axhline(0.0, color='black', linewidth=0.8)
    axs[2].set_ylabel('Power (kW)')
    axs[2].set_xlabel('Hour')
    axs[2].legend(frameon=False)
    axs[2].grid(True, linestyle='--', alpha=0.4)

    fig.suptitle(f"Battery Operation — Day {day} (Reward v1)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / f'v1_singleday_day{day}.png', dpi=200)
    plt.close(fig)


def fig5_daily_profit(daily_df):
    days = daily_df['day'].values
    profit = daily_df['trading_profit'].values
    avg = float(np.mean(profit))

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ['seagreen' if p >= 0 else 'tomato' for p in profit]
    ax.bar(days, profit, color=colors)
    ax.axhline(avg, color='navy', linestyle='--', linewidth=2, label=f'Average = {avg:.2f} £/day')
    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set_title("Figure — Daily Trading Profit on Unseen Test Days 300-364 (Reward v1)")
    ax.set_xlabel('Test day')
    ax.set_ylabel('Daily trading profit (£)')
    ax.legend(frameon=False)
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'v1_daily_profit.png', dpi=200)
    plt.close(fig)
    return avg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    agent, base_env, train_rewards, train_costs = get_or_train_agent()

    # `train()` snapshots the agent to disk only when greedy eval improves, so
    # the in-memory agent may not be the best one seen. Reload from disk (if a
    # checkpoint was actually persisted) before running the test-day figures.
    if (CKPT_DIR / 'model').exists():
        eval_agent = QAgent.load_from_disk(base_env, CKPT_DIR)
    else:
        eval_agent = agent

    fig1_training_convergence(train_rewards, train_costs)

    eval_env = MicroGridEnv(reward_mode=REWARD_MODE)
    evaluate(eval_agent, eval_env, DAY0_EVAL, DAYN_EVAL, epsilon=0.0, seed=0)
    fig2_eval_convergence(
        eval_agent.eval_days, eval_agent.eval_rewards_per_day, eval_agent.eval_costs_per_day,
    )

    trace_env = MicroGridEnv(reward_mode=REWARD_MODE)
    hourly_df, daily_df = trace_days(eval_agent, trace_env, DAY0_EVAL, DAYN_EVAL)

    fig3_multiday_operation(hourly_df, days=list(range(300, 307)))
    fig4_singleday_snapshot(hourly_df, day=310)
    avg_profit = fig5_daily_profit(daily_df)

    saved = [
        OUT_DIR / 'v1_training_convergence.png',
        OUT_DIR / 'v1_eval_convergence.png',
        OUT_DIR / 'v1_multiday_operation.png',
        OUT_DIR / 'v1_singleday_day310.png',
        OUT_DIR / 'v1_daily_profit.png',
    ]

    print("=" * 68)
    print("Saved figures:")
    for p in saved:
        print(" -", p.resolve())
    print("-" * 68)
    print(f"Computed average daily trading profit over test days "
          f"{DAY0_EVAL}-{DAYN_EVAL - 1} ({len(daily_df)} days): {avg_profit:.2f} £/day")
    print(f"Total trading profit: {daily_df['trading_profit'].sum():.2f}")
    print(f"Best day  : day {int(daily_df.loc[daily_df['trading_profit'].idxmax(), 'day'])} "
          f"({daily_df['trading_profit'].max():.2f})")
    print(f"Worst day : day {int(daily_df.loc[daily_df['trading_profit'].idxmin(), 'day'])} "
          f"({daily_df['trading_profit'].min():.2f})")


if __name__ == '__main__':
    main()
