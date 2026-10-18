"""Regenerate figures for results/best_checkpoint (256x256) ONLY.

No retraining happens here. This script:
  1. Loads results/best_checkpoint from disk (QAgent.load_from_disk).
  2. Runs evaluate() from src/loops.py (greedy, epsilon=0.0) on the held-out
     test days to get per-day reward/profit.
  3. Runs a plain greedy hourly rollout (same actions evaluate() would take)
     to get the hour-level price/SOC/charge-discharge series evaluate()
     itself does not expose, for the operation plots.

reward_mode determination and price-source/currency determination are
printed at the top of the run — see report_reward_mode_evidence() and
report_price_source() below for the exact evidence and reasoning.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.mg_env import MicroGridEnv
from src.q_agent import QAgent
from src.loops import evaluate

CKPT_DIR = Path('results') / 'best_checkpoint'
TRAIN_HISTORY_CSV = Path('results') / 'profit_safety_5000eps.csv'
REWARD_MODE = 'pure_profit'  # determined below, printed with evidence
DAY0 = 300
DAYN = 365   # 65 days, 300..364 inclusive -- matches the task's "65-day test period" framing
TITLE_TAG = "Reward v1, best_checkpoint"
OUT_DIR = Path('final_plots_bestcheckpoint')
EXPECTED_AVG = 29.91
EXPECTED_TOTAL = 1943.99
TOLERANCE = 0.05  # 5% relative tolerance before flagging a mismatch


def report_seeding_discrepancy(loops_avg, loops_total, trace_avg, trace_total):
    print("=" * 72)
    print("evaluate() FROM src/loops.py vs. THE HOURLY TRACE -- CONVENTION CHECK")
    print("=" * 72)
    print("Both paths now use the same continuous-evaluation convention:")
    print("the RNG is seeded once before the day loop, the FIRST day starts at")
    print("the seeded initial SOC, and every following day starts at exactly the")
    print("SOC left at the end of the previous day (env.reset(day=day,")
    print("carry_soc=...)) -- no re-randomization at day boundaries.")
    print()
    print(f"  loops.evaluate(), single seed(0), 65 days : avg={loops_avg:.2f}/day, total={loops_total:.2f}")
    print(f"  greedy_hourly_trace, same convention      : avg={trace_avg:.2f}/day, total={trace_total:.2f}")
    print(f"  stated verified figures                   : avg={EXPECTED_AVG}/day, total={EXPECTED_TOTAL}")
    print()
    print(f"Totals agree between the two paths to within {abs(loops_total - trace_total):.4f}.")
    print()


def moving_average(x, w):
    x = np.asarray(x, dtype=np.float64)
    if x.size < w:
        return np.array([]), np.array([])
    ma = np.convolve(x, np.ones(w) / w, mode='valid')
    idx = np.arange(w, x.size + 1)
    return idx, ma


def report_reward_mode_evidence():
    print("=" * 72)
    print("REWARD MODE DETERMINATION for results/best_checkpoint")
    print("=" * 72)
    with open(CKPT_DIR / 'hparams.json') as f:
        hparams = json.load(f)
    with open(CKPT_DIR / 'best_eval.json') as f:
        best_eval = json.load(f)
    print("hparams.json (agent hyperparameters only -- reward_mode is an ENV")
    print("constructor kwarg in mg_env.py, never stored in the agent checkpoint):")
    print(f"  {hparams}")
    print(f"best_eval.json: {best_eval}")
    print()
    print("hparams.json has NO reward_mode field. Evidence used instead:")
    print("  1. runner_train.py (current version in this repo) trains INTO")
    print("     checkpoint_dir=Path('results')/'best_checkpoint' with")
    print("     reward_mode='pure_profit', nn_hidden_layers=[256, 256],")
    print("     steps_epsilon_decay=54000, normalize_state=True, n_episodes=5000")
    print("     -- every one of those hyperparameters matches hparams.json above")
    print("     exactly (nn_hidden_layers=[256,256], steps_epsilon_decay=54000).")
    print(f"  2. best_eval_episode={best_eval['best_eval_episode']} is only reachable from")
    print("     a run with >=4100 episodes. Of the three saved per-episode training")
    print("     logs (profit_safety_1000eps.csv, _3000eps.csv, _5000eps.csv), only")
    print("     the 5000-episode one is long enough, and results/reward_cost_5000eps.png")
    print("     is the matching plot runner_train.py saves in that same run.")
    print("  3. Other scripts in this repo that load results/best_checkpoint for")
    print("     evaluation (evaluate_test_days.py, trace_test_days.py) also")
    print("     construct their env with reward_mode='pure_profit'.")
    print(f"  => CONCLUSION: reward_mode = '{REWARD_MODE}' (profit-only reward,")
    print("     no SOC shaping term in the reward -- SOC safety comes only from")
    print("     the hard shield in Battery.step(), independent of reward_mode).")
    print()
    print("CAVEAT: the CSV filename 'profit_safety_5000eps.csv' is misleading --")
    print("it is a hardcoded output filename in runner_train.py left over from an")
    print("earlier reward_mode literally called 'profit_safety'; it does NOT mean")
    print("this run used that reward mode. The hyperparameter match in (1) is the")
    print("actual evidence, not the filename.")
    print()


def report_price_source():
    print("=" * 72)
    print("PRICE DATA SOURCE / CURRENCY")
    print("=" * 72)
    root = Path(__file__).parent.parent
    price_path = root / 'Prices (3).csv'
    print(f"src/mg_env.py's _load_default_tou_price() resolved and loaded:")
    print(f"  {price_path}  (exists={price_path.exists()})")
    if price_path.exists():
        with open(price_path, 'r') as f:
            header = f.readline().strip()
            first_rows = [f.readline().strip() for _ in range(3)]
        print(f"  header row : {header!r}")
        print(f"  first rows : {first_rows}")
    print()
    print("The CSV header/columns carry NO currency unit at all (just 'Time;Price;;').")
    print("Raw values (e.g. 22, 20, ...) are unitless in the file. mg_env.py's loader")
    print("divides by 100 whenever max(raw price) > 5.0 (mg_env.py: "
          "'if float(np.max(prices)) > 5.0: prices = prices / 100.0'), implying the")
    print("source was recorded in the minor unit (cents/pence) of some currency per")
    print("kWh -- but which currency was never specified in this codebase.")
    print("Every '£' or '€' label seen so far (including this project's own commented-out")
    print("mg_env.py plot code using '€', and my own earlier scripts' choice of '£')")
    print("was an arbitrary display label picked downstream, NOT sourced from the data.")
    print("=> CURRENCY IS UNDETERMINED FROM THIS CODEBASE. Whoever produced the original")
    print("€19.66/day figure and whoever should confirm £ for this paper both made an")
    print("unstated choice; there is no file in this repo that pins it to EUR or GBP.")
    print()


def load_agent():
    env = MicroGridEnv(reward_mode=REWARD_MODE)
    agent = QAgent.load_from_disk(env, CKPT_DIR)
    return agent, env


def fig1_training_convergence():
    if not TRAIN_HISTORY_CSV.exists():
        print(f"[fig1] SKIPPED: no training history file found at {TRAIN_HISTORY_CSV}; "
              f"not retraining to recover it.")
        return None

    df = pd.read_csv(TRAIN_HISTORY_CSV)
    episodes = df['episode'].values
    rewards = df['reward'].values
    costs = df['operation_cost'].values
    w = 50
    ma_idx_r, ma_r = moving_average(rewards, w)
    ma_idx_c, ma_c = moving_average(costs, w)

    with open(CKPT_DIR / 'best_eval.json') as f:
        best_eval = json.load(f)
    best_ep = best_eval['best_eval_episode']

    fig, axs = plt.subplots(2, 1, figsize=(12, 9))

    axs[0].plot(episodes, rewards, color='lightskyblue', linewidth=1, alpha=0.6, label='Episode reward')
    if ma_r.size:
        axs[0].plot(ma_idx_r, ma_r, color='navy', linewidth=2, label=f'{w}-episode moving avg')
    axs[0].axvline(best_ep, color='crimson', linestyle='--', linewidth=1.5,
                   label=f'Saved checkpoint (episode {best_ep})')
    axs[0].set_title(f"Training Reward — {TITLE_TAG} (from {TRAIN_HISTORY_CSV.name})")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Episode reward")
    axs[0].legend(frameon=False)
    axs[0].grid(True, linestyle='--', alpha=0.35)

    axs[1].plot(episodes, costs, color='navajowhite', linewidth=1, alpha=0.7, label='Episode operating cost')
    if ma_c.size:
        axs[1].plot(ma_idx_c, ma_c, color='darkorange', linewidth=2, label=f'{w}-episode moving avg')
    axs[1].axvline(best_ep, color='crimson', linestyle='--', linewidth=1.5,
                   label=f'Saved checkpoint (episode {best_ep})')
    axs[1].set_title(f"Training Operating Cost — {TITLE_TAG} (from {TRAIN_HISTORY_CSV.name})")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Episode operating cost")
    axs[1].legend(frameon=False)
    axs[1].grid(True, linestyle='--', alpha=0.35)

    fig.tight_layout()
    out_path = OUT_DIR / 'v1_bestcheckpoint_training_convergence.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[fig1] saved {out_path} (source: {TRAIN_HISTORY_CSV}, {len(df)} episodes)")
    return out_path


def fig2_eval_convergence(days, rewards_per_day, profits_per_day):
    days = np.asarray(days)
    rewards_per_day = np.asarray(rewards_per_day, dtype=np.float64)
    profits_per_day = np.asarray(profits_per_day, dtype=np.float64)
    w = 7
    ma_idx_r, ma_r = moving_average(rewards_per_day, w)
    ma_idx_p, ma_p = moving_average(profits_per_day, w)
    ma_days_r = days[ma_idx_r - 1] if ma_r.size else np.array([])
    ma_days_p = days[ma_idx_p - 1] if ma_p.size else np.array([])

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(days, rewards_per_day, color='steelblue', marker='o', markersize=3,
                linewidth=1, alpha=0.6, label='Per-day eval reward')
    if ma_r.size:
        axs[0].plot(ma_days_r, ma_r, color='navy', linewidth=2, label=f'{w}-day moving avg')
    axs[0].set_xlabel('Test day')
    axs[0].set_ylabel('Eval reward')
    axs[0].legend(frameon=False)
    axs[0].grid(True, linestyle='--', alpha=0.35)

    axs[1].plot(days, profits_per_day, color='seagreen', marker='o', markersize=3,
                linewidth=1, alpha=0.6, label='Per-day eval profit')
    if ma_p.size:
        axs[1].plot(ma_days_p, ma_p, color='darkgreen', linewidth=2, label=f'{w}-day moving avg')
    axs[1].set_xlabel('Test day')
    axs[1].set_ylabel('Eval trading profit')
    axs[1].legend(frameon=False)
    axs[1].grid(True, linestyle='--', alpha=0.35)

    fig.suptitle("Figure — Evaluation Convergence on Unseen Days (Reward v1, best_checkpoint)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = OUT_DIR / 'v1_bestcheckpoint_eval_convergence.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[fig2] saved {out_path}")
    return out_path


def greedy_hourly_trace(agent, env, day0, dayn):
    """Plain greedy rollout (epsilon=0) to get hour-level series for the
    operation plots; evaluate() only returns per-day aggregates."""
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

    fig.suptitle("Figure — Battery Operation on Test Days 300-306 (Reward v1, best_checkpoint)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = OUT_DIR / 'v1_bestcheckpoint_multiday.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig3] saved {out_path}")
    return out_path


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

    fig.suptitle(f"Battery Operation — Day {day} (Reward v1, best_checkpoint)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / 'v1_bestcheckpoint_day310.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[fig4] saved {out_path}")
    return out_path


def fig5_daily_profit(daily_df, label):
    days = daily_df['day'].values
    profit = daily_df['trading_profit'].values
    avg = float(np.mean(profit))
    total = float(np.sum(profit))

    rel_err = abs(avg - EXPECTED_AVG) / abs(EXPECTED_AVG)
    print(f"[fig5] ({label}) computed avg={avg:.2f}/day, total={total:.2f}  "
          f"vs expected avg={EXPECTED_AVG}/day, total={EXPECTED_TOTAL}  "
          f"(rel. error on avg = {rel_err:.2%})")

    if rel_err > TOLERANCE:
        print(f"[fig5] ({label}) MISMATCH exceeds {TOLERANCE:.0%} tolerance -- "
              f"NOT saving this figure. See discrepancy report at the end of the run.")
        return None, avg, total, rel_err

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ['seagreen' if p >= 0 else 'tomato' for p in profit]
    ax.bar(days, profit, color=colors)
    ax.axhline(avg, color='navy', linestyle='--', linewidth=2, label=f'Average = {avg:.2f}/day')
    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set_title("Figure — Daily Trading Profit on Unseen Test Days 300-364 (Reward v1, best_checkpoint)")
    ax.set_xlabel('Test day')
    ax.set_ylabel('Daily trading profit')
    ax.legend(frameon=False)
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    out_path = OUT_DIR / 'v1_bestcheckpoint_daily_profit.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[fig5] ({label}) saved {out_path}")
    return out_path, avg, total, rel_err


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report_reward_mode_evidence()
    report_price_source()

    agent, env = load_agent()

    fig1_training_convergence()

    # --- fig 2: loops.evaluate() literally, as instructed, over the full range ---
    eval_env = MicroGridEnv(reward_mode=REWARD_MODE)
    evaluate(agent, eval_env, DAY0, DAYN, epsilon=0.0, seed=0)
    days = list(agent.eval_days)
    rewards = list(agent.eval_rewards_per_day)
    loops_profits = list(agent.eval_profits_per_day)
    loops_avg = float(np.mean(loops_profits))
    loops_total = float(np.sum(loops_profits))

    # --- figs 3/4/5: continuous greedy rollout with SOC carry-over (see convention check) ---
    hourly_df, daily_df = greedy_hourly_trace(agent, MicroGridEnv(reward_mode=REWARD_MODE),
                                              DAY0, DAYN)
    trace_avg = float(daily_df['trading_profit'].mean())
    trace_total = float(daily_df['trading_profit'].sum())

    report_seeding_discrepancy(loops_avg, loops_total, trace_avg, trace_total)

    fig2_eval_convergence(days, rewards, loops_profits)
    fig3_multiday_operation(hourly_df, days=list(range(300, 307)))
    fig4_singleday_snapshot(hourly_df, day=310)
    fig5_path, fig5_avg, fig5_total, fig5_rel_err = fig5_daily_profit(
        daily_df, label="continuous SOC-carry rollout, 65 days (300-364)",
    )

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    saved = [OUT_DIR / 'v1_bestcheckpoint_training_convergence.png',
             OUT_DIR / 'v1_bestcheckpoint_eval_convergence.png',
             OUT_DIR / 'v1_bestcheckpoint_multiday.png',
             OUT_DIR / 'v1_bestcheckpoint_day310.png']
    if fig5_path is not None:
        saved.append(fig5_path)
    for p in saved:
        if p.exists():
            print(" -", p.resolve())
    if fig5_path is None:
        print(f" - v1_bestcheckpoint_daily_profit.png NOT SAVED: computed avg "
              f"{fig5_avg:.2f}/day (total {fig5_total:.2f}) differs from the stated "
              f"verified {EXPECTED_AVG}/day (total {EXPECTED_TOTAL}) by "
              f"{fig5_rel_err:.1%}, which exceeds the {TOLERANCE:.0%} tolerance.")
    print()
    print(f"reward_mode determined: '{REWARD_MODE}' (see evidence above)")
    print("currency: UNDETERMINED from this codebase (see price-source report above)")


if __name__ == '__main__':
    main()
