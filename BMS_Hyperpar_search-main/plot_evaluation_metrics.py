"""Plot evaluation metrics for the trained agent on held-out test days.

Reads results/eval_hourly_300_364.csv and results/eval_daily_300_364.csv
(produced by evaluate_test_days.py) and saves 4 plots into results/eval_plots/:
  1. daily_trading_profit.png      - daily trading profit (sell_benefit - buy_cost)
  2. cumulative_profit_cost.png    - cumulative trading profit and cumulative cost
  3. soc_heatmap.png               - SOC per hour per day (heatmap)
  4. daily_energy_bought_sold.png  - daily energy charged (bought) vs discharged (sold)
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path('results')
HOUR_CSV = RESULTS / 'eval_hourly_300_364.csv'
DAY_CSV = RESULTS / 'eval_daily_300_364.csv'
OUT_DIR = RESULTS / 'eval_plots'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hourly', type=str, default=str(HOUR_CSV))
    parser.add_argument('--daily', type=str, default=str(DAY_CSV))
    parser.add_argument('--out', type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.hourly)
    daily = pd.read_csv(args.daily)

    days = daily['day'].values
    n_days = len(days)
    total = daily.sum(numeric_only=True)

    # 1) Daily trading profit
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['seagreen' if p >= 0 else 'tomato' for p in daily['trading_profit']]
    ax.bar(days, daily['trading_profit'], color=colors)
    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set_title(f'Daily trading profit on test days {days[0]}..{days[-1]} '
                 f'(total {total["trading_profit"]:.2f})')
    ax.set_xlabel('Day')
    ax.set_ylabel('Trading profit (sell_benefit - buy_cost)')
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / 'daily_trading_profit.png', dpi=200)
    plt.close(fig)

    # 2) Cumulative profit and cost
    cum_profit = daily['trading_profit'].cumsum().values
    cum_buy = daily['buy_cost'].cumsum().values
    cum_sell = daily['sell_benefit'].cumsum().values
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(days, cum_profit, color='purple', linewidth=2.2, label='Cumulative trading profit')
    ax.plot(days, cum_sell, color='steelblue', linewidth=1.8, label='Cumulative sell benefit')
    ax.plot(days, cum_buy, color='tomato', linewidth=1.8, label='Cumulative buy cost')
    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set_title(f'Cumulative profit & cost over test days {days[0]}..{days[-1]}')
    ax.set_xlabel('Day')
    ax.set_ylabel('Cumulative value')
    ax.legend(frameon=False)
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / 'cumulative_profit_cost.png', dpi=200)
    plt.close(fig)

    # 3) SOC heatmap (day x hour)
    soc = df.pivot_table(index='day', columns='hour', values='SOC')
    soc = soc.reindex(columns=np.arange(24))
    fig, ax = plt.subplots(figsize=(13, max(4, n_days * 0.25)))
    im = ax.imshow(soc.values, aspect='auto', cmap='YlGn', vmin=0.2, vmax=0.8,
                   extent=[-0.5, 23.5, days[-1] + 0.5, days[0] - 0.5])
    ax.set_title(f'SOC per hour per day on test days {days[0]}..{days[-1]}')
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Day')
    ax.set_yticks(days[::5])
    ax.set_xticks(np.arange(0, 24, 2))
    fig.colorbar(im, ax=ax, label='SOC')
    fig.tight_layout()
    fig.savefig(out_dir / 'soc_heatmap.png', dpi=200)
    plt.close(fig)

    # SOC daily range plot
    daily_soc_min = df.groupby('day')['SOC'].min()
    daily_soc_max = df.groupby('day')['SOC'].max()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(days, daily_soc_min.values, daily_soc_max.values,
                    color='seagreen', alpha=0.4, label='SOC range (min-max)')
    ax.plot(days, daily_soc_max.values, color='darkgreen', linewidth=1, label='Daily max SOC')
    ax.plot(days, daily_soc_min.values, color='seagreen', linewidth=1, label='Daily min SOC')
    ax.axhline(0.7, color='tomato', linestyle='--', linewidth=1, alpha=0.8, label='Boundary high 0.7')
    ax.axhline(0.3, color='tomato', linestyle='--', linewidth=1, alpha=0.8, label='Boundary low 0.3')
    ax.set_ylim(0.1, 0.9)
    ax.set_title(f'SOC daily range on test days {days[0]}..{days[-1]}')
    ax.set_xlabel('Day')
    ax.set_ylabel('SOC')
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / 'soc_daily_range.png', dpi=200)
    plt.close(fig)

    # 4) Daily energy bought (charged) and sold (discharged)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_days)
    w = 0.4
    ax.bar(x - w / 2, daily['energy_charged'].values, w, color='steelblue',
           label='Energy bought (charged)')
    ax.bar(x + w / 2, daily['energy_discharged'].values, w, color='gold',
           label='Energy sold (discharged)')
    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set_xticks(x[::5])
    ax.set_xticklabels(days[::5])
    ax.set_title(f'Daily energy bought & sold on test days {days[0]}..{days[-1]} '
                 f'(bought {total["energy_charged"]:.0f} kWh, sold {total["energy_discharged"]:.0f} kWh)')
    ax.set_xlabel('Day')
    ax.set_ylabel('Energy (kWh)')
    ax.legend(frameon=False)
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / 'daily_energy_bought_sold.png', dpi=200)
    plt.close(fig)

    print(f'Plots saved to {out_dir.resolve()}')
    for p in sorted(out_dir.glob('*.png')):
        print(' -', p.name)


if __name__ == '__main__':
    main()
