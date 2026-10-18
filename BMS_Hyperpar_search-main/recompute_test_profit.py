"""Recompute energy-trading profit from RECORDED testing data only.

Reads results/eval_hourly_300_364.csv (produced by evaluate_test_days.py for the
DQN best checkpoint on held-out test days 300-364). No experiments are rerun.

Common profit function applied identically to every recorded hour:

    Profit = sum( SellPrice * P_dch - BuyPrice * P_ch )

where
    P_ch  = max(energy_change, 0)   (battery charging -> power imported/bought)
    P_dch = max(-energy_change, 0)  (battery discharging -> power exported/sold)

Outputs a per-day CSV under results/ and prints a summary table.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HOUR_CSV = Path('results') / 'eval_hourly_300_364.csv'
OUT_CSV = Path('results') / 'recomputed_test_profit_daily.csv'


def compute_profit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['P_ch'] = df['energy_change'].clip(lower=0.0)     # charging / buying  [kW]
    df['P_dch'] = (-df['energy_change']).clip(lower=0.0) # discharging / selling [kW]
    # Common profit function: SellPrice * P_dch - BuyPrice * P_ch
    df['profit'] = df['sell_price'] * df['P_dch'] - df['buy_price'] * df['P_ch']
    df['buy_cost'] = df['buy_price'] * df['P_ch']
    df['sell_benefit'] = df['sell_price'] * df['P_dch']
    return df


def main():
    df = pd.read_csv(HOUR_CSV)
    df = compute_profit(df)

    # Sanity check against the recorded trading_profit column
    recorded = df['trading_profit'].sum()
    recomputed = df['profit'].sum()
    if abs(recorded - recomputed) > 1e-6:
        print(f'WARNING: recomputed profit {recomputed:.4f} != recorded '
              f'trading_profit {recorded:.4f}')
    else:
        print(f'OK: recomputed profit == recorded trading_profit '
              f'({recomputed:.4f})')

    daily = df.groupby('day').agg(
        P_ch_sum=('P_ch', 'sum'),
        P_dch_sum=('P_dch', 'sum'),
        buy_cost=('buy_cost', 'sum'),
        sell_benefit=('sell_benefit', 'sum'),
        profit=('profit', 'sum'),
    ).reset_index()
    daily['net_profit'] = daily['profit'] - df.groupby('day')['battery_cost'].sum().values

    daily.to_csv(OUT_CSV, index=False)

    total = daily.sum(numeric_only=True)
    n_days = len(daily)
    days = daily['day'].values

    print('=' * 68)
    print('Energy trading profit - DQN best checkpoint')
    print(f'Recorded test window: days {days[0]}..{days[-1]} '
          f'({n_days} days, {len(df)} hourly steps)')
    print('Profit = sum(SellPrice * P_dch - BuyPrice * P_ch)')
    print('=' * 68)
    print(f'Total energy charged (P_ch)   : {total["P_ch_sum"]:12.1f} kWh')
    print(f'Total energy discharged (P_dch): {total["P_dch_sum"]:12.1f} kWh')
    print(f'Total buy cost                : {total["buy_cost"]:12.2f} '
          f'(BuyPrice * P_ch)')
    print(f'Total sell benefit            : {total["sell_benefit"]:12.2f} '
          f'(SellPrice * P_dch)')
    print('-' * 68)
    print(f'TOTAL TRADING PROFIT          : {total["profit"]:12.2f}')
    print(f'Mean daily profit             : {total["profit"] / n_days:12.2f} +/- '
          f'{daily["profit"].std():.2f}')
    print(f'Best day                      : day {int(daily.loc[daily["profit"].idxmax(), "day"])} '
          f'({daily["profit"].max():.2f})')
    print(f'Worst day                     : day {int(daily.loc[daily["profit"].idxmin(), "day"])} '
          f'({daily["profit"].min():.2f})')
    print('-' * 68)
    print(f'Saved per-day CSV             : {OUT_CSV}')


if __name__ == '__main__':
    main()
