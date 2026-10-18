from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('BMS_Hyperpar_search-main')
CSV_PATH = BASE / 'results' / 'v1new_ppo' / 'profit_safety_2000eps.csv'
OUT_PATH = BASE / 'ppo_2000eps_cumulative_reward_cost.png'

df = pd.read_csv(CSV_PATH)
episodes = df['episode'].to_numpy()
rewards = df['reward'].to_numpy(dtype=np.float64)
costs = df['operation_cost'].to_numpy(dtype=np.float64)

cum_rewards = np.cumsum(rewards)
cum_costs = np.cumsum(costs)

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

fig.suptitle("PPO training cumulative reward and cost (2000 episodes, src/Prices.csv)",
             fontsize=12, fontweight='bold')

axs[0].plot(episodes, cum_rewards, color='seagreen', linewidth=2)
axs[0].fill_between(episodes, cum_rewards, alpha=0.15, color='seagreen')
axs[0].axhline(cum_rewards[-1], color='black', linestyle=':', linewidth=1)
axs[0].set_ylabel('Cumulative Reward')
axs[0].annotate(f'total = {cum_rewards[-1]:,.1f}',
                xy=(episodes[-1], cum_rewards[-1]),
                xytext=(-120, 0), textcoords='offset points',
                va='bottom', fontsize=9)
axs[0].grid(True, alpha=0.25)

axs[1].plot(episodes, cum_costs, color='darkorange', linewidth=2)
axs[1].fill_between(episodes, cum_costs, alpha=0.15, color='darkorange')
axs[1].axhline(cum_costs[-1], color='black', linestyle=':', linewidth=1)
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Cumulative Cost')
axs[1].annotate(f'total = {cum_costs[-1]:,.1f}',
                xy=(episodes[-1], cum_costs[-1]),
                xytext=(-120, 0), textcoords='offset points',
                va='bottom', fontsize=9)
axs[1].grid(True, alpha=0.25)

fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=220)
print('Saved', OUT_PATH.resolve())
print(f'Episodes: {len(rewards)}')
print(f'Cumulative reward total: {cum_rewards[-1]:.2f}')
print(f'Cumulative cost total  : {cum_costs[-1]:.2f}')
