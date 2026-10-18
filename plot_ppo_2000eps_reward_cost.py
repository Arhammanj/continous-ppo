import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('BMS_Hyperpar_search-main')
CSV_PATH = BASE / 'results' / 'v1new_ppo' / 'profit_safety_2000eps.csv'
LOG_PATH = BASE / 'results' / 'v1new_ppo' / 'train_2000eps.log'
OUT_PATH = BASE / 'ppo_2000eps_reward_cost.png'

window = 50

df = pd.read_csv(CSV_PATH)
episodes = df['episode'].to_numpy()
rewards = df['reward'].to_numpy(dtype=np.float64)
costs = df['operation_cost'].to_numpy(dtype=np.float64)

reward_ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
cost_ma = np.convolve(costs, np.ones(window) / window, mode='valid')
ma_x = np.arange(window, len(rewards) + 1)

eval_eps, eval_rs = [], []
if LOG_PATH.exists():
    raw = LOG_PATH.read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        text = raw.decode('utf-16')
    else:
        text = raw.decode('utf-8', errors='ignore')
    m_eps = re.search(r'Greedy eval checkpoints:\s*\[([^\]]*)\]', text)
    m_rs = re.search(r'Greedy eval rewards\s*:\s*\[([^\]]*)\]', text)
    if m_eps and m_rs:
        eval_eps = [int(x) for x in m_eps.group(1).split(',') if x.strip()]
        eval_rs = [float(x) for x in m_rs.group(1).split(',') if x.strip()]

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

fig.suptitle("Training data source: src/Prices.csv", fontsize=12, fontweight='bold')

axs[0].plot(episodes, rewards, color='deepskyblue', alpha=0.45, linewidth=2)
axs[0].plot(ma_x, reward_ma, color='salmon', linewidth=3, label='Moving average')
if eval_eps:
    axs[0].plot(eval_eps, eval_rs, marker='o', markersize=5, color='black',
                linewidth=2, label='Greedy eval (deterministic)')
axs[0].legend(loc='upper left', frameon=False)
axs[0].set_ylabel('Cumulative Reward')
axs[0].grid(True, alpha=0.25)

axs[1].plot(episodes, costs, color='deepskyblue', alpha=0.45, linewidth=2)
axs[1].plot(ma_x, cost_ma, color='orange', linewidth=3, label='Moving average')
axs[1].legend(loc='upper right', frameon=False)
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Cumulative Cost')
axs[1].grid(True, alpha=0.25)

fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=220)
print('Saved', OUT_PATH.resolve())
print(f'Episodes: {len(rewards)} | eval points: {len(eval_eps)}')
