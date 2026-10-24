import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys; sys.path.append('BMS_Hyperpar_search-main'); from src.mg_env import MicroGridEnv

# If import path fails, try relative import fallback
try:
    import sys; sys.path.append('BMS_Hyperpar_search-main'); from src.mg_env import MicroGridEnv
except Exception:
    try:
        from src.mg_env import MicroGridEnv
    except Exception:
        raise

total_episodes = 300
window = 10

env = MicroGridEnv()
env.seedy(1)

ep_rewards = []
ep_costs = []

for ep in range(1, total_episodes + 1):
    state = env.reset()
    ep_reward = 0.0
    ep_cost = 0.0
    while True:
        action = env.action_space.sample()
        state, reward, terminal, _ = env.step(action)
        ep_reward += float(reward)
        # env.operation_cost is set per step in env.step
        ep_cost += float(env.operation_cost)
        env.render()
        if terminal:
            break
    ep_rewards.append(ep_reward)
    ep_costs.append(ep_cost)

# cumulative sums
cum_rewards = np.cumsum(ep_rewards)
cum_costs = np.cumsum(ep_costs)

# moving averages
if len(cum_rewards) >= window:
    ma_rewards = np.convolve(cum_rewards, np.ones(window)/window, mode='valid')
    ma_x_r = np.arange(window-1, len(cum_rewards)) + 1
else:
    ma_rewards = None

if len(cum_costs) >= window:
    ma_costs = np.convolve(cum_costs, np.ones(window)/window, mode='valid')
    ma_x_c = np.arange(window-1, len(cum_costs)) + 1
else:
    ma_costs = None

fig, axs = plt.subplots(2,1, figsize=(10,8), sharex=True)

# Top: cumulative reward
axs[0].plot(range(1, total_episodes+1), cum_rewards, color='skyblue', alpha=0.6, linewidth=0.8)
axs[0].scatter(range(1, total_episodes+1), cum_rewards, color='skyblue', s=8, alpha=0.6)
if ma_rewards is not None:
    axs[0].plot(ma_x_r, ma_rewards, color='tomato', linewidth=2.5, label='Moving average')
axs[0].set_ylabel('Cumulative Reward')
axs[0].grid(alpha=0.3)
axs[0].legend()

# Bottom: cumulative cost
axs[1].plot(range(1, total_episodes+1), cum_costs, color='skyblue', alpha=0.6, linewidth=0.8)
axs[1].scatter(range(1, total_episodes+1), cum_costs, color='skyblue', s=8, alpha=0.6)
if ma_costs is not None:
    axs[1].plot(ma_x_c, ma_costs, color='orange', linewidth=2.5, label='Moving average')
axs[1].set_ylabel('Cumulative Cost')
axs[1].set_xlabel('Episode')
axs[1].grid(alpha=0.3)
axs[1].legend()

plt.tight_layout()
out_path = 'BMS_Hyperpar_search-main/results/reward_cost_profit_energy_soc_300eps.png'
plt.savefig(out_path, dpi=200)
print('Saved', out_path)
