from pathlib import Path
from src.mg_env import MicroGridEnv, DEFAULT_DAY0_TRAIN, DEFAULT_DAYN_TRAIN
from src.q_agent import QAgent
from src.loops import train, plot_reward_cost_curves

# Set up environment and agent
env = MicroGridEnv(
	day0=DEFAULT_DAY0_TRAIN,
	dayn=DEFAULT_DAYN_TRAIN,
	reward_mode="simple",
)
agent = QAgent(env)

# Train
n_episodes = 1000
print(f"Training for {n_episodes} episodes...")
rewards, costs = train(agent, env, n_episodes=n_episodes, log_dir=None)

# Save plot
out_path = Path('results') / f'reward_cost_{n_episodes}eps.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
fig, axs = plot_reward_cost_curves(rewards, costs, moving_avg_window=50, save_path=str(out_path))
print(f"Plot saved to {out_path}")
