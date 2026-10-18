from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.mg_env import MicroGridEnv, DEFAULT_DAY0_TRAIN, DEFAULT_DAYN_TRAIN
from src.ppo_agent import PPOAgent
from src.ppo_loops import train_ppo
from src.loops import plot_reward_cost_curves
from src.utils import set_seed

# Set up environment and agent
env = MicroGridEnv(
	day0=DEFAULT_DAY0_TRAIN,
	dayn=DEFAULT_DAYN_TRAIN,
	reward_mode="pure_profit",
)

parser = ArgumentParser(description="Train the PPO agent on the microgrid environment.")
parser.add_argument('--episodes', type=int, default=5000, help='number of training episodes')
parser.add_argument('--eval-freq', type=int, default=100, help='greedy eval frequency (episodes)')
parser.add_argument('--rollout-steps', type=int, default=2048, help='env steps per PPO update')
args = parser.parse_args()

# PPO counterpart of the v1 DQN hyperparameter set:
# - nn_hidden_layers=[256, 256] (same architecture as the DQN v1 run)
# - normalize_state=True: running z-score of observations (DQN used frozen
#   statistics from a 10k-sample random rollout; PPO updates them online).
# - rollout_steps=2048 env steps per PPO update (~85 one-day episodes),
#   n_epochs=10 clipped-surrogate passes, minibatch 64, clip_range 0.2.
# - gamma 0.99, GAE lambda 0.95, Adam lr 3e-4, grad clip 0.5.
# - Continuous action: the policy outputs any battery setpoint in
#   [-80, +80] kW (tanh-squashed Gaussian) instead of one of the 18 discrete
#   levels; the Battery safety shield still clips SOC to the safe band.
# - reward_mode="pure_profit": reward is normalized profit only.
agent = PPOAgent(
    env,
    learning_rate=3e-4,
    discount_factor=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    n_epochs=10,
    batch_size=64,
    rollout_steps=args.rollout_steps,
    nn_hidden_layers=[256, 256],
    max_grad_norm=0.5,
    normalize_state=True,
    ent_coef=0.0,
    vf_coef=0.5,
    init_log_std=0.0,
)

# Greedy evaluation callback (deterministic mean action: act(..., epsilon=0)).
# Re-seeding each checkpoint means every eval evaluates the same set of days,
# so points are comparable.
def greedy_eval(agent, env, n_episodes=10, seed=0):
    set_seed(env, seed)
    ep_rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            action = agent.act(state, epsilon=0.0)
            next_state, reward, done, _ = env.step(action)
            ep_r += float(reward)
            state = next_state
        ep_rewards.append(ep_r)
    return float(np.mean(ep_rewards))

# Train
n_episodes = args.episodes
eval_freq = args.eval_freq
run_dir = Path('results') / 'v1new_ppo'
ckpt_dir = run_dir / 'best_checkpoint'
print(f"Training for {n_episodes} episodes with PPO (rollout {args.rollout_steps} steps, 10 epochs/update, greedy eval every {eval_freq} eps)...")
print(f"normalize_state = {agent.normalize_state} | hidden_layers = [256, 256] | action space = continuous [-80, 80] kW")
print(f"best-eval checkpoint dir = {ckpt_dir}")
rewards, costs = train_ppo(agent, env, n_episodes=n_episodes, log_dir=None,
                           eval_fn=greedy_eval, freq_episodes_eval=eval_freq,
                           checkpoint_dir=ckpt_dir)

# Save plot
out_path = run_dir / f'reward_cost_{n_episodes}eps.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
fig, axs = plot_reward_cost_curves(rewards, costs, moving_avg_window=50, save_path=str(out_path),
                                   data_label="src/Prices.csv")

# Overlay greedy eval line on the reward panel
if len(agent.eval_episodes) > 0:
    axs[0].plot(agent.eval_episodes, agent.eval_rewards, marker='o', markersize=5,
                color='black', linewidth=2, label='Greedy eval (deterministic)')
    axs[0].legend(loc='upper left', frameon=False)

fig.tight_layout()
fig.savefig(str(out_path), dpi=220)
print(f"Plot saved to {out_path}")
print(f"Greedy eval checkpoints: {agent.eval_episodes}")
print(f"Greedy eval rewards    : {[round(r, 4) for r in agent.eval_rewards]}")
print(f"Best greedy eval       : episode {agent.best_eval_episode}, reward={agent.best_eval_reward:.4f} -> saved in {ckpt_dir}")
print(f"PPO updates performed  : {agent.updates} | last update metrics: "
      f"policy_loss={agent.last_update_metrics.get('policy_loss', float('nan')):.5f}, "
      f"value_loss={agent.last_update_metrics.get('value_loss', float('nan')):.5f}, "
      f"policy_log_std={agent.last_update_metrics.get('policy_log_std', float('nan')):.3f}")

# Log per-episode profit / safety metrics (populated by the profit_safety reward mode)
if len(agent.profit_per_episode) > 0:
    profits = np.asarray(agent.profit_per_episode)
    viols = np.asarray(agent.safety_violation_per_episode)
    print(f"Profit per episode   : total={profits.sum():.2f} mean={profits.mean():.3f}")
    print(f"Safety violation peak: mean={viols.mean():.4f} max={viols.max():.4f}")

    # Save metrics to CSV for later reporting (thesis plots etc.)
    csv_path = run_dir / f'profit_safety_{n_episodes}eps.csv'
    with open(csv_path, 'w') as f:
        f.write("episode,reward,operation_cost,profit,safety_violation,avg_action_kw\n")
        for i, (r, c, p, v, a) in enumerate(zip(rewards, costs, agent.profit_per_episode,
                                                agent.safety_violation_per_episode,
                                                agent.avg_action_per_episode)):
            f.write(f"{i + 1},{r:.6f},{c:.6f},{p:.6f},{v:.6f},{a:.6f}\n")
    print(f"Metrics CSV saved to {csv_path}")
