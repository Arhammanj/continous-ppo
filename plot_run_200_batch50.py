import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

log_path = 'run_200_batch50.log'
out_png = 'run_200_batch50.png'

episodes = []
re_episode = re.compile(r"Episode\s+(\d+):\s*total_reward=([\-0-9.eE]+)")
with open(log_path, 'r') as f:
    for line in f:
        m = re_episode.search(line)
        if m:
            idx = int(m.group(1))
            val = float(m.group(2))
            episodes.append((idx, val))

if not episodes:
    raise SystemExit('No episode data found in log')

# Sort by episode index and build arrays
episodes.sort()
idxs = [e[0] for e in episodes]
vals = [e[1] for e in episodes]

# Plot
plt.figure(figsize=(10,4))
plt.plot(idxs, vals, marker='o', linestyle='-', markersize=3)
plt.axhline(0, color='k', linewidth=0.5)
plt.title('Episode total rewards (200 eps)')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.grid(alpha=0.3)

# Also plot moving average (window 10)
window = 10
if len(vals) >= window:
    ma = np.convolve(vals, np.ones(window)/window, mode='valid')
    ma_x = idxs[window-1:]
    plt.plot(ma_x, ma, color='orange', linewidth=2, label=f'{window}-ep MA')
    plt.legend()

plt.tight_layout()
plt.savefig(out_png, dpi=200)
print('Saved', out_png)
