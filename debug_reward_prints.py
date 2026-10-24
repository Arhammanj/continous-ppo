import random
import numpy as np
import src.mg_env as mg

# enable verbose prints inside mg_env
mg.VERBOSE = True

env = mg.MicroGridEnv()
env.seedy(1)

# run a few episodes and steps to capture reward prints
for ep in range(5):
    state = env.reset()
    while True:
        action = env.action_space.sample()
        state, reward, terminal, _ = env.step(action)
        if terminal:
            break

print('Done')
