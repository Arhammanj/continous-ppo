"""Legacy Gym -> Gymnasium adapter for :class:`src.mg_env.MicroGridEnv`.

The microgrid environment follows the original (legacy) Gym API:

* ``reset()``            -> ``obs``
* ``step(a)``            -> ``(obs, reward, terminal, info)``

Stable-Baselines3 (>= 2.0) requires the Gymnasium API:

* ``reset()``            -> ``(obs, info)``
* ``step(a)``            -> ``(obs, reward, terminated, truncated, info)``

This adapter is a *thin* translation layer: it does **not** change the battery
dynamics, the safety shield, the price data, the action semantics or the
pure-profit reward. It only:

* wraps the legacy return values,
* re-declares the continuous action and observation spaces as
  ``gymnasium.spaces`` (``gym`` spaces are a different class from
  ``gymnasium`` spaces),
* exposes observation bounds that actually contain the observed values (the
  env's own ``Box(0, 1)`` is exceeded by the time-step feature at the terminal
  step and by the signed price window).
"""
from __future__ import annotations

import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.mg_env import MicroGridEnv, ACTIONS


def observation_bounds(env: MicroGridEnv):
    """Return (low, high) arrays consistent with the env's actual observations."""
    iterations = env.iterations
    time_high = iterations / (iterations - 1) if iterations > 1 else 1.0
    soc_low = float(env.battery.min_soc) / float(env.battery.max_soc)
    soc_high = 1.0

    n = env.observation_space.shape[0]
    low = np.empty(n, dtype=np.float32)
    high = np.empty(n, dtype=np.float32)
    low[0], high[0] = 0.0, float(time_high)          # normalized time-step
    low[1], high[1] = float(soc_low), float(soc_high)  # SOC / max_soc
    low[2:], high[2:] = -1.0, 1.0                    # normalized price window
    return low, high


class GymnasiumAdapter(gym.Env):
    """Wrap a legacy ``MicroGridEnv`` to expose the Gymnasium API."""

    metadata = {"render_modes": []}

    def __init__(self, env: MicroGridEnv):
        super().__init__()
        self.env = env
        self.render_mode = None

        # Re-declare spaces using gymnasium.spaces (SB3 inspects these).
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        low, high = observation_bounds(env)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # -- Gymnasium API -----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # Seed every RNG the legacy env relies on: python ``random`` for
            # day selection, numpy for the battery state, and the action space.
            random.seed(seed)
            np.random.seed(seed)
            self.action_space.seed(seed)
            try:
                self.env.action_space.seed(seed)
            except Exception:
                pass

        obs = self.env.reset()
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        normalized_action = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        normalized_action = float(np.clip(normalized_action, -1.0, 1.0))
        physical_action = normalized_action * float(max(abs(value) for value in ACTIONS))
        obs, reward, terminated, info = self.env.step(physical_action)
        truncated = False  # the 24-step episode ends via ``terminated``
        return (np.asarray(obs, dtype=np.float32), float(reward),
                bool(terminated), truncated, info if info is not None else {})

    def render(self):
        # The legacy env's ``render`` returns the operation cost and appends to
        # module-level lists; it is intentionally NOT used for metrics.
        return self.env.render()

    def close(self):
        return None

    # -- Convenience pass-throughs (used by the metrics callback) ----------
    @property
    def last_profit(self):
        return float(getattr(self.env, "last_profit", 0.0))

    @property
    def operation_cost(self):
        return float(getattr(self.env, "operation_cost", 0.0))

    @property
    def total_current_output(self):
        return float(getattr(self.env, "total_current_output", 0.0))

    @property
    def last_safety_violation(self):
        return float(getattr(self.env, "last_safety_violation", 0.0))
