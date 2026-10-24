"""Smoke tests for the standardized microgrid DQN/PPO pipelines.

Run from the project root::

    pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

from src import mg_env
from src.mg_env import MicroGridEnv, ACTIONS
from evaluation.metrics import evaluate_agent, make_env
from ppo.adapter import GymnasiumAdapter


# --------------------------------------------------------------------------
# 1. Environment reset and step
# --------------------------------------------------------------------------
def test_env_reset_and_step():
    env = make_env(day0=300, dayn=365, reward_mode="pure_profit")
    obs = env.reset(day=300)
    assert np.asarray(obs).shape == (26,)

    done = False
    steps = 0
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        steps += 1
        assert np.asarray(obs).shape == (26,)
        assert np.isfinite(float(reward))
    assert steps == env.iterations == 24


# --------------------------------------------------------------------------
# 2. Pure-profit reward calculation
# --------------------------------------------------------------------------
def test_pure_profit_reward_formula():
    env = make_env(day0=300, dayn=365, reward_mode="pure_profit")
    scale = env._get_profit_norm_scale()
    reward = env._pure_profit_reward(sell_benefit=10.0, buy_cost=4.0)
    assert reward == pytest.approx((10.0 - 4.0) / scale)
    assert env.last_profit == pytest.approx(6.0)
    assert env.last_safety_violation == 0.0


def test_pure_profit_step_invariants():
    """reward == profit/scale, operation_cost == -profit, for every step."""
    env = make_env(day0=300, dayn=365, reward_mode="pure_profit")
    scale = env._get_profit_norm_scale()
    env.reset(day=301)
    done = False
    while not done:
        _, reward, done, _ = env.step(env.action_space.sample())
        assert float(reward) == pytest.approx(float(env.last_profit) / scale)
        assert float(env.operation_cost) == pytest.approx(-float(env.last_profit))


# --------------------------------------------------------------------------
# 3. Battery safety limits and action mapping
# --------------------------------------------------------------------------
def test_battery_shield_and_action_mapping():
    env = make_env(day0=300, dayn=365, reward_mode="pure_profit")
    env.reset(day=302)
    # Action index 0 maps to the most negative power (max discharge).
    env.step(0)
    assert env.prev_battery_action == float(ACTIONS[0])

    env.reset(day=303)
    done = False
    while not done:
        # Always attempt extreme actions to stress the shield.
        _, _, done, _ = env.step(0 if np.random.rand() < 0.5 else len(ACTIONS) - 1)
        soc = float(env.battery.SOC)
        assert env.battery.safe_soc_low - 1e-9 <= soc <= env.battery.safe_soc_high + 1e-9
        assert env.battery.min_soc - 1e-9 <= soc <= env.battery.max_soc + 1e-9


# --------------------------------------------------------------------------
# 4. Gymnasium adapter compatibility
# --------------------------------------------------------------------------
def test_gymnasium_adapter_api():
    import gymnasium

    adapter = GymnasiumAdapter(make_env(day0=300, dayn=365, reward_mode="pure_profit"))
    assert isinstance(adapter, gymnasium.Env)

    obs, info = adapter.reset(seed=0)
    assert isinstance(info, dict)
    assert obs.shape == (26,)
    assert adapter.observation_space.contains(obs)

    obs, reward, terminated, truncated, info = adapter.step(np.array([0.0], dtype=np.float32))
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert truncated is False


def test_adapter_passes_sb3_check_env():
    from stable_baselines3.common.env_checker import check_env
    adapter = GymnasiumAdapter(make_env(day0=300, dayn=365, reward_mode="pure_profit"))
    check_env(adapter, warn=True)


def test_adapter_observation_bounds_hold_over_episode():
    adapter = GymnasiumAdapter(make_env(day0=300, dayn=365, reward_mode="pure_profit"))
    obs, _ = adapter.reset(seed=1)
    done = False
    while not done:
        obs, _, terminated, truncated, _ = adapter.step(
            np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)
        )
        assert adapter.observation_space.contains(obs), obs
        done = terminated or truncated


# --------------------------------------------------------------------------
# 5. DQN training (smoke) and checkpoint loading
# --------------------------------------------------------------------------
def test_dqn_checkpoint_roundtrip(tmp_path):
    from dqn.agent import QAgent

    env = make_env(day0=0, dayn=300, reward_mode="pure_profit")
    agent = QAgent(env, steps_epsilon_decay=1000, normalize_state=True,
                   nn_hidden_layers=[64, 64])
    agent.save_to_disk(tmp_path)
    assert (tmp_path / "hparams.json").exists()
    assert (tmp_path / "model").exists()

    loaded = QAgent.load_from_disk(env, tmp_path)
    state = env.reset(day=300)
    action = loaded.act(state, epsilon=0.0)
    assert 0 <= action < len(ACTIONS)


# --------------------------------------------------------------------------
# 6. PPO training (smoke) and checkpoint loading
# --------------------------------------------------------------------------
def test_ppo_train_and_load(tmp_path):
    from stable_baselines3 import PPO

    adapter = GymnasiumAdapter(make_env(day0=0, dayn=300, reward_mode="pure_profit"))
    model = PPO("MlpPolicy", adapter, n_steps=48, batch_size=16, n_epochs=1,
                seed=0, verbose=0)
    model.learn(total_timesteps=48)
    path = tmp_path / "ppo_model"
    model.save(str(path))
    assert Path(str(path) + ".zip").exists()

    reloaded = PPO.load(str(path), env=adapter)
    obs, _ = adapter.reset(seed=0)
    action, _ = reloaded.predict(obs, deterministic=True)
    assert reloaded.action_space.contains(np.asarray(action, dtype=np.float32))


# --------------------------------------------------------------------------
# 7. Economic logging and evaluation
# --------------------------------------------------------------------------
def test_evaluation_metrics():
    env = make_env(day0=300, dayn=365, reward_mode="pure_profit")
    rng = np.random.default_rng(0)
    action_fn = lambda state: int(rng.integers(len(ACTIONS)))
    results = evaluate_agent(env, action_fn, day0=300, dayn=303)

    assert results["n_days"] == 3
    # Economic profit equals the sum of the env's actual per-step profit.
    expected = sum(d["profit"] for d in results["per_day"])
    assert results["total_profit"] == pytest.approx(expected)
    assert results["total_energy_bought"] >= 0.0
    assert results["total_energy_sold"] >= 0.0
    assert results["max_violation"] == 0.0
    assert len(results["daily_profit"]) == 3


# --------------------------------------------------------------------------
# 8. All supported training entry points use only pure_profit
# --------------------------------------------------------------------------
def test_standardized_reward_mode():
    import yaml
    from src import config

    assert mg_env.DEFAULT_REWARD_MODE == "pure_profit"
    assert config.REWARD_MODE == "pure_profit"
    assert make_env(day0=0, dayn=300).reward_mode == "pure_profit"

    for cfg_name in ("dqn.yaml", "ppo.yaml"):
        cfg = yaml.safe_load(open(ROOT / "configs" / cfg_name))
        assert cfg["env"]["reward_mode"] == "pure_profit"

    # The preserved reference DQN entry point must also be explicit.
    runner = (ROOT / "runner_train.py").read_text()
    assert 'reward_mode="pure_profit"' in runner

    assert config.EVAL_DAY0 == 300 and config.EVAL_DAYN == 365
