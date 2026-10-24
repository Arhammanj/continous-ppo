# Microgrid Battery Management — DQN & PPO on a Pure-Profit Objective

Reinforcement-learning control of a grid-connected battery in a small microgrid.
Two independent training pipelines share **one** environment implementation and
**one** reward objective: `pure_profit`.

* **DQN** — the preserved Double-DQN implementation (PyTorch).
* **PPO** — Stable-Baselines3 (`MlpPolicy`) with a one-dimensional continuous
  action in `[-1, 1]`, mapped to the physical battery power range
  `[-80, 80]`, interacting with the same environment through a thin
  Gym→Gymnasium adapter.

The battery safety shield and all environment economics are unchanged from the
original project; the DQN reference experiment is preserved as the baseline.

---

## Reward definition (single objective)

The only supported training reward is `pure_profit` (`src/mg_env.py`,
`_pure_profit_reward`):

```
profit       = sell_benefit − buy_cost
reward       = profit / profit_scale
profit_scale = (p98(buy_price) − p2(buy_price)) × max|ACTIONS|
```

* Charging (`energy ≥ 0`): `buy_cost = buy_price × energy`
* Discharging (`energy < 0`): `sell_benefit = sell_price × |energy|`, where
  `sell_price = buy_price × 0.7`
* No battery-degradation term, no SOC penalty, no price-alignment bonus, no
  reward shaping and no schedules.

**Safety is enforced by the environment, not the reward.** `Battery.step()`
clips every action so the projected SOC stays inside the hard safe band
`[0.3, 0.7]`; safety violations are therefore impossible by construction and
`last_safety_violation` is always `0`.

Economic profit (currency) and the normalized training reward are distinct:
evaluation always reports `env.last_profit` summed over the day, never the sum
of normalized rewards.

---

## Folder structure

```
Papercodemicrogrid-master/
├── src/
│   ├── mg_env.py            # shared environment + pure_profit reward + safety shield
│   ├── config.py            # shared paths/settings (models/, results/, eval days, REWARD_MODE)
│   ├── q_agent.py           # preserved Double-DQN agent (checkpoint format unchanged)
│   ├── loops.py             # preserved DQN training loop
│   ├── model_factory.py     # preserved Q-network factory
│   ├── agent_memory.py      # preserved replay buffer
│   ├── utils.py             # seeded helpers, dims, epsilon schedule
│   └── Prices.csv           # hourly day-ahead prices (8760 h = 365 days)
├── dqn/
│   ├── agent.py             # thin re-export of the preserved DQN agent
│   ├── train.py             # DQN training entry point
│   └── evaluate.py          # DQN held-out evaluation
├── ppo/
│   ├── adapter.py           # legacy Gym → Gymnasium adapter (continuous action + API)
│   ├── train.py             # PPO training entry point (+ economic-metrics callback)
│   └── evaluate.py          # PPO held-out evaluation
├── evaluation/
│   ├── metrics.py           # shared economic metrics + evaluation protocol
│   └── compare_agents.py    # DQN vs PPO on identical held-out days
├── configs/
│   ├── dqn.yaml             # DQN experiment configuration
│   └── ppo.yaml             # PPO experiment configuration
├── models/{dqn,ppo}/        # algorithm-specific checkpoints
├── results/{dqn,ppo}/       # algorithm-specific logs & plots
├── tests/test_pipeline.py   # smoke tests
├── conftest.py              # makes the project root importable under pytest
└── runner_train.py          # preserved original DQN entry point (reference baseline)
```

---

## Setup

Python 3.10+ with:

```bash
pip install torch matplotlib numpy pandas scipy pyyaml tqdm gym gymnasium stable-baselines3 pytest
```

All commands below are run **from the project root**
(`Papercodemicrogrid-master/`).

---

## Training

### DQN

```bash
python -m dqn.train                      # uses configs/dqn.yaml
python dqn/train.py --episodes 2000      # quick override
python dqn/train.py --eval-freq 50       # greedy-eval frequency override
```

Outputs:
* best checkpoint → `models/dqn/best_checkpoint/`
* final agent     → `models/dqn/final/`
* training plot   → `results/dqn/reward_cost_<N>eps.png`

### PPO

```bash
python -m ppo.train                      # uses configs/ppo.yaml
python ppo/train.py --timesteps 100000   # quick override
python ppo/train.py --seed 1
```

Outputs:
* model          → `models/ppo/ppo_model.zip`
* metrics CSV    → `results/ppo/ppo_training_metrics.csv`
  (per-episode economic profit and operation cost, logged by a callback — not
  by `env.render`)

Algorithm-specific hyper-parameters live in `configs/dqn.yaml` and
`configs/ppo.yaml`.

---

## Evaluation (held-out days 300–364)

Both agents are evaluated **deterministically** on the **same** held-out days
`300–364` (training uses days `0–299`).

```bash
python -m dqn.evaluate          # -> results/dqn/eval.json, eval_daily.csv, eval_daily_profit.png
python -m ppo.evaluate          # -> results/ppo/eval.json, eval_daily.csv, eval_daily_profit.png
python -m evaluation.compare_agents   # -> results/comparison.json, comparison.png
```

Reported metrics: daily/total economic profit, mean & std daily profit, energy
bought/sold, safety violations, total operation cost.

---

## Tests

```bash
pytest -q
```

Covers: env reset/step, pure-profit reward formula and per-step invariants,
battery safety shield and action mapping, Gymnasium adapter compatibility
(including SB3 `check_env`), DQN checkpoint round-trip, PPO train/load, economic
evaluation, and confirmation that all supported entry points use `pure_profit`.

---

## Design notes

* **One environment, two algorithms.** Both pipelines construct
  `MicroGridEnv(...)` with `reward_mode="pure_profit"`; PPO only adds the
  `GymnasiumAdapter`, which translates the legacy 4-tuple API to the Gymnasium
  5-tuple API and re-declares the spaces as `gymnasium.spaces` (the env's own
  `Box(0, 1)` is exceeded by the terminal time-step and the signed price
  window, so the adapter exposes bounds that actually contain the observations).
* **Preserved baseline.** `src/q_agent.py`, `src/loops.py` and
  `runner_train.py` are untouched in behaviour; `dqn/agent.py` is a thin
  re-export so the DQN code has a stable import surface.
* **Legacy reward modes.** The other reward functions remain in `mg_env.py`
  for research/legacy scripts (`trace_*.py`, `analyze_profit_vs_spread.py` use
  `profit_safety`), but **no** supported training entry point uses them. The
  module default was standardized to `pure_profit`.
