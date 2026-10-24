import os
import pathlib
root_dir = pathlib.Path(__file__).parent.resolve().parent

SAVED_AGENTS_DIR = root_dir / 'saved_agents'
TENSORBOARD_LOG_DIR = root_dir / 'tensorboard_logs'
OPTUNA_DB = root_dir / 'optuna.db'
DATA_SUPERVISED_ML = root_dir / 'data_supervised_ml'
MLFLOW_RUNS_DIR = root_dir / 'mlflow_runs'

# --- Standardized project layout (DQN / PPO pipelines) ---------------------
CONFIGS_DIR = root_dir / 'configs'
MODELS_DIR = root_dir / 'models'
RESULTS_DIR = root_dir / 'results'

DQN_MODEL_DIR = MODELS_DIR / 'dqn'
PPO_MODEL_DIR = MODELS_DIR / 'ppo'
DQN_RESULT_DIR = RESULTS_DIR / 'dqn'
PPO_RESULT_DIR = RESULTS_DIR / 'ppo'

# Held-out evaluation days shared by both algorithms (prices: 365 days total,
# training uses days 0-299, evaluation uses the untouched 300-364 range).
EVAL_DAY0 = 300
EVAL_DAYN = 365
TRAIN_DAY0 = 0
TRAIN_DAYN = 300

# The single, standardized reward objective for all training entry points.
REWARD_MODE = "pure_profit"

for _d in (SAVED_AGENTS_DIR, TENSORBOARD_LOG_DIR, DATA_SUPERVISED_ML,
           MLFLOW_RUNS_DIR, CONFIGS_DIR, MODELS_DIR, RESULTS_DIR,
           DQN_MODEL_DIR, PPO_MODEL_DIR, DQN_RESULT_DIR, PPO_RESULT_DIR):
    if not _d.exists():
        os.makedirs(_d)