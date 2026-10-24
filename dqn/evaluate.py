"""DQN evaluation on the shared held-out days (default 300-364).

Run from the project root::

    python -m dqn.evaluate
    python dqn/evaluate.py --checkpoint models/dqn/best_checkpoint
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn.agent import QAgent
from evaluation.metrics import (
    evaluate_agent, load_config, make_env, print_evaluation, save_evaluation,
)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "dqn.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint dir (default: <model_dir>/best_checkpoint)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    model_dir = resolve(cfg["paths"]["model_dir"])
    result_dir = resolve(cfg["paths"]["result_dir"])

    ckpt = resolve(args.checkpoint) if args.checkpoint else model_dir / "best_checkpoint"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    env = make_env(day0=env_cfg["day0"], dayn=env_cfg["dayn"],
                   reward_mode=env_cfg["reward_mode"])
    agent = QAgent.load_from_disk(env, ckpt)

    day0, dayn = cfg["eval"]["day0"], cfg["eval"]["dayn"]
    action_fn = lambda state: agent.act(state, epsilon=0.0)
    results = evaluate_agent(env, action_fn, day0=day0, dayn=dayn)
    print_evaluation("DQN", results)
    save_evaluation(result_dir, "DQN", results, checkpoint=str(ckpt),
                    reward_mode=env_cfg["reward_mode"])
    return results


if __name__ == "__main__":
    main()
