"""DQN agent entry point.

This is a thin compatibility wrapper around the existing implementation in
``src/q_agent.py``. The reference Double-DQN algorithm, Q-network, replay
buffer, optimizer, exploration schedule and checkpoint format are **unchanged**
-- they are simply re-exported here so the DQN pipeline has a single, stable
import surface (``from dqn.agent import QAgent``).

Nothing in ``src/q_agent.py`` was rewritten for PPO; keeping the original file
in place also preserves the legacy trace/analyse scripts that import it.
"""
from src.q_agent import QAgent
from src.model_factory import get_model, count_parameters
from src.agent_memory import AgentMemory, Transition

__all__ = ["QAgent", "get_model", "count_parameters", "AgentMemory", "Transition"]
