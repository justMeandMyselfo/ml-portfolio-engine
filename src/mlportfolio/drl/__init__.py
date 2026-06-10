"""Deep reinforcement-learning allocation (experimental).

Stable-Baselines3 + PyTorch are OPTIONAL. The environment and the
policy-to-strategy plumbing work with only gymsium installed; training requires
the ``[drl]`` extra (``pip install -e ".[drl]"``).
"""

from .env import PortfolioEnv, compute_observation, normalize_action
from .agent import (
    DRLStrategy,
    random_policy_strategy,
    build_rebalance_dates,
    train_drl,
    HAS_SB3,
)

__all__ = [
    "PortfolioEnv",
    "compute_observation",
    "normalize_action",
    "DRLStrategy",
    "random_policy_strategy",
    "build_rebalance_dates",
    "train_drl",
    "HAS_SB3",
]
