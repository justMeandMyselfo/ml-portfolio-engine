"""Training and deployment of the DRL allocation agent.

Stable-Baselines3 / PyTorch are imported lazily so the rest of the package works
without them. A trained policy is wrapped as an ordinary strategy function and
evaluated through the *same* walk-forward backtester as every other strategy, so
the comparison stays apples-to-apples.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .env import compute_observation, normalize_action

try:
    import stable_baselines3  # noqa: F401
    HAS_SB3 = True
except Exception:
    HAS_SB3 = False


def build_rebalance_dates(index: pd.DatetimeIndex, freq: str = "M", warmup_days: int = 252) -> List[pd.Timestamp]:
    """Month/quarter-end dates within ``index`` after a warmup, pandas-version safe."""
    from ..backtest.engine import _normalize_freq

    if len(index) == 0:
        return []
    start = index[min(warmup_days, len(index) - 1)]
    usable = index[index >= start]
    marks = pd.Series(usable, index=usable).resample(_normalize_freq(freq)).last()
    return [pd.Timestamp(d) for d in marks.values if pd.notna(d)]


class DRLStrategy:
    """Wrap a trained policy as a ``strategy(data_slice, asof) -> weights`` callable.

    ``model`` only needs a ``predict(obs)`` method returning ``(action, _state)``
    (the Stable-Baselines3 convention). Any object satisfying that works, which is
    what lets the random-policy fallback and the unit tests run without PyTorch.
    """

    def __init__(self, model, tickers: List[str], vix: Optional[pd.Series] = None, deterministic: bool = True):
        self.model = model
        self.tickers = list(tickers)
        self.n_assets = len(self.tickers)
        self.vix = vix
        self.deterministic = deterministic

    def __call__(self, data_slice, asof) -> pd.Series:
        returns_hist = data_slice.returns[self.tickers]
        vix_val = None
        if self.vix is not None:
            v = self.vix.loc[:asof]
            vix_val = float(v.iloc[-1]) if len(v) else None
        obs = compute_observation(returns_hist, vix_val)
        try:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
        except TypeError:
            action, _ = self.model.predict(obs)
        weights = normalize_action(np.asarray(action).ravel()[: self.n_assets], self.n_assets)
        return pd.Series(weights, index=self.tickers, name="weight")


class _RandomPolicy:
    def __init__(self, n_assets: int, seed: int = 0):
        self.n_assets = n_assets
        self._rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic: bool = True):
        return self._rng.random(self.n_assets), None


def random_policy_strategy(tickers: List[str], vix: Optional[pd.Series] = None, seed: int = 0) -> DRLStrategy:
    """A baseline 'agent' that allocates randomly. Used for tests and as a sanity floor."""
    return DRLStrategy(_RandomPolicy(len(tickers), seed=seed), tickers, vix=vix)


def train_drl(
    returns: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    vix: Optional[pd.Series] = None,
    total_timesteps: int = 20000,
    cost: float = 0.001,
    algo: str = "PPO",
    seed: int = 42,
    verbose: int = 0,
):
    """Train a PPO/A2C agent on the portfolio environment. Requires the [drl] extra."""
    if not HAS_SB3:
        raise ImportError(
            "stable-baselines3 + torch are required to train the DRL agent. "
            'Install the optional extra:  pip install -e ".[drl]"'
        )
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from .env import PortfolioEnv

    def _make():
        return PortfolioEnv(returns, rebalance_dates, vix=vix, cost=cost, seed=seed)

    venv = DummyVecEnv([_make])
    Algo = {"PPO": PPO, "A2C": A2C}.get(algo.upper(), PPO)
    model = Algo("MlpPolicy", venv, seed=seed, verbose=verbose)
    model.learn(total_timesteps=total_timesteps)
    return model
