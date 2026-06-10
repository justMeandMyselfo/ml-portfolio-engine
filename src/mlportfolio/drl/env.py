"""Gymnasium environment for reinforcement-learning portfolio allocation.

The agent acts as the portfolio manager. At each rebalance date it observes a
compact, leak-safe summary of the market (per-asset momentum and volatility plus
broad-market and VIX context) and outputs a target allocation. It is rewarded
with the **differential Sharpe ratio** (Moody & Saffell, 1998) of the realized,
cost-adjusted return — an online proxy for the Sharpe ratio — so over many
simulated episodes it learns a risk-adjusted allocation policy without ever being
told the Markowitz equations.

Design notes
------------
* Steps are monthly (the rebalance schedule), which keeps episodes short enough to
  train on a laptop and makes the learned policy directly comparable to the other
  strategies through the same walk-forward backtester.
* Observations at date ``d`` use only data up to and including ``d``; realized
  returns are taken over the *following* period. No look-ahead.
* Actions are non-negative weights, normalized to sum to 1 (long-only, fully
  invested in the risky sleeve; the risk overlays handle cash separately).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:  # pragma: no cover - gymnasium is a light dependency
    gym = object  # type: ignore
    spaces = None  # type: ignore
    _HAS_GYM = False


def compute_observation(returns_hist: pd.DataFrame, vix_value: Optional[float]) -> np.ndarray:
    """Build the observation vector from history up to (and including) a date.

    Per asset: 1-month momentum, 3-month momentum, 1-month volatility.
    Market-wide: average 1-month momentum and a normalized VIX level.
    Shape: ``3 * n_assets + 2``.
    """
    r = returns_hist
    last21 = r.tail(21)
    last63 = r.tail(63)
    mom21 = last21.sum().to_numpy()
    mom63 = last63.sum().to_numpy()
    vol21 = last21.std(ddof=0).to_numpy() * np.sqrt(252)
    mkt_mom = float(np.nanmean(mom21)) if mom21.size else 0.0
    vix_feat = 0.0 if vix_value is None or np.isnan(vix_value) else (vix_value / 20.0 - 1.0)
    obs = np.concatenate([mom21, mom63, vol21, [mkt_mom, vix_feat]]).astype(np.float32)
    return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_action(action: np.ndarray, n_assets: int) -> np.ndarray:
    """Map a raw action to long-only weights summing to 1."""
    a = np.clip(np.asarray(action, dtype=np.float64), 0.0, None)
    total = a.sum()
    if total <= 1e-8:
        return np.repeat(1.0 / n_assets, n_assets)
    return a / total


class _BaseEnv:
    """Environment mechanics, independent of gymnasium (testable on its own)."""

    def __init__(
        self,
        returns: pd.DataFrame,
        rebalance_dates: List[pd.Timestamp],
        vix: Optional[pd.Series] = None,
        cost: float = 0.001,
        eta: float = 0.04,
        seed: int = 42,
    ) -> None:
        self.returns = returns.fillna(0.0)
        self.tickers = list(returns.columns)
        self.n_assets = len(self.tickers)
        self.dates = [pd.Timestamp(d) for d in rebalance_dates if pd.Timestamp(d) in returns.index]
        if len(self.dates) < 3:
            raise ValueError("Need at least 3 rebalance dates for an episode.")
        self.vix = vix
        self.cost = cost
        self.eta = eta
        self.obs_dim = 3 * self.n_assets + 2
        self._rng = np.random.default_rng(seed)
        self.reset_state()

    def reset_state(self):
        self._i = 0
        self._prev_w = np.zeros(self.n_assets)
        self._A = 0.0
        self._B = 0.0
        self._returns_log: List[float] = []
        return self._obs_at(self._i)

    def _obs_at(self, i: int) -> np.ndarray:
        d = self.dates[i]
        hist = self.returns.loc[:d]
        vix_val = None
        if self.vix is not None:
            v = self.vix.loc[:d]
            vix_val = float(v.iloc[-1]) if len(v) else None
        return compute_observation(hist, vix_val)

    def _period_asset_returns(self, i: int) -> np.ndarray:
        d0, d1 = self.dates[i], self.dates[i + 1]
        window = self.returns.loc[d0:d1].iloc[1:]  # exclude the decision day itself
        if window.empty:
            return np.zeros(self.n_assets)
        return ((1.0 + window).prod() - 1.0).to_numpy()

    def _differential_sharpe(self, ret: float) -> float:
        delta_a = ret - self._A
        delta_b = ret ** 2 - self._B
        denom = (self._B - self._A ** 2) ** 1.5
        d = 0.0
        if denom > 1e-12:
            d = (self._B * delta_a - 0.5 * self._A * delta_b) / denom
        self._A += self.eta * delta_a
        self._B += self.eta * delta_b
        return float(d)

    def step_state(self, action: np.ndarray):
        w = normalize_action(action, self.n_assets)
        asset_ret = self._period_asset_returns(self._i)
        gross = float((w * asset_ret).sum())
        turnover = float(np.abs(w - self._prev_w).sum())
        net = gross - turnover * self.cost
        reward = self._differential_sharpe(net)

        self._prev_w = w
        self._returns_log.append(net)
        self._i += 1
        terminated = self._i >= (len(self.dates) - 1)
        obs = self._obs_at(self._i) if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        info = {"net_return": net, "turnover": turnover, "weights": w}
        return obs, reward, terminated, info


if _HAS_GYM:

    class PortfolioEnv(gym.Env):
        """Gymnasium wrapper around :class:`_BaseEnv`."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            returns: pd.DataFrame,
            rebalance_dates: List[pd.Timestamp],
            vix: Optional[pd.Series] = None,
            cost: float = 0.001,
            eta: float = 0.04,
            seed: int = 42,
        ) -> None:
            super().__init__()
            self._core = _BaseEnv(returns, rebalance_dates, vix, cost, eta, seed)
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self._core.n_assets,), dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._core.obs_dim,), dtype=np.float32
            )

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self._core._rng = np.random.default_rng(seed)
            obs = self._core.reset_state()
            return obs.astype(np.float32), {}

        def step(self, action):
            obs, reward, terminated, info = self._core.step_state(action)
            return obs.astype(np.float32), float(reward), bool(terminated), False, info

        @property
        def tickers(self):
            return self._core.tickers

else:  # pragma: no cover

    class PortfolioEnv:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("gymnasium is required for PortfolioEnv. pip install gymnasium")
