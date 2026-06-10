"""Classical Markowitz mean-variance optimization.

Implemented directly on top of ``scipy.optimize`` so the project is
self-contained (no heavyweight optimizer dependency) and so the mechanics are
fully visible — which is the point of contrasting it with the ML approaches.

All optimizers honor the same constraint set: fully invested (weights sum to 1),
per-asset bounds, and optional long-only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _bounds(n: int, min_w: float, max_w: float, allow_short: bool):
    lo = -abs(max_w) if allow_short else max(min_w, 0.0)
    return [(lo, max_w) for _ in range(n)]


def _sum_to_one():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def equal_weights(tickers) -> pd.Series:
    n = len(tickers)
    return pd.Series(np.repeat(1.0 / n, n), index=tickers, name="weight")


def sixty_forty_weights(tickers, equity_proxy: str, bond_proxy: str) -> pd.Series:
    """60% equity proxy / 40% bond proxy; falls back to equal-weight if missing."""
    w = pd.Series(0.0, index=tickers, name="weight")
    if equity_proxy in w.index and bond_proxy in w.index:
        w[equity_proxy] = 0.60
        w[bond_proxy] = 0.40
        return w
    return equal_weights(tickers)


def mean_variance_weights(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float = 3.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    allow_short: bool = False,
) -> pd.Series:
    """Maximize ``mu' w - 0.5 * lambda * w' Sigma w`` subject to constraints."""
    tickers = list(cov.columns)
    mu = expected_returns.reindex(tickers).fillna(0.0).to_numpy()
    sigma = cov.to_numpy()
    n = len(tickers)

    def neg_utility(w):
        return -(mu @ w - 0.5 * risk_aversion * w @ sigma @ w)

    w0 = np.repeat(1.0 / n, n)
    res = minimize(
        neg_utility,
        w0,
        method="SLSQP",
        bounds=_bounds(n, min_weight, max_weight, allow_short),
        constraints=[_sum_to_one()],
        options={"maxiter": 500, "ftol": 1e-9},
    )
    w = res.x if res.success else w0
    w = np.clip(w, None, max_weight)
    w = w / w.sum() if w.sum() != 0 else w0
    return pd.Series(w, index=tickers, name="weight")


def max_sharpe_weights(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    risk_free: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    allow_short: bool = False,
) -> pd.Series:
    """Maximize the (annualized) Sharpe ratio of the portfolio."""
    tickers = list(cov.columns)
    mu = expected_returns.reindex(tickers).fillna(0.0).to_numpy()
    sigma = cov.to_numpy()
    n = len(tickers)

    def neg_sharpe(w):
        ret = mu @ w - risk_free
        vol = np.sqrt(max(w @ sigma @ w, 1e-12))
        return -ret / vol

    w0 = np.repeat(1.0 / n, n)
    res = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=_bounds(n, min_weight, max_weight, allow_short),
        constraints=[_sum_to_one()],
        options={"maxiter": 500, "ftol": 1e-9},
    )
    w = res.x if res.success else w0
    w = np.clip(w, None, max_weight)
    w = w / w.sum() if w.sum() != 0 else w0
    return pd.Series(w, index=tickers, name="weight")
