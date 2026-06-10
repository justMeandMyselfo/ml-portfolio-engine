"""Risk overlays applied on top of the optimizer's raw weights.

Two overlays, both aimed at the weaknesses our backtest exposed:

* **Volatility targeting** scales total exposure so the portfolio's estimated
  annualized volatility sits near a target. When markets are calm it leans in;
  when the covariance widens (turbulent regimes) it automatically pulls back and
  parks the remainder in cash. This is the main lever for cutting drawdowns.
* **Turnover control** damps trading. The optimizer can swing weights sharply
  month to month; left unchecked, transaction costs erode the ML edge (exactly
  what we saw). We blend partway toward the new target and ignore trades smaller
  than a no-trade band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Annualized portfolio volatility. ``cov`` is expected to be annualized."""
    w = weights.reindex(cov.columns).fillna(0.0).to_numpy()
    var = float(w @ cov.to_numpy() @ w)
    return float(np.sqrt(max(var, 0.0)))


def volatility_target(
    weights: pd.Series,
    cov: pd.DataFrame,
    target_vol: float,
    max_leverage: float = 1.0,
) -> pd.Series:
    """Scale ``weights`` toward ``target_vol``; the unallocated remainder is cash.

    Returns weights that may sum to less than 1 (the shortfall is the implied
    cash holding). Never levers beyond ``max_leverage`` (default 1.0 = no
    leverage, long-only cash buffer only).
    """
    vol = portfolio_volatility(weights, cov)
    if vol <= 1e-8:
        return weights.copy()
    scale = min(target_vol / vol, max_leverage)
    scale = max(scale, 0.0)
    return weights * scale


def apply_turnover_control(
    target: pd.Series,
    current: pd.Series,
    smoothing: float = 1.0,
    no_trade_band: float = 0.0,
) -> pd.Series:
    """Blend from ``current`` toward ``target`` and suppress tiny trades.

    ``smoothing`` is the fraction of the gap to close (1.0 = full rebalance to
    target, 0.5 = move halfway). ``no_trade_band`` leaves a position untouched
    when the proposed change is smaller than the band, avoiding churn on noise.
    Both series are fractions of the portfolio over the same asset set.
    """
    idx = target.index.union(current.index)
    t = target.reindex(idx).fillna(0.0)
    c = current.reindex(idx).fillna(0.0)

    smoothing = float(np.clip(smoothing, 0.0, 1.0))
    blended = c + smoothing * (t - c)

    if no_trade_band > 0:
        move = (blended - c).abs()
        blended = blended.where(move >= no_trade_band, c)

    return blended
