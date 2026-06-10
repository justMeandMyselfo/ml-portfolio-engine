"""Performance and risk metrics.

All functions take a series of *periodic* (daily) returns and annualize using
``periods_per_year`` (252 trading days by default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    growth = (1.0 + returns).prod()
    years = len(returns) / periods_per_year
    if years <= 0:
        return float("nan")
    return float(growth ** (1.0 / years) - 1.0)


def annualized_vol(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    excess = returns.dropna() - risk_free / periods_per_year
    vol = excess.std(ddof=0)
    # Near-zero volatility (e.g. a constant return stream) leaves the Sharpe
    # ratio undefined; guard with a tolerance rather than exact-zero equality.
    if not np.isfinite(vol) or vol < 1e-12:
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    excess = returns.dropna() - risk_free / periods_per_year
    downside = excess[excess < 0]
    dd = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    # No downside observations -> downside deviation is zero -> Sortino is
    # (positively) undefined; report NaN rather than a spurious infinity.
    if not np.isfinite(dd) or dd < 1e-12:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Most negative peak-to-trough drawdown of the cumulative equity curve."""
    curve = (1.0 + returns.dropna()).cumprod()
    if curve.empty:
        return float("nan")
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(annualized_return(returns, periods_per_year) / abs(mdd))


def performance_summary(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS
) -> dict:
    """Return a dict of headline metrics for one return stream."""
    return {
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_vol": annualized_vol(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, 0.0, periods_per_year),
        "sortino": sortino_ratio(returns, 0.0, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
    }
