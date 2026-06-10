"""Covariance estimators used by the optimizers.

Sample covariance is noisy in high dimensions; Ledoit-Wolf shrinkage toward a
scaled identity is a standard, robust default for portfolio work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_covariance(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    cov = returns.cov() * periods_per_year
    return cov


def ledoit_wolf_covariance(
    returns: pd.DataFrame, periods_per_year: int = 252
) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage covariance (annualized)."""
    from sklearn.covariance import LedoitWolf

    clean = returns.dropna(how="any")
    if clean.shape[0] < 2:
        # Degenerate; fall back to a tiny diagonal so optimizers stay stable.
        n = returns.shape[1]
        return pd.DataFrame(
            np.eye(n) * 1e-4, index=returns.columns, columns=returns.columns
        )
    lw = LedoitWolf().fit(clean.to_numpy())
    cov = lw.covariance_ * periods_per_year
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
