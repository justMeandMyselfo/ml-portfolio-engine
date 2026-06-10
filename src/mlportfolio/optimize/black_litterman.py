"""Black-Litterman with machine-learning views.

Classic Black-Litterman blends a market-implied *prior* (reverse-optimized from
benchmark weights) with an investor's *views*. Here the views are not subjective
analyst opinions — they are the Random Forest's forecasts of next-period returns.
View uncertainty (the Omega matrix) is derived from the forecaster's own
cross-tree confidence, so the model leans harder on views it is more sure about.

References
----------
He & Litterman (1999), "The Intuition Behind Black-Litterman Model Portfolios."
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .markowitz import max_sharpe_weights


def market_implied_prior(
    cov: pd.DataFrame, market_weights: pd.Series, risk_aversion: float
) -> pd.Series:
    """Reverse-optimization: pi = lambda * Sigma * w_mkt."""
    w = market_weights.reindex(cov.columns).fillna(0.0).to_numpy()
    pi = risk_aversion * cov.to_numpy() @ w
    return pd.Series(pi, index=cov.columns, name="prior")


def black_litterman_returns(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    view_returns: pd.Series,
    view_confidence: Optional[pd.Series] = None,
    risk_aversion: float = 3.0,
    tau: float = 0.05,
) -> pd.Series:
    """Return the Black-Litterman posterior expected returns.

    We use absolute views (one view per asset), so the picking matrix P is the
    identity over the assets we actually have a view on.
    """
    tickers = list(cov.columns)
    sigma = cov.to_numpy()
    pi = market_implied_prior(cov, market_weights, risk_aversion).to_numpy()

    view_returns = view_returns.reindex(tickers)
    have_view = view_returns.notna().to_numpy()
    if not have_view.any():
        return pd.Series(pi, index=tickers, name="bl_return")

    idx = np.where(have_view)[0]
    P = np.eye(len(tickers))[idx]                 # k x n picking matrix
    Q = view_returns.to_numpy()[idx]              # k view returns

    tau_sigma = tau * sigma

    # Omega: diagonal view-uncertainty. Base it on the view variances, scaled by
    # the forecaster's confidence (higher confidence => smaller Omega).
    base = np.diag(P @ tau_sigma @ P.T)
    if view_confidence is not None:
        conf = view_confidence.reindex(tickers).to_numpy()[idx]
        conf = np.nan_to_num(conf, nan=np.nanmedian(conf) if np.isfinite(np.nanmedian(conf)) else 1.0)
        conf = conf / (np.median(conf) + 1e-12)   # normalize around 1
        scale = 1.0 / np.clip(conf, 1e-3, 1e3)    # more confidence -> less uncertainty
    else:
        scale = np.ones_like(base)
    omega = np.diag(np.clip(base * scale, 1e-10, None))

    # Posterior mean (He-Litterman closed form).
    tau_sigma_inv = np.linalg.pinv(tau_sigma)
    omega_inv = np.linalg.pinv(omega)
    post_cov = np.linalg.pinv(tau_sigma_inv + P.T @ omega_inv @ P)
    post_mean = post_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

    return pd.Series(post_mean, index=tickers, name="bl_return")


def black_litterman_weights(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    view_returns: pd.Series,
    view_confidence: Optional[pd.Series] = None,
    risk_aversion: float = 3.0,
    tau: float = 0.05,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    allow_short: bool = False,
) -> pd.Series:
    """Full pipeline: BL posterior returns -> max-Sharpe weights."""
    bl_mu = black_litterman_returns(
        cov=cov,
        market_weights=market_weights,
        view_returns=view_returns,
        view_confidence=view_confidence,
        risk_aversion=risk_aversion,
        tau=tau,
    )
    return max_sharpe_weights(
        expected_returns=bl_mu,
        cov=cov,
        min_weight=min_weight,
        max_weight=max_weight,
        allow_short=allow_short,
    )
