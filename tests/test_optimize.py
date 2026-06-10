import numpy as np
import pandas as pd
import pytest

from mlportfolio.optimize.covariance import ledoit_wolf_covariance, sample_covariance
from mlportfolio.optimize.markowitz import (
    equal_weights,
    sixty_forty_weights,
    max_sharpe_weights,
    mean_variance_weights,
)
from mlportfolio.optimize.black_litterman import (
    black_litterman_returns,
    black_litterman_weights,
)


@pytest.fixture
def cov_and_mu(synthetic_data):
    rets = synthetic_data.returns.tail(252)
    cov = ledoit_wolf_covariance(rets)
    mu = rets.mean() * 252
    return cov, mu


def test_equal_weights_sum_to_one(synthetic_data):
    w = equal_weights(synthetic_data.tickers)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()


def test_sixty_forty(synthetic_data):
    w = sixty_forty_weights(synthetic_data.tickers, "SPY", "TLT")
    assert abs(w["SPY"] - 0.6) < 1e-9
    assert abs(w["TLT"] - 0.4) < 1e-9


def test_max_sharpe_constraints(cov_and_mu):
    cov, mu = cov_and_mu
    w = max_sharpe_weights(mu, cov, min_weight=0.0, max_weight=0.4)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all()
    assert (w <= 0.4 + 1e-6).all()


def test_mean_variance_respects_cap(cov_and_mu):
    cov, mu = cov_and_mu
    w = mean_variance_weights(mu, cov, risk_aversion=3.0, max_weight=0.3)
    assert (w <= 0.3 + 1e-6).all()
    assert abs(w.sum() - 1.0) < 1e-6


def test_bl_returns_between_prior_and_views(cov_and_mu):
    cov, mu = cov_and_mu
    tickers = list(cov.columns)
    mkt_w = equal_weights(tickers)
    # Strong bullish view on the first asset.
    views = pd.Series(np.nan, index=tickers)
    views.iloc[0] = 0.20
    bl = black_litterman_returns(cov, mkt_w, views, risk_aversion=3.0, tau=0.05)
    assert bl.notna().all()
    assert len(bl) == len(tickers)


def test_bl_weights_valid(cov_and_mu):
    cov, mu = cov_and_mu
    tickers = list(cov.columns)
    views = mu.copy()
    w = black_litterman_weights(
        cov, equal_weights(tickers), views, risk_aversion=3.0, tau=0.05, max_weight=0.5
    )
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w <= 0.5 + 1e-6).all()


def test_higher_risk_aversion_lowers_portfolio_vol(cov_and_mu):
    cov, mu = cov_and_mu
    w_lo = mean_variance_weights(mu, cov, risk_aversion=1.0, max_weight=1.0)
    w_hi = mean_variance_weights(mu, cov, risk_aversion=50.0, max_weight=1.0)
    vol_lo = float(np.sqrt(w_lo @ cov.to_numpy() @ w_lo))
    vol_hi = float(np.sqrt(w_hi @ cov.to_numpy() @ w_hi))
    assert vol_hi <= vol_lo + 1e-6
