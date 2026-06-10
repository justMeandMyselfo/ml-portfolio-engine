import numpy as np
import pandas as pd

from mlportfolio.optimize.risk import (
    portfolio_volatility,
    volatility_target,
    apply_turnover_control,
)
from mlportfolio.optimize.covariance import ledoit_wolf_covariance
from mlportfolio.optimize.markowitz import equal_weights


def test_portfolio_vol_nonnegative(synthetic_data):
    cov = ledoit_wolf_covariance(synthetic_data.returns.tail(252))
    w = equal_weights(synthetic_data.tickers)
    assert portfolio_volatility(w, cov) >= 0


def test_vol_target_scales_down_high_vol(synthetic_data):
    cov = ledoit_wolf_covariance(synthetic_data.returns.tail(252))
    w = equal_weights(synthetic_data.tickers)
    raw_vol = portfolio_volatility(w, cov)
    target = raw_vol / 2.0
    scaled = volatility_target(w, cov, target_vol=target, max_leverage=1.0)
    # Exposure shrinks, leaving cash; realized vol approaches the target.
    assert scaled.sum() < w.sum()
    assert abs(portfolio_volatility(scaled, cov) - target) < 1e-6


def test_vol_target_respects_max_leverage(synthetic_data):
    cov = ledoit_wolf_covariance(synthetic_data.returns.tail(252))
    w = equal_weights(synthetic_data.tickers)
    # Huge target would imply leverage; capped at max_leverage.
    scaled = volatility_target(w, cov, target_vol=10.0, max_leverage=1.0)
    assert scaled.sum() <= 1.0 + 1e-9


def test_turnover_smoothing_endpoints():
    cur = pd.Series({"A": 0.5, "B": 0.5})
    tgt = pd.Series({"A": 1.0, "B": 0.0})
    full = apply_turnover_control(tgt, cur, smoothing=1.0, no_trade_band=0.0)
    none = apply_turnover_control(tgt, cur, smoothing=0.0, no_trade_band=0.0)
    half = apply_turnover_control(tgt, cur, smoothing=0.5, no_trade_band=0.0)
    assert np.allclose(full.values, tgt.values)
    assert np.allclose(none.values, cur.values)
    assert abs(half["A"] - 0.75) < 1e-9


def test_no_trade_band_suppresses_small_moves():
    cur = pd.Series({"A": 0.50, "B": 0.50})
    tgt = pd.Series({"A": 0.52, "B": 0.48})
    out = apply_turnover_control(tgt, cur, smoothing=1.0, no_trade_band=0.05)
    # 0.02 move is below the 0.05 band -> unchanged.
    assert np.allclose(out.values, cur.values)
