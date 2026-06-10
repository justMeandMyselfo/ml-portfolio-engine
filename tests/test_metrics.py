import numpy as np
import pandas as pd

from mlportfolio.backtest.metrics import (
    annualized_return,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    performance_summary,
)


def test_sharpe_of_constant_positive_returns_is_infinite_vol_zero():
    r = pd.Series([0.001] * 252)
    # Zero volatility -> Sharpe undefined (NaN), by our convention.
    assert np.isnan(sharpe_ratio(r))


def test_sharpe_sign_matches_mean():
    rng = np.random.default_rng(0)
    pos = pd.Series(rng.normal(0.001, 0.01, 1000))
    neg = pd.Series(rng.normal(-0.001, 0.01, 1000))
    assert sharpe_ratio(pos) > 0
    assert sharpe_ratio(neg) < 0


def test_max_drawdown_known_case():
    # +100% then -50% returns to start: drawdown is -50%.
    r = pd.Series([1.0, -0.5])
    assert abs(max_drawdown(r) - (-0.5)) < 1e-9


def test_annualized_return_positive_growth():
    r = pd.Series([0.0004] * 252)  # ~10.6% annual
    ar = annualized_return(r)
    assert 0.08 < ar < 0.13


def test_sortino_ignores_upside_vol():
    # Big upside spikes with only small downside should give Sortino a higher
    # score than Sharpe, because Sortino does not penalize upside volatility.
    r = pd.Series([0.05, -0.005, 0.05, -0.005, 0.05, -0.005] * 20)
    assert sortino_ratio(r) > sharpe_ratio(r)


def test_sortino_undefined_without_downside():
    # No negative returns -> downside deviation is zero -> Sortino is NaN.
    r = pd.Series([0.0, 0.05, 0.0, 0.05, 0.0])
    assert np.isnan(sortino_ratio(r))


def test_summary_keys():
    r = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 500))
    summ = performance_summary(r)
    for key in ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar"]:
        assert key in summ
