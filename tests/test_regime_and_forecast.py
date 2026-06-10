import numpy as np
import pandas as pd

from mlportfolio.data.features import build_market_features, build_feature_panel
from mlportfolio.regime.hmm import RegimeModel
from mlportfolio.forecast.random_forest import RandomForestForecaster


def test_market_features_no_nan_and_have_vol(synthetic_data):
    feats = build_market_features(synthetic_data, [21, 63], 21, [50, 200])
    assert "mkt_vol" in feats.columns
    assert not feats.isna().any().any()
    assert len(feats) > 100


def test_regime_detects_two_states(synthetic_data):
    feats = build_market_features(synthetic_data, [21, 63], 21, [50, 200])
    rm = RegimeModel(n_regimes=2, n_iter=50, seed=1)
    res = rm.fit_predict(feats, vol_col="mkt_vol")
    assert set(res.states.unique()).issubset({0, 1})
    # Risk multiplier for the calm regime should be <= turbulent regime's.
    assert res.risk_multiplier.min() <= res.risk_multiplier.max()


def test_regime_calm_state_has_lower_vol(synthetic_data):
    feats = build_market_features(synthetic_data, [21, 63], 21, [50, 200])
    rm = RegimeModel(n_regimes=2, n_iter=100, seed=1)
    res = rm.fit_predict(feats, vol_col="mkt_vol")
    vol_by_state = feats["mkt_vol"].groupby(res.states).mean()
    # State 0 is defined as the calmest, so its mean vol must be the smallest.
    assert vol_by_state.idxmin() == vol_by_state.index.min()


def test_forecaster_is_leak_safe_and_predicts(synthetic_data):
    panels = build_feature_panel(synthetic_data, [21, 63, 126], 21, [50, 200])
    cutoff = synthetic_data.returns.index[800]
    rf = RandomForestForecaster(horizon=21, n_estimators=80, seed=3)
    rf.fit(panels, synthetic_data.prices, cutoff=cutoff)
    forecast = rf.predict(panels, asof=cutoff)
    assert forecast is not None
    assert forecast.expected_returns.notna().all()
    assert set(forecast.expected_returns.index).issubset(set(synthetic_data.tickers))


def test_forecaster_returns_none_without_history(synthetic_data):
    panels = build_feature_panel(synthetic_data, [21, 63, 126], 21, [50, 200])
    early = synthetic_data.returns.index[30]
    rf = RandomForestForecaster(horizon=21, n_estimators=50, seed=3)
    rf.fit(panels, synthetic_data.prices, cutoff=early)
    # Too little data -> model not fit -> predict returns None.
    assert rf.predict(panels, asof=early) is None
