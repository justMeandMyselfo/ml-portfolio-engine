"""End-to-end integration test on synthetic data (kept small for speed)."""

import numpy as np

from mlportfolio.config import Config
from mlportfolio.pipeline import Pipeline, summary_table
from mlportfolio.data.features import build_feature_panel, build_market_features


def _small_pipeline():
    cfg = Config.default().override(
        {
            "forecast.n_estimators": 40,
            "regime.n_iter": 50,
            "backtest.warmup_days": 400,
        }
    )
    pipe = Pipeline(cfg, synthetic=True)
    pipe.load()
    # Shrink the sample so the monthly refit loop stays fast in CI.
    cutoff = pipe.data.returns.index[650]
    pipe.data = pipe.data.slice(cutoff)
    pipe.panels = build_feature_panel(pipe.data, pipe.momentum_windows, pipe.vol_window, pipe.ma_windows)
    pipe.market_features = build_market_features(
        pipe.data, pipe.momentum_windows, pipe.vol_window, pipe.ma_windows
    )
    return pipe


def test_pipeline_runs_all_strategies():
    pipe = _small_pipeline()
    artifacts = pipe.run()
    expected = {
        "EqualWeight",
        "SixtyForty",
        "Markowitz",
        "BlackLitterman_RF",
        "RegimeAware_BL_RF",
    }
    assert set(artifacts.results.keys()) == expected
    for name, res in artifacts.results.items():
        assert np.isfinite(res.equity_curve.iloc[-1]), name
        assert res.equity_curve.iloc[-1] > 0, name


def test_summary_table_well_formed():
    pipe = _small_pipeline()
    artifacts = pipe.run()
    table = summary_table(artifacts)
    assert "sharpe" in table.columns
    assert "max_drawdown" in table.columns
    assert len(table) == 5
