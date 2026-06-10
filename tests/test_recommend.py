import numpy as np
import pandas as pd

from mlportfolio.config import Config
from mlportfolio.recommend import recommend, format_report


def _cfg():
    return Config.default().override({
        "forecast.n_estimators": 60,
        "regime.n_iter": 60,
    })


def test_recommendation_structure():
    rec = recommend(_cfg(), synthetic=True)
    w = rec.target_weights
    assert (w >= -1e-9).all()
    assert w.sum() <= 1.0 + 1e-6
    assert 0.0 <= rec.cash_weight <= 1.0
    assert abs(w.sum() + rec.cash_weight - 1.0) < 1e-6
    assert rec.est_volatility >= 0
    assert rec.regime in (0, 1, None)
    assert "NOT financial" in rec.disclaimer


def test_recommendation_trades_vs_holdings():
    holdings = {"SPY": 0.5, "TLT": 0.5}
    rec = recommend(_cfg(), synthetic=True, current_holdings=holdings)
    assert rec.trades is not None
    # Trade column equals target minus current for listed rows.
    for asset, row in rec.trades.iterrows():
        assert abs(row["trade"] - (row["target"] - row["current"])) < 1e-9


def test_format_report_markdown():
    rec = recommend(_cfg(), synthetic=True)
    md = format_report(rec)
    assert "Allocation recommendation" in md
    assert "Recommended allocation" in md
    assert "NOT financial" in md
