"""Feature engineering.

Builds the feature panel used by both the regime model and the return
forecaster. Every feature at row ``t`` is computable using information available
*up to and including* day ``t`` (no forward-looking transforms), which is what
keeps the downstream walk-forward honest.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .loaders import MarketData


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder), bounded 0-100."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def build_market_features(
    data: MarketData,
    momentum_windows: List[int],
    vol_window: int,
    ma_windows: List[int],
) -> pd.DataFrame:
    """Aggregate, market-level features used by the HMM regime detector.

    These describe the *state of the market as a whole* rather than any single
    asset: broad return, broad volatility, the VIX, and macro spreads.
    """
    eq = data.returns.mean(axis=1)  # equal-weight market return proxy
    feats: Dict[str, pd.Series] = {}
    feats["mkt_ret_5"] = eq.rolling(5).mean()
    feats["mkt_ret_21"] = eq.rolling(21).mean()
    feats["mkt_vol"] = eq.rolling(vol_window).std()
    feats["vix"] = data.vix
    feats["vix_chg"] = data.vix.pct_change(5)

    for col in data.macro.columns:
        feats[f"macro_{col}"] = data.macro[col]

    out = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan)
    return out.dropna()


def build_feature_panel(
    data: MarketData,
    momentum_windows: List[int],
    vol_window: int,
    ma_windows: List[int],
) -> Dict[str, pd.DataFrame]:
    """Per-asset technical features for the return forecaster.

    Returns a dict ``{ticker: feature_dataframe}``. Each feature dataframe shares
    the same columns so models can be trained per-asset or pooled.
    """
    panels: Dict[str, pd.DataFrame] = {}
    for ticker in data.tickers:
        px = data.prices[ticker]
        ret = data.returns[ticker]
        feats: Dict[str, pd.Series] = {}

        for w in momentum_windows:
            feats[f"mom_{w}"] = px.pct_change(w)
        feats["vol"] = ret.rolling(vol_window).std()
        for w in ma_windows:
            feats[f"ma_ratio_{w}"] = px / px.rolling(w).mean() - 1.0
        feats["rsi_14"] = _rsi(px, 14)

        # Market-wide context shared across assets.
        feats["vix"] = data.vix
        feats["vix_chg"] = data.vix.pct_change(5)
        for col in data.macro.columns:
            feats[f"macro_{col}"] = data.macro[col]

        df = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan)
        panels[ticker] = df
    return panels
