"""Market-data loaders.

Two paths are supported:

* :func:`load_market_data` pulls real data: asset prices and the VIX from
  Yahoo Finance (via ``yfinance``) and macro series from FRED (via
  ``pandas-datareader``). Results are cached to disk so re-runs are fast and so
  the project still runs if a provider is briefly unavailable.
* :func:`make_synthetic_market_data` generates a regime-switching synthetic
  market so the entire pipeline (and the test suite / CI) can run with **no
  network access**.

Both return the same :class:`MarketData` container, so everything downstream is
agnostic to where the data came from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    """Container for everything the pipeline needs.

    Attributes
    ----------
    prices : DataFrame
        Adjusted close prices, indexed by date, columns are tickers.
    returns : DataFrame
        Daily simple returns derived from ``prices``.
    vix : Series
        Daily VIX level (or a synthetic volatility proxy).
    macro : DataFrame
        Macro series (yield-curve spread, level, credit spread), daily,
        forward-filled.
    """

    prices: pd.DataFrame
    returns: pd.DataFrame
    vix: pd.Series
    macro: pd.DataFrame

    @property
    def tickers(self) -> List[str]:
        return list(self.prices.columns)

    def slice(self, end) -> "MarketData":
        """Return a copy truncated to rows on/before ``end`` (no look-ahead)."""
        end = pd.Timestamp(end)
        return MarketData(
            prices=self.prices.loc[:end],
            returns=self.returns.loc[:end],
            vix=self.vix.loc[:end],
            macro=self.macro.loc[:end],
        )

    def align(self) -> "MarketData":
        """Align all frames on the common date index and drop empty rows."""
        idx = self.returns.dropna(how="all").index
        idx = idx.intersection(self.prices.index)
        prices = self.prices.reindex(idx)
        returns = self.returns.reindex(idx)
        vix = self.vix.reindex(idx).ffill()
        macro = self.macro.reindex(idx).ffill()
        return MarketData(prices=prices, returns=returns, vix=vix, macro=macro)


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: str, name: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{name}.parquet")


def _read_cache(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:  # pragma: no cover - corrupt cache is non-fatal
            return None
    return None


def _write_cache(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_parquet(path)
    except Exception:  # pragma: no cover - parquet engine may be missing
        df.to_csv(path.replace(".parquet", ".csv"))


def _download_prices(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so offline/synthetic runs need no network

    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    # yfinance returns a column MultiIndex when multiple tickers are requested.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]
    prices = prices.dropna(how="all").sort_index()
    return prices


def _download_macro(series: Sequence[str], start: str, end: str) -> pd.DataFrame:
    """Fetch FRED series directly from FRED's public CSV endpoint."""
    frames = []
    for code in series:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
        try:
            df = pd.read_csv(url, na_values=".")
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            df.columns = [code]
            frames.append(df)
        except Exception:  # pragma: no cover
            continue
    if not frames:
        return pd.DataFrame()
    macro = pd.concat(frames, axis=1).sort_index()
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return macro.loc[(macro.index >= lo) & (macro.index <= hi)]


def load_market_data(
    tickers: Sequence[str],
    start: str,
    end: str,
    vix_ticker: str = "^VIX",
    fred_series: Optional[Sequence[str]] = None,
    cache_dir: str = "data_cache",
    use_cache: bool = True,
) -> MarketData:
    """Load real market data (cached on disk).

    Falls back gracefully: if VIX or macro downloads fail, the pipeline still
    runs with neutral placeholders for those features.
    """
    fred_series = list(fred_series or [])

    prices_cache = _cache_path(cache_dir, "prices")
    macro_cache = _cache_path(cache_dir, "macro")
    vix_cache = _cache_path(cache_dir, "vix")

    prices = _read_cache(prices_cache) if use_cache else None
    if prices is None or list(prices.columns) != list(tickers):
        prices = _download_prices(tickers, start, end)
        _write_cache(prices, prices_cache)

    vix_df = _read_cache(vix_cache) if use_cache else None
    if vix_df is None:
        try:
            vix_df = _download_prices([vix_ticker], start, end)
            vix_df.columns = ["VIX"]
        except Exception:  # pragma: no cover
            vix_df = pd.DataFrame(index=prices.index, data={"VIX": np.nan})
        _write_cache(vix_df, vix_cache)
    vix = vix_df.iloc[:, 0].rename("VIX")

    macro = _read_cache(macro_cache) if use_cache else None
    if macro is None:
        macro = _download_macro(fred_series, start, end)
        _write_cache(macro, macro_cache)

    returns = prices.pct_change()
    data = MarketData(prices=prices, returns=returns, vix=vix, macro=macro)
    return data.align()


# --------------------------------------------------------------------------- #
# Synthetic data (offline / CI)
# --------------------------------------------------------------------------- #
def make_synthetic_market_data(
    tickers: Optional[Sequence[str]] = None,
    n_days: int = 2500,
    start: str = "2012-01-02",
    seed: int = 42,
) -> MarketData:
    """Generate a regime-switching synthetic market with no network access.

    The generator alternates between a calm "bull" regime (positive drift, low
    volatility, low VIX) and a turbulent "bear" regime (negative drift, high
    volatility, high VIX). This gives the HMM a genuine signal to find and lets
    the whole pipeline be exercised deterministically.
    """
    tickers = list(tickers or ["SPY", "QQQ", "EFA", "TLT", "IEF", "GLD", "DBC"])
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)

    # Two regimes: 0 = calm/bull, 1 = turbulent/bear.
    # Per-asset daily drift and vol under each regime.
    base_drift = rng.uniform(0.0002, 0.0006, size=n_assets)
    base_vol = rng.uniform(0.008, 0.016, size=n_assets)
    # Bonds (TLT/IEF) behave defensively: they catch a bid in the bear regime.
    defensive = np.array([1 if t in ("TLT", "IEF", "GLD") else 0 for t in tickers])

    drift = {
        0: base_drift,
        1: -base_drift * 1.5 + defensive * 0.0008,
    }
    vol = {
        0: base_vol,
        1: base_vol * 2.2,
    }

    # Regime persistence via a simple Markov chain.
    trans = np.array([[0.98, 0.02], [0.05, 0.95]])
    regimes = np.empty(n_days, dtype=int)
    regimes[0] = 0
    for t in range(1, n_days):
        regimes[t] = rng.choice(2, p=trans[regimes[t - 1]])

    # Correlated shocks; correlation rises in the bear regime.
    rets = np.empty((n_days, n_assets))
    for t in range(n_days):
        r = regimes[t]
        corr = 0.3 if r == 0 else 0.6
        cov = np.full((n_assets, n_assets), corr)
        np.fill_diagonal(cov, 1.0)
        shock = rng.multivariate_normal(np.zeros(n_assets), cov)
        rets[t] = drift[r] + vol[r] * shock

    dates = pd.bdate_range(start=start, periods=n_days)
    returns = pd.DataFrame(rets, index=dates, columns=tickers)
    prices = 100.0 * (1.0 + returns).cumprod()

    # Synthetic VIX: high in the bear regime, mean-reverting noise around it.
    vix_level = np.where(regimes == 0, 14.0, 32.0) + rng.normal(0, 2.0, n_days)
    vix = pd.Series(np.clip(vix_level, 9.0, 80.0), index=dates, name="VIX")

    # Synthetic macro: term spread inverts (goes negative) before/within stress.
    term_spread = np.where(regimes == 0, 1.2, -0.3) + rng.normal(0, 0.15, n_days)
    dgs10 = 2.5 + rng.normal(0, 0.1, n_days).cumsum() * 0.0
    credit = np.where(regimes == 0, 1.8, 3.2) + rng.normal(0, 0.1, n_days)
    macro = pd.DataFrame(
        {"T10Y2Y": term_spread, "DGS10": dgs10, "BAA10Y": credit}, index=dates
    )

    data = MarketData(prices=prices, returns=returns, vix=vix, macro=macro)
    return data.align()
