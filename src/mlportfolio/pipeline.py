"""End-to-end orchestration.

Builds the data + feature panels once (leak-safe), then defines each competing
strategy as a closure that the walk-forward engine can call at every rebalance.
Models (HMM, Random Forest) are **refit at each rebalance using only past data**,
so the comparison is an honest out-of-sample test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .data.loaders import MarketData, load_market_data, make_synthetic_market_data
from .data.features import build_feature_panel, build_market_features
from .regime.hmm import RegimeModel
from .forecast.random_forest import RandomForestForecaster
from .optimize.covariance import ledoit_wolf_covariance
from .optimize.markowitz import (
    equal_weights,
    sixty_forty_weights,
    max_sharpe_weights,
)
from .optimize.black_litterman import black_litterman_weights
from .backtest.engine import WalkForwardBacktester, BacktestResult


@dataclass
class PipelineArtifacts:
    data: MarketData
    results: Dict[str, BacktestResult]
    rmse: Dict[str, float]


class Pipeline:
    def __init__(self, config: Config, synthetic: bool = False) -> None:
        self.cfg = config
        self.synthetic = synthetic
        np.random.seed(self.cfg.seed)

        self.momentum_windows = self.cfg.get("features", "momentum_windows", default=[21, 63, 126])
        self.vol_window = self.cfg.get("features", "vol_window", default=21)
        self.ma_windows = self.cfg.get("features", "ma_windows", default=[50, 200])

        self.data: Optional[MarketData] = None
        self.panels: Dict[str, pd.DataFrame] = {}
        self.market_features: Optional[pd.DataFrame] = None
        self._rf_rmse: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def load(self) -> "Pipeline":
        if self.synthetic:
            self.data = make_synthetic_market_data(
                tickers=self.cfg.tickers, seed=self.cfg.seed
            )
        else:
            self.data = load_market_data(
                tickers=self.cfg.tickers,
                start=self.cfg.get("dates", "start", default="2010-01-01"),
                end=self.cfg.get("dates", "end", default="2024-12-31"),
                vix_ticker=self.cfg.get("data", "vix_ticker", default="^VIX"),
                fred_series=self.cfg.get("data", "fred_series", default=[]),
                cache_dir=self.cfg.get("data", "cache_dir", default="data_cache"),
            )
        self.panels = build_feature_panel(
            self.data, self.momentum_windows, self.vol_window, self.ma_windows
        )
        self.market_features = build_market_features(
            self.data, self.momentum_windows, self.vol_window, self.ma_windows
        )
        return self

    # ------------------------------------------------------------------ #
    # Strategy builders. Each returns a StrategyFn(data_slice, asof) -> weights.
    # ------------------------------------------------------------------ #
    def _cov_at(self, asof: pd.Timestamp) -> pd.DataFrame:
        lookback = self.cfg.get("optimize", "lookback_days", default=252)
        window = self.data.returns.loc[:asof].tail(lookback)
        return ledoit_wolf_covariance(window)

    def strat_equal_weight(self):
        tickers = self.data.tickers

        def fn(_data, _asof):
            return equal_weights(tickers)

        return fn

    def strat_sixty_forty(self):
        tickers = self.data.tickers
        eq = self.cfg.get("universe", "equity_proxy", default="SPY")
        bd = self.cfg.get("universe", "bond_proxy", default="TLT")

        def fn(_data, _asof):
            return sixty_forty_weights(tickers, eq, bd)

        return fn

    def strat_markowitz(self):
        ra = self.cfg.get("optimize", "risk_aversion", default=3.0)
        minw = self.cfg.get("optimize", "min_weight", default=0.0)
        maxw = self.cfg.get("optimize", "max_weight", default=0.4)
        short = self.cfg.get("optimize", "allow_short", default=False)
        lookback = self.cfg.get("optimize", "lookback_days", default=252)

        def fn(data_slice, asof):
            cov = self._cov_at(asof)
            # Classic input: trailing realized mean return, annualized.
            mu = data_slice.returns.tail(lookback).mean() * 252
            return max_sharpe_weights(mu, cov, min_weight=minw, max_weight=maxw, allow_short=short)

        return fn

    def _forecast_at(self, asof: pd.Timestamp):
        horizon = 21
        rf = RandomForestForecaster(
            horizon=horizon,
            n_estimators=self.cfg.get("forecast", "n_estimators", default=300),
            max_depth=self.cfg.get("forecast", "max_depth", default=5),
            min_samples_leaf=self.cfg.get("forecast", "min_samples_leaf", default=20),
            seed=self.cfg.seed,
        )
        rf.fit(self.panels, self.data.prices, cutoff=asof)
        forecast = rf.predict(self.panels, asof=asof)
        return rf, forecast

    def strat_black_litterman(self, regime_aware: bool):
        base_ra = self.cfg.get("optimize", "risk_aversion", default=3.0)
        tau = self.cfg.get("optimize", "bl_tau", default=0.05)
        minw = self.cfg.get("optimize", "min_weight", default=0.0)
        maxw = self.cfg.get("optimize", "max_weight", default=0.4)
        short = self.cfg.get("optimize", "allow_short", default=False)
        tickers = self.data.tickers
        mkt_w = equal_weights(tickers)  # equal-weight market prior

        def fn(data_slice, asof):
            cov = self._cov_at(asof)
            risk_aversion = base_ra
            if regime_aware and self.market_features is not None:
                feats = self.market_features.loc[:asof]
                if len(feats) > 60:
                    rm = RegimeModel(
                        n_regimes=self.cfg.get("regime", "n_regimes", default=2),
                        covariance_type=self.cfg.get("regime", "covariance_type", default="full"),
                        n_iter=self.cfg.get("regime", "n_iter", default=200),
                        risk_scaling=self.cfg.get("regime", "risk_scaling", default=[1.0, 2.5]),
                        seed=self.cfg.seed,
                    )
                    try:
                        res = rm.fit_predict(feats, vol_col="mkt_vol")
                        risk_aversion = base_ra * float(res.risk_multiplier.iloc[-1])
                    except Exception:
                        risk_aversion = base_ra

            rf, forecast = self._forecast_at(asof)
            if forecast is None:
                # Fallback to trailing mean if the forecaster has no signal yet.
                mu = data_slice.returns.tail(252).mean() * 252
                return max_sharpe_weights(mu, cov, min_weight=minw, max_weight=maxw, allow_short=short)

            self._rf_rmse[str(asof.date())] = rf.in_sample_rmse
            views = forecast.expected_returns
            conf = forecast.confidence
            return black_litterman_weights(
                cov=cov,
                market_weights=mkt_w,
                view_returns=views,
                view_confidence=conf,
                risk_aversion=risk_aversion,
                tau=tau,
                min_weight=minw,
                max_weight=maxw,
                allow_short=short,
            )

        return fn

    # ------------------------------------------------------------------ #
    def run(self) -> PipelineArtifacts:
        if self.data is None:
            self.load()

        bt = WalkForwardBacktester(
            rebalance=self.cfg.get("optimize", "rebalance", default="M"),
            transaction_cost_bps=self.cfg.get("backtest", "transaction_cost_bps", default=10),
            warmup_days=self.cfg.get("backtest", "warmup_days", default=504),
        )

        strategies = {
            "EqualWeight": self.strat_equal_weight(),
            "SixtyForty": self.strat_sixty_forty(),
            "Markowitz": self.strat_markowitz(),
            "BlackLitterman_RF": self.strat_black_litterman(regime_aware=False),
            "RegimeAware_BL_RF": self.strat_black_litterman(regime_aware=True),
        }

        results: Dict[str, BacktestResult] = {}
        for name, fn in strategies.items():
            results[name] = bt.run(self.data, fn, name=name)

        avg_rmse = float(np.mean(list(self._rf_rmse.values()))) if self._rf_rmse else float("nan")
        return PipelineArtifacts(data=self.data, results=results, rmse={"rf_avg_in_sample_rmse": avg_rmse})


def summary_table(artifacts: PipelineArtifacts) -> pd.DataFrame:
    rows = []
    for name, res in artifacts.results.items():
        row = dict(res.summary)
        row["total_return"] = res.total_return
        rows.append(row)
    df = pd.DataFrame(rows).set_index("name")
    cols = ["total_return", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "avg_turnover"]
    return df[[c for c in cols if c in df.columns]].round(4)
