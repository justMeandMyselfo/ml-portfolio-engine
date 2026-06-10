"""End-to-end orchestration with risk overlays and reusable recommendation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .data.loaders import MarketData, load_market_data, make_synthetic_market_data
from .data.features import build_feature_panel, build_market_features
from .regime.hmm import RegimeModel, RegimeResult
from .forecast.random_forest import RandomForestForecaster, ForecastResult
from .optimize.covariance import ledoit_wolf_covariance
from .optimize.markowitz import equal_weights, sixty_forty_weights, max_sharpe_weights
from .optimize.black_litterman import black_litterman_weights
from .optimize.risk import volatility_target
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

    def load(self) -> "Pipeline":
        if self.synthetic:
            self.data = make_synthetic_market_data(tickers=self.cfg.tickers, seed=self.cfg.seed)
        else:
            self.data = load_market_data(
                tickers=self.cfg.tickers,
                start=self.cfg.get("dates", "start", default="2010-01-01"),
                end=self.cfg.get("dates", "end", default="2024-12-31"),
                vix_ticker=self.cfg.get("data", "vix_ticker", default="^VIX"),
                fred_series=self.cfg.get("data", "fred_series", default=[]),
                cache_dir=self.cfg.get("data", "cache_dir", default="data_cache"),
            )
        self.panels = build_feature_panel(self.data, self.momentum_windows, self.vol_window, self.ma_windows)
        self.market_features = build_market_features(self.data, self.momentum_windows, self.vol_window, self.ma_windows)
        return self

    def _cov_at(self, asof: pd.Timestamp) -> pd.DataFrame:
        lookback = self.cfg.get("optimize", "lookback_days", default=252)
        window = self.data.returns.loc[:asof].tail(lookback)
        return ledoit_wolf_covariance(window)

    def regime_at(self, asof: pd.Timestamp) -> Optional[RegimeResult]:
        if self.market_features is None:
            return None
        feats = self.market_features.loc[:asof]
        if len(feats) <= 60:
            return None
        rm = RegimeModel(
            n_regimes=self.cfg.get("regime", "n_regimes", default=2),
            covariance_type=self.cfg.get("regime", "covariance_type", default="full"),
            n_iter=self.cfg.get("regime", "n_iter", default=200),
            risk_scaling=self.cfg.get("regime", "risk_scaling", default=[1.0, 2.5]),
            seed=self.cfg.seed,
        )
        try:
            return rm.fit_predict(feats, vol_col="mkt_vol")
        except Exception:
            return None

    def forecast_at(self, asof: pd.Timestamp) -> Tuple[RandomForestForecaster, Optional[ForecastResult]]:
        rf = RandomForestForecaster(
            horizon=21,
            n_estimators=self.cfg.get("forecast", "n_estimators", default=300),
            max_depth=self.cfg.get("forecast", "max_depth", default=5),
            min_samples_leaf=self.cfg.get("forecast", "min_samples_leaf", default=20),
            model=self.cfg.get("forecast", "model", default="random_forest"),
            shrinkage=self.cfg.get("forecast", "shrinkage", default=1.0),
            seed=self.cfg.seed,
        )
        rf.fit(self.panels, self.data.prices, cutoff=asof)
        return rf, rf.predict(self.panels, asof=asof)

    def _apply_vol_target(self, weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
        if not self.cfg.get("optimize", "vol_targeting", default=False):
            return weights
        return volatility_target(
            weights, cov,
            target_vol=self.cfg.get("optimize", "target_vol", default=0.10),
            max_leverage=self.cfg.get("optimize", "max_leverage", default=1.0),
        )

    def bl_weights(self, data_slice: MarketData, asof: pd.Timestamp, regime_aware: bool) -> pd.Series:
        cov = self._cov_at(asof)
        minw = self.cfg.get("optimize", "min_weight", default=0.0)
        maxw = self.cfg.get("optimize", "max_weight", default=0.4)
        short = self.cfg.get("optimize", "allow_short", default=False)
        base_ra = self.cfg.get("optimize", "risk_aversion", default=3.0)
        tau = self.cfg.get("optimize", "bl_tau", default=0.05)
        tickers = self.data.tickers
        mkt_w = equal_weights(tickers)

        risk_aversion = base_ra
        if regime_aware:
            res = self.regime_at(asof)
            if res is not None:
                risk_aversion = base_ra * float(res.risk_multiplier.iloc[-1])

        rf, forecast = self.forecast_at(asof)
        if forecast is None:
            mu = data_slice.returns.tail(252).mean() * 252
            w = max_sharpe_weights(mu, cov, min_weight=minw, max_weight=maxw, allow_short=short)
            return self._apply_vol_target(w, cov)

        self._rf_rmse[str(asof.date())] = rf.in_sample_rmse
        w = black_litterman_weights(
            cov=cov, market_weights=mkt_w,
            view_returns=forecast.expected_returns, view_confidence=forecast.confidence,
            risk_aversion=risk_aversion, tau=tau,
            min_weight=minw, max_weight=maxw, allow_short=short,
        )
        return self._apply_vol_target(w, cov)

    def strat_equal_weight(self):
        tickers = self.data.tickers
        return lambda _d, _a: equal_weights(tickers)

    def strat_sixty_forty(self):
        tickers = self.data.tickers
        eq = self.cfg.get("universe", "equity_proxy", default="SPY")
        bd = self.cfg.get("universe", "bond_proxy", default="TLT")
        return lambda _d, _a: sixty_forty_weights(tickers, eq, bd)

    def strat_markowitz(self):
        minw = self.cfg.get("optimize", "min_weight", default=0.0)
        maxw = self.cfg.get("optimize", "max_weight", default=0.4)
        short = self.cfg.get("optimize", "allow_short", default=False)
        lookback = self.cfg.get("optimize", "lookback_days", default=252)

        def fn(data_slice, asof):
            cov = self._cov_at(asof)
            mu = data_slice.returns.tail(lookback).mean() * 252
            w = max_sharpe_weights(mu, cov, min_weight=minw, max_weight=maxw, allow_short=short)
            return self._apply_vol_target(w, cov)

        return fn

    def strat_black_litterman(self, regime_aware: bool):
        return lambda data_slice, asof: self.bl_weights(data_slice, asof, regime_aware)

    def run(self) -> PipelineArtifacts:
        if self.data is None:
            self.load()
        bt = WalkForwardBacktester(
            rebalance=self.cfg.get("optimize", "rebalance", default="M"),
            transaction_cost_bps=self.cfg.get("backtest", "transaction_cost_bps", default=10),
            warmup_days=self.cfg.get("backtest", "warmup_days", default=504),
            smoothing=self.cfg.get("backtest", "smoothing", default=1.0),
            no_trade_band=self.cfg.get("backtest", "no_trade_band", default=0.0),
            risk_free_annual=self.cfg.get("backtest", "risk_free_annual", default=0.0),
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
    cols = ["total_return", "ann_return", "ann_vol", "sharpe", "sortino",
            "max_drawdown", "calmar", "avg_turnover", "avg_cash"]
    return df[[c for c in cols if c in df.columns]].round(4)
