"""Walk-forward backtest engine with cash bucket, turnover control and risk-free cash."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ..data.loaders import MarketData
from ..optimize.risk import apply_turnover_control
from .metrics import performance_summary


def _normalize_freq(freq: str) -> str:
    """Return a resample alias valid for the installed pandas version."""
    from pandas.tseries.frequencies import to_offset

    try:
        to_offset(freq)
        return freq
    except Exception:
        return {
            "M": "ME", "Q": "QE", "Y": "YE", "A": "YE",
            "BM": "BME", "BQ": "BQE", "BA": "BYE",
        }.get(freq, freq)


StrategyFn = Callable[[MarketData, pd.Timestamp], pd.Series]


@dataclass
class BacktestResult:
    returns: pd.Series
    equity_curve: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    cash: pd.Series
    summary: dict

    @property
    def total_return(self) -> float:
        return float(self.equity_curve.iloc[-1] - 1.0) if len(self.equity_curve) else float("nan")


class WalkForwardBacktester:
    def __init__(
        self,
        rebalance: str = "M",
        transaction_cost_bps: float = 10.0,
        warmup_days: int = 504,
        smoothing: float = 1.0,
        no_trade_band: float = 0.0,
        risk_free_annual: float = 0.0,
    ) -> None:
        self.rebalance = rebalance
        self.cost = transaction_cost_bps / 1e4
        self.warmup_days = warmup_days
        self.smoothing = smoothing
        self.no_trade_band = no_trade_band
        self.rf_daily = risk_free_annual / 252.0

    def _rebalance_dates(self, index: pd.DatetimeIndex) -> List[pd.Timestamp]:
        if len(index) == 0:
            return []
        start = index[min(self.warmup_days, len(index) - 1)]
        usable = index[index >= start]
        marks = pd.Series(usable, index=usable).resample(_normalize_freq(self.rebalance)).last()
        dates = [d for d in marks.values if pd.notna(d)]
        return [pd.Timestamp(d) for d in dates]

    def run(self, data: MarketData, strategy: StrategyFn, name: str = "strategy") -> BacktestResult:
        data = data.align()
        returns = data.returns.fillna(0.0)
        index = returns.index
        tickers = list(returns.columns)

        rebal_dates = self._rebalance_dates(index)
        if not rebal_dates:
            raise ValueError("Not enough data for the configured warmup/rebalance.")
        rebal_set = set(rebal_dates)

        weights_log: Dict[pd.Timestamp, pd.Series] = {}
        turnover_log: Dict[pd.Timestamp, float] = {}
        cash_log: Dict[pd.Timestamp, float] = {}

        port_rets = pd.Series(0.0, index=index)
        current_w = pd.Series(0.0, index=tickers)
        cash = 0.0

        prev_date = None
        for date in index:
            if prev_date is not None and (current_w.abs().sum() > 0 or cash > 0):
                day_ret = returns.loc[date]
                gross = float((current_w * day_ret).sum()) + cash * self.rf_daily
                port_rets.loc[date] = gross
                grown = current_w * (1.0 + day_ret)
                cash_grown = cash * (1.0 + self.rf_daily)
                total = float(grown.sum()) + cash_grown
                if total != 0:
                    current_w = grown / total
                    cash = cash_grown / total

            if date in rebal_set:
                target = strategy(data.slice(date), date).reindex(tickers).fillna(0.0)
                target = target.clip(lower=0.0)
                if target.sum() > 1.0:
                    target = target / target.sum()

                new_w = apply_turnover_control(
                    target, current_w, self.smoothing, self.no_trade_band
                ).clip(lower=0.0)
                if new_w.sum() > 1.0:
                    new_w = new_w / new_w.sum()
                new_cash = float(max(0.0, 1.0 - new_w.sum()))

                turnover = float((new_w - current_w).abs().sum()) + abs(new_cash - cash)
                port_rets.loc[date] = port_rets.loc[date] - turnover * self.cost

                current_w = new_w
                cash = new_cash
                weights_log[date] = new_w
                turnover_log[date] = turnover
                cash_log[date] = new_cash

            prev_date = date

        equity = (1.0 + port_rets).cumprod()
        weights_df = pd.DataFrame(weights_log).T.sort_index()
        turnover_s = pd.Series(turnover_log).sort_index()
        cash_s = pd.Series(cash_log).sort_index()

        summary = performance_summary(port_rets)
        summary["name"] = name
        summary["avg_turnover"] = float(turnover_s.mean()) if len(turnover_s) else 0.0
        summary["avg_cash"] = float(cash_s.mean()) if len(cash_s) else 0.0

        return BacktestResult(
            returns=port_rets,
            equity_curve=equity,
            weights=weights_df,
            turnover=turnover_s,
            cash=cash_s,
            summary=summary,
        )
