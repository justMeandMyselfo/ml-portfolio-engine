"""Walk-forward backtest engine.

Design goals
------------
* **No look-ahead.** At each rebalance date ``t`` a strategy receives a
  :class:`MarketData` slice truncated at ``t`` and must return target weights
  using only that information. The engine then realizes returns *after* ``t``.
* **Realistic frictions.** Turnover at each rebalance is charged a transaction
  cost. Between rebalances, weights drift with realized returns.
* **Apples-to-apples.** Every strategy runs through this identical loop, so
  performance differences come from the allocation logic, not the harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ..data.loaders import MarketData
from .metrics import performance_summary


def _normalize_freq(freq: str) -> str:
    """Return a resample alias valid for the installed pandas version.

    pandas >= 2.2 renamed period-end offset aliases (``M`` -> ``ME``,
    ``Q`` -> ``QE``, ``Y`` -> ``YE`` ...) and pandas 3.0 removed the old forms
    entirely. We accept the legacy alias in config for readability and remap it
    on the fly only when the running pandas rejects it, so a single config works
    across versions.
    """
    from pandas.tseries.frequencies import to_offset

    try:
        to_offset(freq)
        return freq
    except Exception:
        return {
            "M": "ME", "Q": "QE", "Y": "YE", "A": "YE",
            "BM": "BME", "BQ": "BQE", "BA": "BYE",
        }.get(freq, freq)

# A strategy maps (data-up-to-asof, asof_date) -> target weight Series.
StrategyFn = Callable[[MarketData, pd.Timestamp], pd.Series]


@dataclass
class BacktestResult:
    returns: pd.Series           # daily net portfolio returns
    equity_curve: pd.Series      # cumulative growth of $1
    weights: pd.DataFrame        # target weights at each rebalance
    turnover: pd.Series          # turnover at each rebalance
    summary: dict                # headline metrics

    @property
    def total_return(self) -> float:
        return float(self.equity_curve.iloc[-1] - 1.0) if len(self.equity_curve) else float("nan")


class WalkForwardBacktester:
    def __init__(
        self,
        rebalance: str = "M",
        transaction_cost_bps: float = 10.0,
        warmup_days: int = 504,
    ) -> None:
        self.rebalance = rebalance
        self.cost = transaction_cost_bps / 1e4
        self.warmup_days = warmup_days

    # ------------------------------------------------------------------ #
    def _rebalance_dates(self, index: pd.DatetimeIndex) -> List[pd.Timestamp]:
        if len(index) == 0:
            return []
        start = index[min(self.warmup_days, len(index) - 1)]
        usable = index[index >= start]
        # Last trading day of each period that actually exists in the index.
        marks = pd.Series(usable, index=usable).resample(_normalize_freq(self.rebalance)).last()
        dates = [d for d in marks.values if pd.notna(d)]
        return [pd.Timestamp(d) for d in dates]

    # ------------------------------------------------------------------ #
    def run(self, data: MarketData, strategy: StrategyFn, name: str = "strategy") -> BacktestResult:
        data = data.align()
        returns = data.returns.fillna(0.0)
        index = returns.index
        tickers = list(returns.columns)

        rebal_dates = self._rebalance_dates(index)
        if not rebal_dates:
            raise ValueError("Not enough data for the configured warmup/rebalance.")

        weights_log: Dict[pd.Timestamp, pd.Series] = {}
        turnover_log: Dict[pd.Timestamp, float] = {}

        port_rets = pd.Series(0.0, index=index)
        current_w = pd.Series(0.0, index=tickers)  # drifted weights actually held
        rebal_set = set(rebal_dates)
        next_target = None

        # Iterate day by day so weights drift correctly between rebalances.
        prev_date = None
        for date in index:
            # Apply that day's return to currently held weights (drift).
            if prev_date is not None and current_w.abs().sum() > 0:
                day_ret = returns.loc[date]
                gross = float((current_w * day_ret).sum())
                port_rets.loc[date] = gross
                # Drift weights by realized asset growth, renormalize.
                grown = current_w * (1.0 + day_ret)
                total = grown.sum()
                if total != 0:
                    current_w = grown / total

            # Rebalance at the close of rebalance dates (affects the *next* days).
            if date in rebal_set:
                sliced = data.slice(date)
                target = strategy(sliced, date)
                target = target.reindex(tickers).fillna(0.0)
                if target.abs().sum() > 0:
                    target = target / target.sum()
                turnover = float((target - current_w).abs().sum())
                cost = turnover * self.cost
                # Charge cost on the rebalance day's return.
                port_rets.loc[date] = port_rets.loc[date] - cost
                current_w = target
                weights_log[date] = target
                turnover_log[date] = turnover

            prev_date = date

        equity = (1.0 + port_rets).cumprod()
        weights_df = pd.DataFrame(weights_log).T.sort_index()
        turnover_s = pd.Series(turnover_log).sort_index()
        summary = performance_summary(port_rets)
        summary["name"] = name
        summary["avg_turnover"] = float(turnover_s.mean()) if len(turnover_s) else 0.0

        return BacktestResult(
            returns=port_rets,
            equity_curve=equity,
            weights=weights_df,
            turnover=turnover_s,
            summary=summary,
        )
