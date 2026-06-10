"""Forward-looking allocation recommendation.

This turns the backtested engine into a tool you can point at *today's* data: it
fits the regime model and the return forecaster on everything available up to the
latest date and emits a current target allocation, the cash buffer, the estimated
portfolio volatility, the detected market regime, and — if you supply your current
holdings — the implied trades to get there.

IMPORTANT: this is a research/education model, not financial advice. It cannot
predict the market and offers no performance guarantee. Treat the output as one
input among many; the investment decision and its consequences are yours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import Config
from .pipeline import Pipeline
from .optimize.risk import portfolio_volatility

DISCLAIMER = (
    "Research/education model only - NOT financial advice. No guarantee of future "
    "performance. Markets are uncertain; you are responsible for your own decisions."
)


@dataclass
class Recommendation:
    asof: pd.Timestamp
    regime: Optional[int]
    regime_proba: Optional[pd.Series]
    risk_multiplier: Optional[float]
    expected_returns: Optional[pd.Series]
    target_weights: pd.Series          # risky weights (may sum < 1)
    cash_weight: float
    est_volatility: float
    trades: Optional[pd.DataFrame] = None
    disclaimer: str = field(default=DISCLAIMER)


def recommend(
    config: Config,
    synthetic: bool = False,
    current_holdings: Optional[Dict[str, float]] = None,
    regime_aware: bool = True,
    pipe: Optional[Pipeline] = None,
) -> Recommendation:
    """Produce a current allocation recommendation from data up to the latest date."""
    if pipe is None:
        pipe = Pipeline(config, synthetic=synthetic).load()

    asof = pipe.data.returns.index[-1]
    cov = pipe._cov_at(asof)

    res = pipe.regime_at(asof) if regime_aware else None
    regime = int(res.states.iloc[-1]) if res is not None else None
    proba = res.proba.iloc[-1] if res is not None else None
    risk_mult = float(res.risk_multiplier.iloc[-1]) if res is not None else None

    _, forecast = pipe.forecast_at(asof)
    expected = forecast.expected_returns if forecast is not None else None

    weights = pipe.bl_weights(pipe.data.slice(asof), asof, regime_aware).clip(lower=0.0)
    cash = float(max(0.0, 1.0 - weights.sum()))
    vol = portfolio_volatility(weights, cov)

    trades = None
    if current_holdings:
        cur = pd.Series(current_holdings, dtype=float).reindex(weights.index).fillna(0.0)
        delta = weights - cur
        trades = pd.DataFrame({"current": cur, "target": weights, "trade": delta})
        trades = trades[trades["trade"].abs() > 1e-4].sort_values("trade")

    return Recommendation(
        asof=asof,
        regime=regime,
        regime_proba=proba,
        risk_multiplier=risk_mult,
        expected_returns=expected,
        target_weights=weights,
        cash_weight=cash,
        est_volatility=vol,
        trades=trades,
    )


def format_report(rec: Recommendation, top_n: int = 20) -> str:
    """Render a recommendation as a readable markdown report."""
    lines = []
    lines.append(f"# Allocation recommendation — as of {rec.asof.date()}\n")
    lines.append(f"> {rec.disclaimer}\n")

    if rec.regime is not None:
        label = "calm / low-volatility" if rec.regime == 0 else "turbulent / high-volatility"
        lines.append("## Market regime\n")
        lines.append(f"- Detected regime: **{rec.regime}** ({label})")
        if rec.regime_proba is not None:
            probs = ", ".join(f"{c}={p:.0%}" for c, p in rec.regime_proba.items())
            lines.append(f"- Regime probabilities: {probs}")
        if rec.risk_multiplier is not None:
            lines.append(f"- Risk-aversion multiplier applied: {rec.risk_multiplier:.2f}")
        lines.append("")

    lines.append("## Recommended allocation\n")
    w = rec.target_weights[rec.target_weights > 1e-4].sort_values(ascending=False)
    lines.append("| Asset | Weight |")
    lines.append("| --- | ---: |")
    for asset, wt in w.items():
        lines.append(f"| {asset} | {wt:.1%} |")
    lines.append(f"| Cash | {rec.cash_weight:.1%} |")
    lines.append("")
    lines.append(f"- Estimated annualized volatility: **{rec.est_volatility:.1%}**\n")

    if rec.expected_returns is not None:
        lines.append("## Model return forecasts (next ~1 month, shrunk)\n")
        ex = rec.expected_returns.sort_values(ascending=False).head(top_n)
        lines.append("| Asset | Forecast |")
        lines.append("| --- | ---: |")
        for asset, r in ex.items():
            lines.append(f"| {asset} | {r:+.2%} |")
        lines.append("")

    if rec.trades is not None and not rec.trades.empty:
        lines.append("## Implied trades vs your current holdings\n")
        lines.append("| Asset | Current | Target | Trade |")
        lines.append("| --- | ---: | ---: | ---: |")
        for asset, row in rec.trades.iterrows():
            verb = "BUY" if row["trade"] > 0 else "SELL"
            lines.append(
                f"| {asset} | {row['current']:.1%} | {row['target']:.1%} | "
                f"{verb} {abs(row['trade']):.1%} |"
            )
        lines.append("")

    lines.append(f"\n*{rec.disclaimer}*\n")
    return "\n".join(lines)
