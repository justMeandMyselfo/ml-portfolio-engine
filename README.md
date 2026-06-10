# ML-Enhanced Portfolio Optimization Engine

An end-to-end quantitative pipeline that **compares classical portfolio theory against
modern machine-learning frameworks**. It starts from a textbook Markowitz mean-variance
optimizer and progressively replaces its weakest assumptions with data-driven components:

| Limitation of classical theory | This project's fix |
| --- | --- |
| Markowitz assumes **constant volatility** | A **Hidden Markov Model** detects market *regimes* (calm vs. turbulent) from VIX, yield-curve and return features, and rescales risk dynamically. |
| Black-Litterman needs **subjective analyst views** | A **Random Forest** forecasts next-period asset returns from technical + macro features; those forecasts become the model's *views*. |
| Backtests routinely **leak the future** | A strict **walk-forward** engine with transaction costs, no look-ahead, and proper risk-adjusted metrics against real baselines. |

> The goal is not to claim a magic money machine. It is to demonstrate, rigorously and
> reproducibly, *where* machine learning helps a classical allocator and *where it does not* —
> the exact distinction a quant interviewer cares about.

---

## What's inside

```
src/mlportfolio/
  data/         price + VIX (yfinance) and macro (FRED) loaders, feature engineering
  regime/       Gaussian HMM regime detector + regime-conditional risk scaling
  forecast/     Random Forest next-period return forecaster (walk-forward safe)
  optimize/     classic Markowitz baseline + Black-Litterman with ML views
  backtest/     walk-forward engine, transaction costs, performance metrics
  pipeline.py   orchestrates the full comparison
scripts/run_backtest.py   command-line entry point
tests/                    unit tests (run fully offline)
```

## Strategies compared

1. **Equal-weight (1/N)** — the humbling baseline that beats most clever models.
2. **60/40** — classic stock/bond split (when bond proxy present).
3. **Markowitz max-Sharpe** — classical mean-variance using rolling historical inputs.
4. **Black-Litterman + RF views** — ML return forecasts injected as views.
5. **Regime-aware BL** — the above, with HMM-driven risk scaling that de-risks in
   high-volatility regimes.

Every strategy is run through the **same** walk-forward loop so the comparison is apples-to-apples.

## Reproducibility & honesty notes

- All randomness is seeded (`config.yaml: seed`).
- The Random Forest and HMM are **refit only on data available up to each rebalance date** —
  no future information leaks into any decision.
- Transaction costs are charged on turnover at every rebalance.
- Synthetic mode lets the test suite and CI run without network access.


## Roadmap (deliberately not built yet)

- Deep Reinforcement Learning allocator (Stable-Baselines3) as an experimental module.
- LSTM forecaster as an alternative to the Random Forest.

These were scoped out of the first build on purpose: they add the most complexity and the
least reliability. The current pipeline is the solid, defensible core.

## Forward-looking recommendation tool

Beyond backtesting, the engine can propose a **current allocation** from data up to today:

```bash
python scripts/recommend.py --synthetic                          # offline demo
python scripts/recommend.py --holdings "SPY:0.4,TLT:0.3,GLD:0.1" --report
```

It prints (and optionally saves) the detected market regime, the model's shrunk
return forecasts, a volatility-targeted target allocation with a cash buffer, and
the implied buy/sell trades versus your current holdings. The same view is the
**Live recommendation** tab of `streamlit run app.py`.

> **Not financial advice.** This is a research/education model. It cannot predict
> markets and offers no performance guarantee. Treat its output as one input among
> many; investment decisions and their consequences are your own.

## What changed in the upgrade

- **Volatility targeting + cash buffer** — model portfolios are scaled to a target
  annualized volatility, parking the remainder in cash when markets turn turbulent.
  Cuts drawdowns sharply.
- **Turnover control** — partial rebalancing plus a no-trade band stop transaction
  costs from eroding the edge (turnover fell from ~1.0 to ~0.4 in testing).
- **Stronger forecaster** — optional gradient boosting, extra technical features,
  and forecast shrinkage toward zero for stability.
- **Bigger universe** — 13 multi-asset ETFs spanning equity, credit, rates, TIPS,
  gold, commodities and REITs.

## License

MIT — see `LICENSE`.
