"""Interactive Streamlit dashboard for the ML portfolio engine.

Run it with::

    pip install -r requirements.txt        # installs streamlit too
    streamlit run app.py

The dashboard lets you change the inputs that matter — risk aversion,
transaction costs, position caps, rebalance frequency, the Random Forest size,
and the data window — and re-runs the full walk-forward comparison live, then
visualizes equity curves, risk-adjusted metrics, the detected market regimes,
and how each strategy allocates over time.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mlportfolio.config import Config
from mlportfolio.pipeline import Pipeline, summary_table
from mlportfolio.data.features import build_feature_panel, build_market_features
from mlportfolio.regime.hmm import RegimeModel


def compute_backtest(params: dict) -> dict:
    """Run the pipeline for a given parameter set and return plot-ready frames."""
    overrides = {
        "seed": params["seed"],
        "optimize.rebalance": params["rebalance"],
        "optimize.risk_aversion": params["risk_aversion"],
        "optimize.max_weight": params["max_weight"],
        "backtest.transaction_cost_bps": params["cost_bps"],
        "forecast.n_estimators": params["n_estimators"],
    }
    if not params["synthetic"]:
        overrides["dates.start"] = params["start"]
        overrides["dates.end"] = params["end"]
        overrides["universe.tickers"] = params["tickers"]

    cfg = Config.default().override(overrides)
    pipe = Pipeline(cfg, synthetic=params["synthetic"]).load()

    years = params.get("years")
    if years:
        keep = int(years * 252)
        if len(pipe.data.returns) > keep:
            start_idx = pipe.data.returns.index[-keep]
            pipe.data = type(pipe.data)(
                prices=pipe.data.prices.loc[start_idx:],
                returns=pipe.data.returns.loc[start_idx:],
                vix=pipe.data.vix.loc[start_idx:],
                macro=pipe.data.macro.loc[start_idx:],
            )
            pipe.panels = build_feature_panel(
                pipe.data, pipe.momentum_windows, pipe.vol_window, pipe.ma_windows
            )
            pipe.market_features = build_market_features(
                pipe.data, pipe.momentum_windows, pipe.vol_window, pipe.ma_windows
            )

    artifacts = pipe.run()

    equity = pd.DataFrame(
        {name: res.equity_curve for name, res in artifacts.results.items()}
    )
    table = summary_table(artifacts)

    regime_df = None
    feats = pipe.market_features
    if feats is not None and len(feats) > 80:
        rm = RegimeModel(
            n_regimes=cfg.get("regime", "n_regimes", default=2),
            n_iter=cfg.get("regime", "n_iter", default=120),
            risk_scaling=cfg.get("regime", "risk_scaling", default=[1.0, 2.5]),
            seed=params["seed"],
        )
        res = rm.fit_predict(feats, vol_col="mkt_vol")
        regime_df = pd.DataFrame(
            {"regime": res.states, "vix": feats["vix"], "risk_multiplier": res.risk_multiplier}
        )

    weights = {name: res.weights for name, res in artifacts.results.items()}

    return {
        "equity": equity,
        "table": table,
        "regime": regime_df,
        "weights": weights,
        "rmse": artifacts.rmse["rf_avg_in_sample_rmse"],
        "tickers": pipe.data.tickers,
        "span": (pipe.data.returns.index[0], pipe.data.returns.index[-1]),
    }


@st.cache_data(show_spinner=False)
def cached_backtest(params_key: tuple) -> dict:
    return compute_backtest(dict(params_key))


def main() -> None:
    st.set_page_config(page_title="ML Portfolio Engine", layout="wide")
    st.title("ML-Enhanced Portfolio Optimization Engine")
    st.caption(
        "Classical Markowitz vs. HMM regime-switching and Random Forest "
        "Black-Litterman, under a strict walk-forward backtest."
    )

    with st.sidebar:
        st.header("Inputs")
        synthetic = st.toggle("Use synthetic data (offline)", value=True)

        tickers = ["SPY", "QQQ", "EFA", "TLT", "IEF", "GLD", "DBC"]
        start, end = "2010-01-01", "2024-12-31"
        if not synthetic:
            st.markdown("**Universe & dates** (downloads from yfinance + FRED)")
            tickers = st.multiselect("Tickers", tickers, default=tickers)
            start = str(st.date_input("Start", value=pd.Timestamp("2014-01-01")).date())
            end = str(st.date_input("End", value=pd.Timestamp("2024-12-31")).date())

        years = st.slider("Years of history to use", 2, 12, 5,
                          help="Trims to the most recent N years to keep runtime reasonable.")
        rebalance = st.selectbox("Rebalance frequency", ["M", "Q"], index=0,
                                 format_func=lambda x: {"M": "Monthly", "Q": "Quarterly"}[x])
        risk_aversion = st.slider("Risk aversion (lambda)", 1.0, 10.0, 3.0, 0.5)
        max_weight = st.slider("Max weight per asset", 0.10, 1.0, 0.40, 0.05)
        cost_bps = st.slider("Transaction cost (bps)", 0, 50, 10, 1)
        n_estimators = st.slider("Random Forest trees", 50, 400, 200, 50)
        seed = st.number_input("Random seed", value=42, step=1)
        run = st.button("Run backtest", type="primary", use_container_width=True)

    if not run:
        st.info("Set your inputs in the sidebar and click **Run backtest**. "
                "Quarterly rebalancing and fewer trees run faster.")
        return

    if not synthetic and not tickers:
        st.error("Pick at least one ticker.")
        return

    params_key = (
        ("synthetic", synthetic), ("tickers", tuple(tickers)),
        ("start", start), ("end", end), ("years", years),
        ("rebalance", rebalance), ("risk_aversion", risk_aversion),
        ("max_weight", max_weight), ("cost_bps", cost_bps),
        ("n_estimators", n_estimators), ("seed", int(seed)),
    )

    with st.spinner("Running walk-forward backtest across all strategies..."):
        out = cached_backtest(params_key)

    span = out["span"]
    st.success(f"Backtest complete · {len(out['tickers'])} assets · "
               f"{span[0].date()} → {span[1].date()}")

    table = out["table"]
    best_ml = max(
        table.loc["BlackLitterman_RF", "sharpe"],
        table.loc["RegimeAware_BL_RF", "sharpe"],
    )
    mkw = table.loc["Markowitz", "sharpe"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Markowitz Sharpe", f"{mkw:.2f}")
    c2.metric("Best ML Sharpe", f"{best_ml:.2f}",
              delta=f"{(best_ml - mkw) / abs(mkw) * 100:+.1f}% vs Markowitz")
    c3.metric("Equal-weight Sharpe", f"{table.loc['EqualWeight', 'sharpe']:.2f}")
    c4.metric("RF in-sample RMSE", f"{out['rmse']:.4f}")

    st.subheader("Equity curves (growth of $1, net of costs)")
    st.line_chart(out["equity"], height=380)

    st.subheader("Performance & risk metrics")
    st.dataframe(table.style.format("{:.4f}"), use_container_width=True)

    if out["regime"] is not None:
        st.subheader("Detected market regimes (0 = calm, higher = turbulent)")
        st.caption("In-sample HMM labels shown for intuition; the backtest "
                   "refits regimes walk-forward with no look-ahead.")
        rg = out["regime"]
        col_a, col_b = st.columns(2)
        with col_a:
            st.line_chart(rg[["vix"]], height=240)
        with col_b:
            st.area_chart(rg[["regime"]], height=240)

    st.subheader("How a strategy allocates over time")
    choice = st.selectbox("Strategy", list(out["weights"].keys()), index=4)
    w = out["weights"][choice]
    if w is not None and not w.empty:
        st.area_chart(w, height=320)
    else:
        st.write("No rebalance weights recorded for this strategy.")

    st.caption("Synthetic results are illustrative. Switch off synthetic data "
               "in the sidebar to run on real market history.")


if __name__ == "__main__":
    main()
