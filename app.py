"""Interactive Streamlit dashboard for the ML portfolio engine.

    pip install -r requirements.txt
    streamlit run app.py

Two tools in one:
  * a walk-forward backtest comparing all strategies under your chosen settings;
  * a forward-looking allocation recommendation (research/education only).
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
from mlportfolio.recommend import recommend as make_recommendation, format_report


def _cfg_from(params: dict) -> Config:
    overrides = {
        "seed": params["seed"],
        "optimize.rebalance": params["rebalance"],
        "optimize.risk_aversion": params["risk_aversion"],
        "optimize.max_weight": params["max_weight"],
        "optimize.target_vol": params["target_vol"],
        "optimize.vol_targeting": params["vol_targeting"],
        "backtest.transaction_cost_bps": params["cost_bps"],
        "backtest.smoothing": params["smoothing"],
        "forecast.n_estimators": params["n_estimators"],
        "forecast.model": params["model"],
    }
    if not params["synthetic"]:
        overrides["dates.start"] = params["start"]
        overrides["dates.end"] = params["end"]
        overrides["universe.tickers"] = list(params["tickers"])
    return Config.default().override(overrides)


def _trim(pipe, years):
    if not years:
        return pipe
    keep = int(years * 252)
    if len(pipe.data.returns) > keep:
        start_idx = pipe.data.returns.index[-keep]
        pipe.data = type(pipe.data)(
            prices=pipe.data.prices.loc[start_idx:],
            returns=pipe.data.returns.loc[start_idx:],
            vix=pipe.data.vix.loc[start_idx:],
            macro=pipe.data.macro.loc[start_idx:],
        )
        pipe.panels = build_feature_panel(pipe.data, pipe.momentum_windows, pipe.vol_window, pipe.ma_windows)
        pipe.market_features = build_market_features(pipe.data, pipe.momentum_windows, pipe.vol_window, pipe.ma_windows)
    return pipe


def compute_backtest(params: dict) -> dict:
    cfg = _cfg_from(params)
    pipe = _trim(Pipeline(cfg, synthetic=params["synthetic"]).load(), params.get("years"))
    artifacts = pipe.run()
    equity = pd.DataFrame({n: r.equity_curve for n, r in artifacts.results.items()})
    table = summary_table(artifacts)
    regime_df = None
    feats = pipe.market_features
    if feats is not None and len(feats) > 80:
        rm = RegimeModel(n_regimes=cfg.get("regime", "n_regimes", default=2),
                         n_iter=cfg.get("regime", "n_iter", default=120),
                         risk_scaling=cfg.get("regime", "risk_scaling", default=[1.0, 2.5]),
                         seed=params["seed"])
        res = rm.fit_predict(feats, vol_col="mkt_vol")
        regime_df = pd.DataFrame({"regime": res.states, "vix": feats["vix"]})
    weights = {n: r.weights for n, r in artifacts.results.items()}
    return {"equity": equity, "table": table, "regime": regime_df, "weights": weights,
            "rmse": artifacts.rmse["rf_avg_in_sample_rmse"], "tickers": pipe.data.tickers,
            "span": (pipe.data.returns.index[0], pipe.data.returns.index[-1])}


def compute_recommendation(params: dict, holdings: dict | None) -> dict:
    cfg = _cfg_from(params)
    rec = make_recommendation(cfg, synthetic=params["synthetic"], current_holdings=holdings)
    return {
        "asof": rec.asof, "regime": rec.regime, "proba": rec.regime_proba,
        "risk_mult": rec.risk_multiplier, "weights": rec.target_weights,
        "cash": rec.cash_weight, "vol": rec.est_volatility,
        "expected": rec.expected_returns, "trades": rec.trades,
        "report": format_report(rec), "disclaimer": rec.disclaimer,
    }


@st.cache_data(show_spinner=False)
def cached_backtest(params_key: tuple) -> dict:
    return compute_backtest(dict(params_key))


@st.cache_data(show_spinner=False)
def cached_reco(params_key: tuple, holdings_key: tuple) -> dict:
    return compute_recommendation(dict(params_key), dict(holdings_key) if holdings_key else None)


def _parse_holdings(text: str):
    out = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        tic, _, wt = part.partition(":")
        try:
            out[tic.strip().upper()] = float(wt)
        except ValueError:
            continue
    return out


def main() -> None:
    st.set_page_config(page_title="ML Portfolio Engine", layout="wide")
    st.title("ML-Enhanced Portfolio Optimization Engine")
    st.caption("Classical Markowitz vs. HMM regime-switching and Random Forest "
               "Black-Litterman, with volatility targeting and turnover control.")
    st.warning("Research / education tool only. Not financial advice and no guarantee "
               "of performance. Treat any allocation as one input among many.")

    with st.sidebar:
        st.header("Settings")
        synthetic = st.toggle("Use synthetic data (offline)", value=True)
        tickers = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "TIP", "GLD", "DBC", "VNQ"]
        start, end = "2010-01-01", "2024-12-31"
        if not synthetic:
            tickers = st.multiselect("Tickers", tickers, default=tickers)
            start = str(st.date_input("Start", value=pd.Timestamp("2014-01-01")))
            end = str(st.date_input("End", value=pd.Timestamp("2024-12-31")))
        years = st.slider("Years of history", 2, 12, 5)
        rebalance = st.selectbox("Rebalance", ["M", "Q"], index=0,
                                 format_func=lambda x: {"M": "Monthly", "Q": "Quarterly"}[x])
        model = st.selectbox("Forecaster", ["random_forest", "gradient_boosting"], index=0)
        risk_aversion = st.slider("Risk aversion", 1.0, 10.0, 3.0, 0.5)
        max_weight = st.slider("Max weight / asset", 0.10, 1.0, 0.40, 0.05)
        vol_targeting = st.toggle("Volatility targeting", value=True)
        target_vol = st.slider("Target volatility", 0.04, 0.25, 0.10, 0.01)
        smoothing = st.slider("Rebalance smoothing", 0.1, 1.0, 0.6, 0.1,
                              help="Lower = trade less (turnover control).")
        cost_bps = st.slider("Transaction cost (bps)", 0, 50, 10, 1)
        n_estimators = st.slider("Trees / iterations", 50, 400, 150, 50)
        seed = int(st.number_input("Seed", value=42, step=1))

    params = dict(synthetic=synthetic, tickers=tuple(tickers), start=start, end=end,
                  years=years, rebalance=rebalance, model=model, risk_aversion=risk_aversion,
                  max_weight=max_weight, vol_targeting=vol_targeting, target_vol=target_vol,
                  smoothing=smoothing, cost_bps=cost_bps, n_estimators=n_estimators, seed=seed)
    params_key = tuple(sorted(params.items()))

    tab_reco, tab_bt = st.tabs(["Live recommendation", "Backtest comparison"])

    with tab_reco:
        st.subheader("Current allocation recommendation")
        st.caption("Fits the models on all data up to the latest date and proposes a "
                   "target allocation. Enter your current holdings to see the trades.")
        holdings_text = st.text_input("Current holdings (optional)", value="",
                                      placeholder="SPY:0.4, TLT:0.3, GLD:0.1")
        if st.button("Get recommendation", type="primary"):
            holdings = _parse_holdings(holdings_text)
            with st.spinner("Fitting models on latest data..."):
                out = cached_reco(params_key, tuple(sorted(holdings.items())) if holdings else ())
            m1, m2, m3 = st.columns(3)
            reg_label = {0: "calm", 1: "turbulent"}.get(out["regime"], "n/a")
            m1.metric("Detected regime", f"{out['regime']} ({reg_label})" if out["regime"] is not None else "n/a")
            m2.metric("Estimated volatility", f"{out['vol']:.1%}")
            m3.metric("Cash buffer", f"{out['cash']:.1%}")
            w = out["weights"][out["weights"] > 1e-4].sort_values(ascending=False)
            st.write("**Recommended weights**")
            st.bar_chart(w, height=260)
            if out["trades"] is not None and not out["trades"].empty:
                st.write("**Implied trades vs your holdings**")
                st.dataframe(out["trades"].style.format("{:.2%}"), use_container_width=True)
            with st.expander("Full report (markdown)"):
                st.markdown(out["report"])

    with tab_bt:
        st.subheader("Walk-forward backtest")
        if st.button("Run backtest", type="primary"):
            with st.spinner("Running walk-forward backtest across strategies..."):
                out = cached_backtest(params_key)
            span = out["span"]
            st.success(f"{len(out['tickers'])} assets · {span[0].date()} → {span[1].date()}")
            table = out["table"]
            best_ml = max(table.loc["BlackLitterman_RF", "sharpe"], table.loc["RegimeAware_BL_RF", "sharpe"])
            mkw = table.loc["Markowitz", "sharpe"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Markowitz Sharpe", f"{mkw:.2f}")
            c2.metric("Best ML Sharpe", f"{best_ml:.2f}", delta=f"{(best_ml-mkw)/abs(mkw)*100:+.1f}% vs Markowitz")
            c3.metric("Equal-weight Sharpe", f"{table.loc['EqualWeight','sharpe']:.2f}")
            c4.metric("RF in-sample RMSE", f"{out['rmse']:.4f}")
            st.write("**Equity curves (growth of $1, net of costs)**")
            st.line_chart(out["equity"], height=380)
            st.write("**Performance & risk metrics**")
            st.dataframe(table.style.format("{:.4f}"), use_container_width=True)
            if out["regime"] is not None:
                st.write("**Detected market regimes vs VIX**")
                col_a, col_b = st.columns(2)
                col_a.line_chart(out["regime"][["vix"]], height=220)
                col_b.area_chart(out["regime"][["regime"]], height=220)
            st.write("**Allocation over time**")
            choice = st.selectbox("Strategy", list(out["weights"].keys()), index=4)
            w = out["weights"][choice]
            if w is not None and not w.empty:
                st.area_chart(w, height=300)


if __name__ == "__main__":
    main()
