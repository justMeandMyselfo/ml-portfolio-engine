"""Random Forest next-period return forecaster.

The forecaster predicts each asset's return over the *next* rebalance horizon
(e.g. the next ~21 trading days) from technical + macro features. A single
pooled model is trained across all assets (with the asset's own features), which
shares statistical strength and avoids overfitting tiny per-asset samples.

These forecasts become the **views** fed into Black-Litterman, replacing the
classic model's subjective analyst opinions.

Leak safety: :meth:`fit` is only ever called on data strictly before the
rebalance date by the backtest engine, and the training target is shifted so the
label at row ``t`` is the forward return that is only knowable at ``t+horizon``.
Rows whose forward window extends past the training cutoff are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


@dataclass
class ForecastResult:
    """Per-asset expected return for the upcoming horizon."""

    expected_returns: pd.Series        # indexed by ticker, horizon-total return
    confidence: pd.Series              # 1/variance-style confidence per ticker


class RandomForestForecaster:
    def __init__(
        self,
        horizon: int = 21,
        n_estimators: int = 300,
        max_depth: Optional[int] = 5,
        min_samples_leaf: int = 20,
        seed: int = 42,
    ) -> None:
        self.horizon = horizon
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self._model: Optional[RandomForestRegressor] = None
        self._feature_cols: Optional[List[str]] = None
        self._tree_rmse: float = np.nan

    # ------------------------------------------------------------------ #
    def _build_training_table(
        self,
        panels: Dict[str, pd.DataFrame],
        prices: pd.DataFrame,
        cutoff: pd.Timestamp,
    ):
        rows_X: List[pd.DataFrame] = []
        rows_y: List[pd.Series] = []
        for ticker, feats in panels.items():
            px = prices[ticker]
            # Forward horizon return as the supervised target.
            fwd = px.shift(-self.horizon) / px - 1.0
            df = feats.copy()
            df["__target__"] = fwd
            # Strict leak guard: a sample at date t has a target that depends on
            # the price at t+horizon. We may only train on samples whose *entire*
            # forward window has already closed by the cutoff, i.e. the row dated
            # `horizon` business days *before* the cutoff is the most recent we
            # can use. Filtering on ``index <= cutoff`` alone would let labels
            # near the cutoff peek into the future.
            eligible = px.index[px.index <= cutoff]
            if len(eligible) <= self.horizon:
                continue
            last_train_date = eligible[-(self.horizon + 1)]
            df = df.loc[df.index <= last_train_date]
            df = df.dropna()
            if df.empty:
                continue
            y = df.pop("__target__")
            rows_X.append(df)
            rows_y.append(y)
        if not rows_X:
            return None, None
        X = pd.concat(rows_X, axis=0)
        y = pd.concat(rows_y, axis=0)
        return X, y

    # ------------------------------------------------------------------ #
    def fit(
        self,
        panels: Dict[str, pd.DataFrame],
        prices: pd.DataFrame,
        cutoff: pd.Timestamp,
    ) -> "RandomForestForecaster":
        X, y = self._build_training_table(panels, prices, cutoff)
        if X is None or len(X) < 50:
            # Not enough history yet; leave model unfit (caller handles fallback).
            self._model = None
            return self
        self._feature_cols = list(X.columns)
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
            n_jobs=-1,
        )
        self._model.fit(X.to_numpy(), y.to_numpy())
        # In-sample RMSE, used only as a rough confidence scale.
        pred = self._model.predict(X.to_numpy())
        self._tree_rmse = float(np.sqrt(np.mean((pred - y.to_numpy()) ** 2)))
        return self

    # ------------------------------------------------------------------ #
    def predict(
        self,
        panels: Dict[str, pd.DataFrame],
        asof: pd.Timestamp,
    ) -> Optional[ForecastResult]:
        if self._model is None or self._feature_cols is None:
            return None
        exp: Dict[str, float] = {}
        spread: Dict[str, float] = {}
        for ticker, feats in panels.items():
            sub = feats.loc[feats.index <= asof]
            if sub.empty:
                continue
            x = sub.iloc[[-1]].reindex(columns=self._feature_cols)
            if x.isna().any(axis=1).iloc[0]:
                continue
            x_arr = x.to_numpy()
            # Mean prediction across trees + dispersion as a confidence signal.
            tree_preds = np.array(
                [est.predict(x_arr)[0] for est in self._model.estimators_]
            )
            exp[ticker] = float(tree_preds.mean())
            spread[ticker] = float(tree_preds.std() + 1e-6)
        if not exp:
            return None
        expected = pd.Series(exp, name="expected_return")
        # Confidence is inverse to cross-tree dispersion (more agreement => more confident).
        conf = pd.Series({k: 1.0 / (v ** 2) for k, v in spread.items()}, name="confidence")
        return ForecastResult(expected_returns=expected, confidence=conf)

    @property
    def in_sample_rmse(self) -> float:
        return self._tree_rmse
