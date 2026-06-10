"""ML next-period return forecaster (Random Forest or Gradient Boosting)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


@dataclass
class ForecastResult:
    expected_returns: pd.Series
    confidence: pd.Series


class RandomForestForecaster:
    def __init__(
        self,
        horizon: int = 21,
        n_estimators: int = 300,
        max_depth: Optional[int] = 5,
        min_samples_leaf: int = 20,
        model: str = "random_forest",
        shrinkage: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.horizon = horizon
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = model
        self.shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
        self.seed = seed
        self._model = None
        self._feature_cols: Optional[List[str]] = None
        self._tree_rmse: float = np.nan

    def _make_model(self):
        if self.model == "gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(
                max_depth=self.max_depth,
                max_iter=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                learning_rate=0.05,
                random_state=self.seed,
            )
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
            n_jobs=-1,
        )

    def _build_training_table(self, panels, prices, cutoff):
        rows_X: List[pd.DataFrame] = []
        rows_y: List[pd.Series] = []
        for ticker, feats in panels.items():
            px = prices[ticker]
            fwd = px.shift(-self.horizon) / px - 1.0
            df = feats.copy()
            df["__target__"] = fwd
            eligible = px.index[px.index <= cutoff]
            if len(eligible) <= self.horizon:
                continue
            last_train_date = eligible[-(self.horizon + 1)]
            df = df.loc[df.index <= last_train_date].dropna()
            if df.empty:
                continue
            y = df.pop("__target__")
            rows_X.append(df)
            rows_y.append(y)
        if not rows_X:
            return None, None
        return pd.concat(rows_X, axis=0), pd.concat(rows_y, axis=0)

    def fit(self, panels, prices, cutoff):
        X, y = self._build_training_table(panels, prices, cutoff)
        if X is None or len(X) < 50:
            self._model = None
            return self
        self._feature_cols = list(X.columns)
        self._model = self._make_model()
        self._model.fit(X.to_numpy(), y.to_numpy())
        pred = self._model.predict(X.to_numpy())
        self._tree_rmse = float(np.sqrt(np.mean((pred - y.to_numpy()) ** 2)))
        return self

    def predict(self, panels, asof):
        if self._model is None or self._feature_cols is None:
            return None
        is_rf = isinstance(self._model, RandomForestRegressor)
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
            if is_rf:
                tree_preds = np.array([est.predict(x_arr)[0] for est in self._model.estimators_])
                exp[ticker] = float(tree_preds.mean())
                spread[ticker] = float(tree_preds.std() + 1e-6)
            else:
                exp[ticker] = float(self._model.predict(x_arr)[0])
                spread[ticker] = 1.0
        if not exp:
            return None
        expected = pd.Series(exp, name="expected_return") * self.shrinkage
        conf = pd.Series({k: 1.0 / (v ** 2) for k, v in spread.items()}, name="confidence")
        return ForecastResult(expected_returns=expected, confidence=conf)

    @property
    def in_sample_rmse(self) -> float:
        return self._tree_rmse
