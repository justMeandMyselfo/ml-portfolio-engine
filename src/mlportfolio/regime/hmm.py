"""Hidden Markov Model regime detector.

Fits a Gaussian HMM to market-level features (broad return, broad volatility,
VIX, macro spreads) and labels each day with a latent *regime*. We then map the
raw HMM states to an interpretable ordering (calmest -> most turbulent) using the
average realized volatility within each state, so that "regime 0" always means
"calmest" regardless of how the EM algorithm happened to number its states.

The risk scaling produced here feeds the optimizer: in a turbulent regime we
raise effective risk aversion, pulling the portfolio toward safer allocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeResult:
    """Output of a fitted regime model applied to a feature frame."""

    states: pd.Series          # interpretable regime label per date (0 = calmest)
    proba: pd.DataFrame        # posterior probability of each regime
    risk_multiplier: pd.Series  # optimizer risk-aversion multiplier per date


class RegimeModel:
    """Thin, leak-safe wrapper around :class:`hmmlearn.hmm.GaussianHMM`."""

    def __init__(
        self,
        n_regimes: int = 2,
        covariance_type: str = "full",
        n_iter: int = 200,
        risk_scaling: Optional[List[float]] = None,
        seed: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.risk_scaling = risk_scaling or [1.0, 2.5]
        self.seed = seed

        self._scaler: Optional[StandardScaler] = None
        self._model = None
        self._order: Optional[np.ndarray] = None  # maps raw state -> rank
        self._vol_col: Optional[str] = None

    # ------------------------------------------------------------------ #
    def fit(self, features: pd.DataFrame, vol_col: str = "mkt_vol") -> "RegimeModel":
        from hmmlearn.hmm import GaussianHMM  # lazy import

        self._vol_col = vol_col if vol_col in features.columns else features.columns[0]
        X = features.to_numpy()
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)

        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.seed,
        )
        self._model.fit(Xs)

        # Order raw states by mean (scaled) volatility so labels are interpretable.
        raw_states = self._model.predict(Xs)
        vol_values = features[self._vol_col].to_numpy()
        mean_vol = [
            vol_values[raw_states == s].mean() if np.any(raw_states == s) else np.inf
            for s in range(self.n_regimes)
        ]
        # rank 0 == lowest vol (calmest).
        self._order = np.argsort(np.argsort(mean_vol))
        return self

    # ------------------------------------------------------------------ #
    def predict(self, features: pd.DataFrame) -> RegimeResult:
        if self._model is None or self._scaler is None or self._order is None:
            raise RuntimeError("RegimeModel must be fit before predict().")

        Xs = self._scaler.transform(features.to_numpy())
        raw_states = self._model.predict(Xs)
        raw_proba = self._model.predict_proba(Xs)

        # Remap raw states to interpretable ranks.
        mapped = self._order[raw_states]
        proba_cols = [f"regime_{r}" for r in range(self.n_regimes)]
        proba = np.zeros_like(raw_proba)
        for raw_s in range(self.n_regimes):
            proba[:, self._order[raw_s]] = raw_proba[:, raw_s]

        states = pd.Series(mapped, index=features.index, name="regime")
        proba_df = pd.DataFrame(proba, index=features.index, columns=proba_cols)

        # Risk multiplier = probability-weighted blend of the per-regime scalers,
        # so transitions are smooth rather than a hard step.
        scaling = np.array(self._pad_scaling())
        mult = proba_df.to_numpy() @ scaling
        risk_multiplier = pd.Series(mult, index=features.index, name="risk_multiplier")

        return RegimeResult(states=states, proba=proba_df, risk_multiplier=risk_multiplier)

    def fit_predict(self, features: pd.DataFrame, vol_col: str = "mkt_vol") -> RegimeResult:
        return self.fit(features, vol_col=vol_col).predict(features)

    # ------------------------------------------------------------------ #
    def _pad_scaling(self) -> List[float]:
        """Ensure the risk-scaling vector matches the number of regimes."""
        scaling = list(self.risk_scaling)
        if len(scaling) < self.n_regimes:
            scaling = scaling + [scaling[-1]] * (self.n_regimes - len(scaling))
        return scaling[: self.n_regimes]
