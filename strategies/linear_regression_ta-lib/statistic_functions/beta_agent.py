"""
BETA Agent
==========

30-period rolling **beta** between *close* and *open*:

    βₜ = Cov(close, open) / Var(open)

Features
--------
* beta
* beta slope  (first diff)

Score ∈ [-1, 1] via LogisticRegression.
"""

from __future__ import annotations
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression


def _beta(close: pd.Series, open_: pd.Series, n: int = 30) -> pd.Series:
    cov = close.rolling(n).cov(open_)
    var = open_.rolling(n).var()
    return cov / var


class BETA_Agent:
    def __init__(self, period: int = 30):
        self.n = period
        self.model = LogisticRegression(max_iter=1000)
        self.fitted = False

    # ------------- feature engineering ------------- #
    def _feat(self, df: pd.DataFrame):
        b = _beta(df["close"], df["open"], self.n)
        d = df.copy()
        d["beta"] = b
        d["beta_slope"] = b.diff()
        return d.dropna(subset=["beta", "beta_slope"])

    # ---------------- training --------------------- #
    def fit(self, df):
        data = self._feat(df)
        if len(data) < self.n + 10:
            raise ValueError("Not enough rows for BETA_Agent")
        X = data[["beta", "beta_slope"]][:-1]
        y = (data["close"].shift(-1) > data["close"]).astype(int)[:-1]
        self.model.fit(X, y)
        self.fitted = True

    # ---------------- predict ---------------------- #
    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["beta", "beta_slope"]]
        prob_up = self.model.predict_proba(last)[0, 1]
        return 2 * prob_up - 1
