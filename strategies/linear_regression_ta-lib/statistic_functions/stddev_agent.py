"""
STDDEV Agent
============

20-period **standard deviation** of close.

Features
--------
* stddev
* z-score = (close − ma) / stddev
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


class STDDEV_Agent:
    def __init__(self, period: int = 20):
        self.n = period
        self.m = LogisticRegression(max_iter=1000); self.fitted = False

    def _feat(self, df):
        std = df["close"].rolling(self.n).std(ddof=0)
        ma  = df["close"].rolling(self.n).mean()
        z   = (df["close"] - ma) / std
        d = df.copy()
        d["std"] = std
        d["z"] = z
        return d.dropna(subset=["std", "z"])

    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 5:
            raise ValueError("Not enough rows for STDDEV_Agent")
        X, y = d[["std", "z"]][:-1], (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["std", "z"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
