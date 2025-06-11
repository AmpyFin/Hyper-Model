"""
VAR Agent
=========

20-period **variance** of close (ddof=0).

Features
--------
* var
* var slope (Δvar)
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


class VAR_Agent:
    def __init__(self, period: int = 20):
        self.n = period
        self.m = LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        var = df["close"].rolling(self.n).var(ddof=0)
        d = df.copy()
        d["var"] = var
        d["var_slope"] = var.diff()
        return d.dropna(subset=["var", "var_slope"])

    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 5:
            raise ValueError("Not enough rows for VAR_Agent")
        X, y = d[["var", "var_slope"]][:-1], (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted=True

    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["var", "var_slope"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
