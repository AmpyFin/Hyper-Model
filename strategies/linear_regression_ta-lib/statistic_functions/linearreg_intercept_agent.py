"""
LINEARREG_INTERCEPT Agent
=========================

30-period **linear-regression intercept** a (y = a + b·t).

Features
--------
* intercept (a)
* intercept slope (Δa)
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression


def _lr_intercept(series: pd.Series, n: int = 30) -> pd.Series:
    t = np.arange(n)
    def icpt(window):
        y = window.values
        b, a = np.polyfit(t, y, 1)  # slope, intercept
        return a
    return series.rolling(n).apply(icpt, raw=False)


class LINEARREGINTERCEPT_Agent:
    def __init__(self, period: int = 30):
        self.n = period
        self.m = LogisticRegression(max_iter=1000); self.fitted = False

    def _feat(self, df):
        a = _lr_intercept(df["close"], self.n)
        d = df.copy()
        d["icpt"] = a
        d["icpt_slope"] = a.diff()
        return d.dropna(subset=["icpt", "icpt_slope"])

    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows for LINEARREG_INTERCEPT")
        X, y = d[["icpt", "icpt_slope"]][:-1], (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["icpt", "icpt_slope"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
