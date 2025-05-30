"""
LINEARREG_SLOPE Agent
=====================

30-period **linear-regression slope** b.

Features
--------
* slope  (b)
* slope acceleration (Δb)
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression


def _lr_slope(series: pd.Series, n: int = 30) -> pd.Series:
    t = np.arange(n)
    def slope(window):
        y = window.values
        b, _ = np.polyfit(t, y, 1)
        return b
    return series.rolling(n).apply(slope, raw=False)


class LINEARREGSLOPE_Agent:
    def __init__(self, period: int = 30):
        self.n = period
        self.m = LogisticRegression(max_iter=1000); self.fitted = False

    def _feat(self, df):
        b = _lr_slope(df["close"], self.n)
        d = df.copy()
        d["slope"] = b
        d["accel"] = b.diff()
        return d.dropna(subset=["slope", "accel"])

    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows for LINEARREG_SLOPE")
        X, y = d[["slope", "accel"]][:-1], (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["slope", "accel"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
