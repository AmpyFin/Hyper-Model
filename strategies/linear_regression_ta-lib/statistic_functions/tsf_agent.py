"""
TSF Agent
=========

30-period **Time Series Forecast** (TA-Lib TSF):

    ŷₜ = a + b·(N-1)   where a,b from linear regression of last N closes.

Features
--------
* tsf   (forecast value)
* resid (close − tsf)
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression


def _tsf(series: pd.Series, n: int = 30):
    t = np.arange(n)
    def forecast(window):
        y = window.values
        b, a = np.polyfit(t, y, 1)
        return a + b * (n - 1)
    return series.rolling(n).apply(forecast, raw=False)


class TSF_Agent:
    def __init__(self, period: int = 30):
        self.n = period
        self.m = LogisticRegression(max_iter=1000); self.fitted = False

    def _feat(self, df):
        fcast = _tsf(df["close"], self.n)
        d = df.copy()
        d["tsf"] = fcast
        d["resid"] = df["close"] - fcast
        return d.dropna(subset=["tsf", "resid"])

    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 5:
            raise ValueError("Not enough rows for TSF_Agent")
        X, y = d[["tsf", "resid"]][:-1], (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["tsf", "resid"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
