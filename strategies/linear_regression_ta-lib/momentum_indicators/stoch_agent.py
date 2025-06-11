"""
STOCH Agent
===========

Classic Stochastic Oscillator %K/%D:

    %K = 100 * (close − L_n) / (H_n − L_n)
    %D = SMA(%K, 3)

Default n = 14.

Features
--------
* **%K / 100**
* **%D / 100**
* **K−D Spread**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _stoch(df, n=14):
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    k = 100 * (df["close"] - low_n) / (high_n - low_n)
    d = k.rolling(3).mean()
    return k, d

class STOCH_Agent:
    def __init__(self, period=14):
        self.n = period
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False
    def _feat(self, df):
        req = {"high", "low", "close"}
        if not req.issubset(df.columns): raise ValueError("Need OHLC columns")
        k, dline = _stoch(df, self.n)
        d = df.copy()
        d["k"] = k / 100.0
        d["d"] = dline / 100.0
        d["spread"] = d["k"] - d["d"]
        feats = ["k", "d", "spread"]
        d[feats] = d[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows for STOCH_Agent")
        X = d[["k", "d", "spread"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True
    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["k", "d", "spread"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
