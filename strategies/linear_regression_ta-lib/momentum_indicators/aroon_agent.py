"""
AROON Agent
===========

Aroon Up / Down (default period 25).

Features
--------
* **Up / 100**
* **Down / 100**
* **Osc** = Up − Down (scaled 0-1)
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _aroon(df, n):
    idx = np.arange(len(df))
    up  = (n - (idx - df["high"].rolling(n).apply(np.argmax, raw=True))) / n * 100
    dn  = (n - (idx - df["low"].rolling(n).apply(np.argmin,  raw=True))) / n * 100
    return pd.Series(up, index=df.index), pd.Series(dn, index=df.index)

class AROON_Agent:
    def __init__(self, period: int = 25):
        self.n = period
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False
    def _feat(self, df):
        req = {"high", "low", "close"}
        if not req.issubset(df.columns): raise ValueError("Missing OHLC")
        up, dn = _aroon(df, self.n)
        d = df[["close"]].copy()
        d["up"] = up / 100
        d["down"] = dn / 100
        d["osc"] = d["up"] - d["down"]
        feats = ["up", "down", "osc"]
        d[feats] = d[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 5: raise ValueError("Not enough rows for AROON_Agent")
        X = d[["up", "down", "osc"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True
    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["up", "down", "osc"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
