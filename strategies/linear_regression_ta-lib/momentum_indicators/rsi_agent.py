"""
RSI Agent
=========

Relative Strength Index (RSI-14, Wilder).

Features
--------
* **RSI / 100**    – scaled 0-1
* **RSI Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _rsi(close, n=14):
    diff = close.diff()
    up   = diff.clip(lower=0)
    dn   = -diff.clip(upper=0)
    avg_up = up.ewm(alpha=1/n, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = np.where(avg_dn == 0, 0, avg_up / avg_dn)
    return 100 - (100 / (1 + rs))

class RSI_Agent:
    def __init__(self, period=14):
        self.n = period
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False
    def _feat(self, df):
        if "close" not in df: raise ValueError("Need close column")
        rsi = _rsi(df["close"], self.n)
        d = df.copy()
        d["rsi"] = rsi / 100.0
        d["rsi_slope"] = d["rsi"].diff()
        d["roc3"] = d["close"].pct_change(3)
        feats = ["rsi", "rsi_slope", "roc3"]
        d[feats] = d[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows for RSI_Agent")
        X = d[["rsi", "rsi_slope", "roc3"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True
    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["rsi", "rsi_slope", "roc3"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
