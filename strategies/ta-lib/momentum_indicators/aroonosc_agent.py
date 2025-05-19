"""
AROONOSC Agent
==============

Aroon Oscillator only:

    Osc = AroonUp − AroonDown

Features
--------
* **Osc / 100**
* **Osc Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _aroonosc(df, n):
    idx = np.arange(len(df))
    up = (n - (idx - df["high"].rolling(n).apply(np.argmax, raw=True))) / n * 100
    dn = (n - (idx - df["low"].rolling(n).apply(np.argmin,  raw=True))) / n * 100
    return up - dn

class AROONOSC_Agent:
    def __init__(self, period: int = 25):
        self.n = period
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False
    def _feat(self, df):
        req = {"high", "low", "close"}
        if not req.issubset(df.columns): raise ValueError("Missing OHLC")
        osc = _aroonosc(df, self.n)
        d = df.copy()
        d["osc"] = osc / 100
        d["osc_slope"] = d["osc"].diff()
        d["roc3"] = d["close"].pct_change(3)
        feats = ["osc", "osc_slope", "roc3"]
        d[feats] = d[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 5: raise ValueError("Not enough rows for AROONOSC_Agent")
        X = d[["osc", "osc_slope", "roc3"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True
    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["osc", "osc_slope", "roc3"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
