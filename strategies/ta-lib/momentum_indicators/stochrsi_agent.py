"""
STOCHRSI Agent
==============

Stochastic RSI (RSI on 14, then Stoch over 14).

    RSI_n
    StochRSI = (RSI − min(RSI_n)) / (max(RSI_n) − min(RSI_n))

Features
--------
* **StochRSI**
* **StochRSI Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _rsi(close, n=14):
    diff = close.diff()
    up = diff.clip(lower=0)
    dn = -diff.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / dn.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _stochrsi(rsi, n=14):
    min_r = rsi.rolling(n).min()
    max_r = rsi.rolling(n).max()
    return (rsi - min_r) / (max_r - min_r)

class STOCHRSI_Agent:
    def __init__(self, rsi_period=14, stoch_period=14):
        self.rn, self.sn = rsi_period, stoch_period
        self.m = LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self, df):
        if "close" not in df: raise ValueError("Need close")
        rsi = _rsi(df["close"], self.rn)
        srsi = _stochrsi(rsi, self.sn)
        d = df.copy()
        d["srsi"] = srsi
        d["slope"] = d["srsi"].diff()
        d["roc3"] = d["close"].pct_change(3)
        feats = ["srsi", "slope", "roc3"]
        d[feats] = d[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self, df):
        d = self._feat(df)
        need = max(self.rn, self.sn) + 10
        if len(d) < need: raise ValueError("Not enough rows for STOCHRSI_Agent")
        X = d[["srsi", "slope", "roc3"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted=True
    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["srsi", "slope", "roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
