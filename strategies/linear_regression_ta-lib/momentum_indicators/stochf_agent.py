"""
STOCHF Agent
============

Fast Stochastic Oscillator:

    %K_fast = 100·(close − L_n)/(H_n − L_n)
    %D_fast = SMA(%K_fast, 3)

Defaults: n = 14.

Features
--------
* **Kf / 100**
* **Df / 100**
* **Spread** (Kf − Df)
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _stochf(df, n=14):
    low_n  = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    kf = 100 * (df["close"] - low_n) / (high_n - low_n)
    dfast = kf.rolling(3).mean()
    return kf, dfast

class STOCHF_Agent:
    def __init__(self, period=14):
        self.n = period
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False
    def _feat(self, df):
        req = {"high", "low", "close"}
        if not req.issubset(df.columns):
            raise ValueError("Missing OHLC")
        kf, df_ = _stochf(df, self.n)
        d = df.copy()
        d["kf"] = kf / 100.0
        d["df"] = df_ / 100.0
        d["spread"] = d["kf"] - d["df"]
        feats = ["kf", "df", "spread"]
        d[feats] = d[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 5:
            raise ValueError("Not enough rows for STOCHF_Agent")
        X = d[["kf", "df", "spread"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True
    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["kf", "df", "spread"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
