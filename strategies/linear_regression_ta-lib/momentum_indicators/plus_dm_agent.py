"""
PLUS_DM Agent
=============

Plus Directional Movement (+DM, default 14) *normalised* by ATR.

Features
--------
* **+DM_norm**   – rolling Σ(+DM)/ATR
* **+DM Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _plus_dm_norm(df, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - df["close"].shift()).abs(),
         (df["low"] - df["close"].shift()).abs()],
        axis=1
    ).max(axis=1)
    atr = tr.rolling(n).mean().replace(0, np.nan)
    return pd.Series(plus_dm, index=df.index).rolling(n).sum() / atr

class PLUS_DM_Agent:
    def __init__(self, period=14):
        self.n = period
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False
    def _feat(self, df):
        req = {"high", "low", "close"}
        if not req.issubset(df.columns):
            raise ValueError("Missing OHLC columns")
        pdm = _plus_dm_norm(df, self.n).fillna(0)
        d = df.copy()
        d["pdm"] = pdm
        d["pdm_slope"] = d["pdm"].diff()
        d["roc3"] = d["close"].pct_change(3)
        feats = ["pdm", "pdm_slope", "roc3"]
        d[feats] = d[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows for PLUS_DM_Agent")
        X = d[["pdm", "pdm_slope", "roc3"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True
    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["pdm", "pdm_slope", "roc3"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
