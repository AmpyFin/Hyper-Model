"""
MACD Agent
==========

Standard MACD (EMA-12 − EMA-26) with 9-period signal line.

Features
--------
* **Hist / close**   – (MACD − Signal) / close
* **Hist Slope**
* **Price ROC-5**
"""

from __future__ import annotations
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression

def _ema(series, span): return series.ewm(span=span, adjust=False).mean()

class MACD_Agent:
    def __init__(self, fast=12, slow=26, signal=9):
        self.f, self.s, self.sig = fast, slow, signal
        self.m = LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self, df):
        c = df["close"]; macd=_ema(c,self.f)-_ema(c,self.s); signal=_ema(macd,self.sig)
        hist = macd - signal
        d = df.copy()
        d["hist_n"] = hist / c
        d["hist_slope"] = hist.diff() / c
        d["roc5"] = c.pct_change(5)
        feats=["hist_n","hist_slope","roc5"]
        d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self, df):
        d=self._feat(df)
        if len(d)<self.s+self.sig+10: raise ValueError("Not enough rows for MACD_Agent")
        X=d[["hist_n","hist_slope","roc5"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["hist_n","hist_slope","roc5"]]
        return 2*self.m.predict_proba(last)[0,1]-1
