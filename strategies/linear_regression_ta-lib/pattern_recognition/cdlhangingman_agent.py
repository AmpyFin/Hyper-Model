"""
CDLHANGINGMAN Agent
===================

Hanging Man (bearish single-bar):

* Same geometry as a Hammer (small body near top, long lower shadow ≥ 2× body),
  **but appears after an up-move** (captured here via prior 3-bar ROC > 0).

Features
--------
* flag
* roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _is_hammer_like(df):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    body = (c - o).abs()
    rng = h - l
    upper = h - c.where(c > o, o)
    lower = (o.where(c > o, c) - l)
    return (body / rng <= 0.4) & (upper <= body * 0.25) & (lower >= body * 2)


def _flag(df):
    hammer = _is_hammer_like(df)
    uptrend = df["close"].pct_change(3) > 0          # rough 3-bar up-move
    return (hammer & uptrend).shift().fillna(0).astype(float)


class CDLHANGINGMAN_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000); self.fitted = False
    def _feat(self, df):
        d = df.copy()
        d["flag"] = _flag(df)
        d["roc3"] = d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self, df):
        d=self._feat(df)
        if len(d) < 30: raise ValueError("Not enough rows for HANGINGMAN")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted = True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
