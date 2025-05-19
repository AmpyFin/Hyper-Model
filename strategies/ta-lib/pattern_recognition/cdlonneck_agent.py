"""
CDLONNECK Agent
===============

On-Neck (bearish continuation, 2-bar):

* Bar-1: long **black**.
* Bar-0: white, opens below Bar-1 low and **closes ≈ low** of Bar-1
  (within 1 % of range).

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c, l = df["open"], df["close"], df["low"]
    rng_tick = (df["high"] - df["low"]).median() * 0.01

    black1 = c.shift(1) < o.shift(1)
    long1  = (o.shift(1) - c.shift(1)).abs() >= (df["high"] - df["low"]).shift(1) * 0.6

    white0 = c > o
    open_below = o < l.shift(1)
    close_neck = (c - l.shift(1)).abs() <= rng_tick

    return (black1 & long1 & white0 & open_below & close_neck).shift().fillna(0).astype(float)


class CDLONNECK_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<40: raise ValueError("Not enough rows for ONNECK")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
