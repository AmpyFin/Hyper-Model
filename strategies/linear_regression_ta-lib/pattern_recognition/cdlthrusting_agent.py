"""
CDLTHRUSTING Agent
==================

Thrusting Pattern (bearish continuation):

* Bar-1: long **black**.
* Bar-0: **white**, opens below Bar-1 low (gap down),
  closes **inside body** of Bar-1 but **below midpoint**.

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]

    black1 = c.shift(1) < o.shift(1)
    long1  = (o.shift(1) - c.shift(1)).abs() >= (h.shift(1) - l.shift(1)) * 0.6

    white0 = c > o
    gap_dn = o < l.shift(1)
    close_inside = c.between(c.shift(1), o.shift(1))  # inside body
    below_mid = c < (c.shift(1) + o.shift(1)) / 2

    return (black1 & long1 & white0 & gap_dn & close_inside & below_mid) \
             .shift().fillna(0).astype(float)


class CDLTHRUSTING_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<50: raise ValueError("Not enough rows for THRUSTING")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
