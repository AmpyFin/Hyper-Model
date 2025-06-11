"""
CDLPIERCING Agent
=================

Piercing Pattern (bullish reversal):

* Bar-1: long **black**.
* Bar-0: long **white**, opens below Bar-1 low (gap down),
  closes **above** midpoint but below Bar-1 open.

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
    close_above_mid = c > (o.shift(1) + c.shift(1)) / 2
    close_below_open1 = c < o.shift(1)

    return (black1 & long1 & white0 & gap_dn & close_above_mid & close_below_open1) \
             .shift().fillna(0).astype(float)


class CDLPIERCING_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<50: raise ValueError("Not enough rows for PIERCING")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
