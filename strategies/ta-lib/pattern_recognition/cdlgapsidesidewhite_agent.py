"""
CDLGAPSIDESIDEWHITE Agent
=========================

Up/Down-Gap Side-by-Side White Lines (bullish continuation).

Simplified rules
----------------
* Gap up relative to Bar-2 high.
* Candle-1 and Candle-0 are both white.
* Opens equal (within tick) and bodies similar (<= 10 % difference).

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c, h = df["open"], df["close"], df["high"]

    white1 = c.shift(1) > o.shift(1)
    white0 = c > o
    gap_up = o.shift(1) > h.shift(2)

    open_equal = (o - o.shift(1)).abs() <= (df["high"] - df["low"]).median() * 0.01
    body_equal = (abs(c - o) - abs(c.shift(1) - o.shift(1))).abs() <= \
                 max(abs(c - o), abs(c.shift(1) - o.shift(1))) * 0.1

    return (white1 & white0 & gap_up & open_equal & body_equal) \
              .shift().fillna(0).astype(float)


class CDLGAPSIDESIDEWHITE_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<40: raise ValueError("Not enough rows for GAPSIDESIDEWHITE")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
