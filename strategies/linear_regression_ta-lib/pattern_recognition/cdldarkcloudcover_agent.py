"""
CDLDARKCLOUDCOVER Agent
=======================

Dark Cloud Cover – bearish reversal:

* Bar-1: long white.
* Bar-2: black, **opens above** Bar-1 high (gap up),
  then closes into Bar-1 body, below its midpoint.

"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c, h = df["open"], df["close"], df["high"]

    white1 = c.shift(1) > o.shift(1)
    long1  = (c.shift(1) - o.shift(1)) > (df["high"] - df["low"]).shift(1) * 0.6

    black2 = c < o
    gap_up = o > h.shift(1)
    close_inside = c < (o.shift(1) + c.shift(1)) / 2
    above_open = c > o.shift(1)

    return (white1 & long1 & black2 & gap_up & close_inside & above_open).shift().fillna(0).astype(float)


class CDLDARKCLOUDCOVER_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<50: raise ValueError("Not enough rows for DARKCLOUDCOVER")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
