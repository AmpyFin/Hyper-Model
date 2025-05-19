"""
CDL3INSIDE Agent
================

Three-Inside Up / Down:

* Large candle
* Inside candle
* Third candle closes beyond candle-1 midpoint in opposite direction
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _is_3inside(df):
    o, c = df["open"], df["close"]
    body1 = (c.shift(2) - o.shift(2)).abs()
    body2_small = (c.shift(1) - o.shift(1)).abs() < body1 * 0.6
    inside = (c.shift(1).between(o.shift(2), c.shift(2)) |
              c.shift(1).between(c.shift(2), o.shift(2)))
    bullish = (c.shift(2) > o.shift(2)) & (c < o)  # bearish reversal
    bearish = (c.shift(2) < o.shift(2)) & (c > o)  # bullish reversal
    close_beyond = bullish | bearish
    return (body2_small & inside & close_beyond).astype(float)


class CDL3INSIDE_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        flag=_is_3inside(df).shift()
        d=df.copy(); d["flag"]=flag; d["roc3"]=d["close"].pct_change(3)
        d[["flag","roc3"]]=d[["flag","roc3"]].fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<40: raise ValueError("Not enough rows for CDL3INSIDE_Agent")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
