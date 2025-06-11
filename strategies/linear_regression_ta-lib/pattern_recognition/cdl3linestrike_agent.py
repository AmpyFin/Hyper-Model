"""
CDL3LINESTRIKE Agent
====================

Three-Line Strike (bull or bear):

* Three same-color candles with higher highs (bull) or lower lows (bear)
* Fourth candle opens in trend direction, then **reverses** to close
  beyond candle-1 open.

Feature scheme identical.
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _is_3linestrike(df):
    o, c = df["open"], df["close"]
    up_trend = (c.shift(3) > o.shift(3)) & (c.shift(2) > o.shift(2)) & (c.shift(1) > o.shift(1))
    up_seq = up_trend & (c.shift(2) > c.shift(3)) & (c.shift(1) > c.shift(2))
    fourth_bear = (c < o) & (o > c.shift(1)) & (c < o.shift(3))
    bull_strike = up_seq & fourth_bear

    down_trend = (c.shift(3) < o.shift(3)) & (c.shift(2) < o.shift(2)) & (c.shift(1) < o.shift(1))
    down_seq = down_trend & (c.shift(2) < c.shift(3)) & (c.shift(1) < c.shift(2))
    fourth_bull = (c > o) & (o < c.shift(1)) & (c > o.shift(3))
    bear_strike = down_seq & fourth_bull

    return (bull_strike | bear_strike).astype(float)


class CDL3LINESTRIKE_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        flag=_is_3linestrike(df).shift()
        d=df.copy(); d["flag"]=flag; d["roc3"]=d["close"].pct_change(3)
        d[["flag","roc3"]]=d[["flag","roc3"]].fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<50: raise ValueError("Not enough rows for CDL3LINESTRIKE_Agent")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
