"""
CDLLADDERBOTTOM Agent
=====================

Ladder Bottom – five-bar bullish reversal (simplified):

* Bars 4-2: three long black candles with lower highs & lows.
* Bar 1: small black spinning top gapping down.
* Bar 0: long white closing above Bar 1 high and Bar 2 high.

Features
--------
* flag
* roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    black = lambda i: c.shift(i) < o.shift(i)
    long_body = lambda i: (o.shift(i) - c.shift(i)).abs() > (h.shift(i) - l.shift(i)) * 0.6
    short_body = lambda i: (o.shift(i) - c.shift(i)).abs() <= (h.shift(i) - l.shift(i)) * 0.3

    seq_black = black(4) & black(3) & black(2) & long_body(4) & long_body(3) & long_body(2)
    lower_lows = (l.shift(3) < l.shift(4)) & (l.shift(2) < l.shift(3))
    lower_highs = (h.shift(3) < h.shift(4)) & (h.shift(2) < h.shift(3))
    spinning1 = black(1) & short_body(1)
    gap_down1 = h.shift(1) < l.shift(2)
    white0 = c > o
    close_above = c > h.shift(1)
    flag = seq_black & lower_lows & lower_highs & spinning1 & gap_down1 & white0 & close_above
    return flag.shift().fillna(0).astype(float)


class CDLLADDERBOTTOM_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<120: raise ValueError("Not enough rows for LADDERBOTTOM")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
