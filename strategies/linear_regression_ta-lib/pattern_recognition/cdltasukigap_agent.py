"""
CDLTASUKIGAP Agent
==================

Tasuki Gap (continuation, simplified):

Bullish variant
---------------
* Bar-2 & Bar-1: two **white** candles – Bar-1 gaps **up**.
* Bar-0: **black**, opens inside Bar-1 body, closes inside the gap but
  **does not fill** it (close > Bar-2 high).

Bearish variant = colours reversed / gap down.

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _body(i, df): return (df["close"].shift(i) - df["open"].shift(i)).abs()
def _long(i, df): return _body(i, df) >= (df["high"].shift(i) - df["low"].shift(i)) * 0.5


def _flag(df):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]

    # Bullish
    bull = (
        (c.shift(2) > o.shift(2)) & (c.shift(1) > o.shift(1)) &
        (o.shift(1) > h.shift(2)) &                               # gap up
        (c < o) &                                                 # black bar
        (o < c.shift(1)) & (o > o.shift(2)) &                     # opens inside
        (c > h.shift(2))                                          # closes in gap
    )

    # Bearish
    bear = (
        (c.shift(2) < o.shift(2)) & (c.shift(1) < o.shift(1)) &
        (o.shift(1) < l.shift(2)) &                             # gap down
        (c > o) &
        (o > c.shift(1)) & (o < o.shift(2)) &                   # opens inside
        (c < l.shift(2))
    )
    return (bull | bear).shift().fillna(0).astype(float)


class CDLTASUKIGAP_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<80: raise ValueError("Not enough rows for TASUKIGAP")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
