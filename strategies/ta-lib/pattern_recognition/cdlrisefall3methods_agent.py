"""
CDLRISEFALL3METHODS Agent
=========================

Rising / Falling Three Methods  (continuation, 5-bar – simplified)

Bullish Rising Three
--------------------
* Bar-4: long **white**.
* Bars-3…-1: three small **black** candles inside Bar-4 body.
* Bar-0: long **white**, closes > Bar-4 high.

Bearish Falling Three   = colours reversed.

Features
--------
* flag
* roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _body_size(i, df): return (df["close"].shift(i) - df["open"].shift(i)).abs()
def _long(i, df): return _body_size(i, df) >= (df["high"].shift(i) - df["low"].shift(i)) * 0.6
def _small(i, df): return _body_size(i, df) <= (df["high"].shift(i) - df["low"].shift(i)) * 0.4


def _flag(df):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]

    # Bullish
    bull_seq = (
        _long(4, df) & (c.shift(4) > o.shift(4)) &
        _small(3, df) & _small(2, df) & _small(1, df) &
        (c.shift(3) < c.shift(4)) & (c.shift(2) < c.shift(3)) & (c.shift(1) < c.shift(2)) &
        (o.shift(3) > o.shift(4)) & (o.shift(2) > o.shift(4)) & (o.shift(1) > o.shift(4)) &
        _long(0, df) & (c > o) & (c > h.shift(4))
    )

    # Bearish
    bear_seq = (
        _long(4, df) & (c.shift(4) < o.shift(4)) &
        _small(3, df) & _small(2, df) & _small(1, df) &
        (c.shift(3) > c.shift(4)) & (c.shift(2) > c.shift(3)) & (c.shift(1) > c.shift(2)) &
        (o.shift(3) < o.shift(4)) & (o.shift(2) < o.shift(4)) & (o.shift(1) < o.shift(4)) &
        _long(0, df) & (c < o) & (c < l.shift(4))
    )
    return (bull_seq | bear_seq).shift().fillna(0).astype(float)


class CDLRISEFALL3METHODS_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<150: raise ValueError("Not enough rows for RISEFALL3METHODS")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
