"""
CDLMORNINGSTAR Agent
====================

Morning Star (bullish 3-bar):

* Bar-2: long **black**.
* Bar-1: small body (≤ 50 % Bar-2) **gaps down** below Bar-2 low.
* Bar-0: long **white**, closes above midpoint of Bar-2 body.

Features
--------
* flag  (1 when pattern completed at t-1)
* roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _body(i, df): return (df["close"].shift(i) - df["open"].shift(i)).abs()
def _range(i, df): return (df["high"].shift(i) - df["low"].shift(i)).abs()
def _long(i, df):  return _body(i, df) >= _range(i, df) * 0.6
def _small(i, df): return _body(i, df) <= _range(i, df) * 0.5


def _flag(df):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]

    long_black = _long(2, df) & (c.shift(2) < o.shift(2))
    small_star = _small(1, df)
    gap_down   = h.shift(1) < l.shift(2)
    long_white = _long(0, df) & (c > o)
    close_mid  = c > (o.shift(2) + c.shift(2)) / 2

    return (long_black & small_star & gap_down & long_white & close_mid) \
              .shift().fillna(0).astype(float)


class CDLMORNINGSTAR_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<60: raise ValueError("Not enough rows for MORNINGSTAR")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
