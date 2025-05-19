"""
CDLSTALLEDPATTERN Agent
=======================

Stalled Pattern (bearish 3-bar – simplified):

* Bars-2 & -1: two advancing long **white** candles with higher highs.
* Bodies **shrink** and Bar-0 is a small white candle gapping up, 
  closing inside Bar-1 body.

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _body(i, df): return (df["close"].shift(i) - df["open"].shift(i)).abs()
def _range(i, df): return (df["high"].shift(i) - df["low"].shift(i)).abs()
def _long(i, df):  return _body(i, df) >= _range(i, df) * 0.5
def _small(i, df): return _body(i, df) <= _range(i, df) * 0.3


def _flag(df):
    o, c = df["open"], df["close"]

    white2 = c.shift(2) > o.shift(2) & _long(2, df)
    white1 = c.shift(1) > o.shift(1) & _long(1, df)
    higher = (c.shift(1) > c.shift(2)) & (o.shift(1) > o.shift(2))
    shrink = _body(1, df) < _body(2, df)

    small0 = c > o & _small(0, df)
    gap_up = o > c.shift(1)
    close_inside = c < c.shift(1)  # within body

    return (white2 & white1 & higher & shrink & small0 & gap_up & close_inside) \
             .shift().fillna(0).astype(float)


class CDLSTALLEDPATTERN_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<80: raise ValueError("Not enough rows for STALLEDPATTERN")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
