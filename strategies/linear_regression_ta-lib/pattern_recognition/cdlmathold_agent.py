"""
CDLMATHOLD Agent
================

Mat Hold (bullish continuation, simplified):

* Bar-4: long white.
* Bar-3 opens higher (gap) – small body, closes lower.
* Bars-2 & -1 small bodies drifting lower but stay **above** Bar-4 close.
* Bar-0: long white, closes above Bar-4 high.

Lenient implementation focuses on the gap + consolidation + breakout idea.
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _body_size(i, df): return (df["close"].shift(i) - df["open"].shift(i)).abs()
def _long_body(i, df):  return _body_size(i, df) > (df["high"].shift(i) - df["low"].shift(i)) * 0.6
def _small_body(i, df): return _body_size(i, df) <= (df["high"].shift(i) - df["low"].shift(i)) * 0.4


def _flag(df):
    o, c, h = df["open"], df["close"], df["high"]
    cond = (
        _long_body(4, df) & (c.shift(4) > o.shift(4)) &

        (o.shift(3) > h.shift(4)) & _small_body(3, df) & (c.shift(3) < c.shift(4)) &

        _small_body(2, df) & _small_body(1, df) &
        (c.shift(2) > c.shift(3)) & (c.shift(1) > c.shift(2)) &  # gentle drift
        (c.shift(1) > c.shift(4)) & (c.shift(2) > c.shift(4)) &

        _long_body(0, df) & (c > o) & (c > h.shift(4))
    )
    return cond.shift().fillna(0).astype(float)


class CDLMATHOLD_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<150: raise ValueError("Not enough rows for MATHOLD")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
