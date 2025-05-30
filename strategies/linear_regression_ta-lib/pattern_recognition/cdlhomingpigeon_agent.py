"""
CDLHOMINGPIGEON Agent
=====================

Homing Pigeon – potential bullish reversal:

* Both candles are **black**.
* Bar-0 body is **inside** Bar-1 body.

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c = df["open"], df["close"]

    black1 = c.shift(1) < o.shift(1)
    black0 = c < o

    inside = (o >= o.shift(1)) & (c <= c.shift(1))
    return (black1 & black0 & inside).shift().fillna(0).astype(float)


class CDLHOMINGPIGEON_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<40: raise ValueError("Not enough rows for HOMINGPIGEON")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
