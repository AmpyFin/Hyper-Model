"""
CDLCLOSINGMARUBOZU Agent
========================

Closing Marubozu:

* **White** – close == high, long body (bullish).
* **Black** – close == low,  long body (bearish).

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
    rng = h - l
    body = (c - o).abs()
    long_body = body >= rng * 0.7

    white = (c == h) & (c > o) & long_body
    black = (c == l) & (c < o) & long_body
    return (white | black).shift().fillna(0).astype(float)


class CDLCLOSINGMARUBOZU_Agent:
    def __init__(self):
        self.model=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy()
        d["flag"]=_flag(df)
        d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<40: raise ValueError("Not enough rows for CLOSINGMARUBOZU")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.model.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.model.predict_proba(last)[0,1]-1
