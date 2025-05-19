"""
CDLMARUBOZU Agent
=================

Marubozu (open/close = extremes):

* **White** – open = low,   close = high,   long body.
* **Black** – open = high,  close = low,    long body.

Long body: body ≥ 70 % of range.
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    rng = (h - l).replace(0, pd.NA)
    body = (c - o).abs()
    long_body = body / rng >= 0.7

    white = (o == l) & (c == h) & (c > o) & long_body
    black = (o == h) & (c == l) & (c < o) & long_body
    return (white | black).shift().fillna(0).astype(float)


class CDLMARUBOZU_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<30: raise ValueError("Not enough rows for MARUBOZU")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
