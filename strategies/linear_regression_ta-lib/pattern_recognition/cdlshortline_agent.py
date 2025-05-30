"""
CDLSHORTLINE Agent
==================

Short Line Candle:

* Body ≤ 25 % of total range.
* Shadows also short: each ≤ 25 % of range.

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    rng  = (h - l).replace(0, pd.NA)
    body = (c - o).abs()
    upper = h - c.where(c > o, o)
    lower = o.where(c > o, c) - l

    cond = (body / rng <= 0.25) & (upper / rng <= 0.25) & (lower / rng <= 0.25)
    return cond.shift().fillna(0).astype(float)


class CDLSHORTLINE_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<30: raise ValueError("Not enough rows for SHORTLINE")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
