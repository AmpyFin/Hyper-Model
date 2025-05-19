"""
CDLIDENTICAL3CROWS Agent
========================

Identical Three Crows:

* Three consecutive long **black** candles.
* Each opens equal to prior **close** (± small tick).
* Each closes lower than prior.

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c = df["open"], df["close"]
    rng_tick = (df["high"] - df["low"]).median() * 0.01

    black1 = c.shift(2) < o.shift(2)
    black2 = c.shift(1) < o.shift(1)
    black3 = c < o

    open2_eq = (o.shift(1) - c.shift(2)).abs() <= rng_tick
    open3_eq = (o - c.shift(1)).abs() <= rng_tick

    lower2 = c.shift(1) < c.shift(2)
    lower3 = c < c.shift(1)

    return (black1 & black2 & black3 &
            open2_eq & open3_eq & lower2 & lower3).shift().fillna(0).astype(float)


class CDLIDENTICAL3CROWS_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<50: raise ValueError("Not enough rows for IDENTICAL3CROWS")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
