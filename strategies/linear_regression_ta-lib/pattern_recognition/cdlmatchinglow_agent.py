"""
CDLMATCHINGLOW Agent
====================

Matching Low (bullish support):

* Two consecutive **black** candles.
* Second candle closes lower or equal to first close.
* **Lows match** within small tick (≤ 0.05 × average range).

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c, l = df["open"], df["close"], df["low"]
    rng_tick = (df["high"] - df["low"]).median() * 0.05

    black1 = c.shift(1) < o.shift(1)
    black0 = c < o
    lows_match = (l - l.shift(1)).abs() <= rng_tick

    return (black1 & black0 & lows_match).shift().fillna(0).astype(float)


class CDLMATCHINGLOW_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<40: raise ValueError("Not enough rows for MATCHINGLOW")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
