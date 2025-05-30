"""
CDLCOUNTERATTACK Agent
======================

Counterattack Lines (bullish or bearish):

* Bar-1: long candle.
* Bar-2: opposite colour, **opens with a gap** away from Bar-1 close,
  but **closes equal (≈)** to Bar-1 close.

"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c = df["open"], df["close"]
    body1_up   = c.shift(1) > o.shift(1)
    body1_down = c.shift(1) < o.shift(1)

    opp_color  = ((c > o) & body1_down) | ((c < o) & body1_up)
    gap        = (o - c.shift(1)).abs() > (c.shift(1) - o.shift(1)).abs() * 0.3
    equal_close = (c - c.shift(1)).abs() <= (df["high"] - df["low"]).shift(1) * 0.05

    return (opp_color & gap & equal_close).shift().fillna(0).astype(float)


class CDLCOUNTERATTACK_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<40: raise ValueError("Not enough rows for COUNTERATTACK")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
