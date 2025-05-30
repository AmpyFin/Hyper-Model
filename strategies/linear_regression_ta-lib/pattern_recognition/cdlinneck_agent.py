"""
CDLINNECK Agent
===============

In-Neck Pattern (bearish continuation):

* Bar-1: long black.
* Bar-0: white, opens below Bar-1 low and **closes slightly above**
  Bar-1 low (but inside body).

Close threshold: within 25 % of Bar-1 body above its low.
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c, l = df["open"], df["close"], df["low"]
    body1 = (df["open"].shift(1) - df["close"].shift(1)).abs()
    black1 = df["close"].shift(1) < df["open"].shift(1)
    white0 = c > o
    open_below = o < l.shift(1)
    close_thr = c.between(l.shift(1), l.shift(1) + body1 * 0.25)
    inside_body = c < df["open"].shift(1)

    return (black1 & white0 & open_below & close_thr & inside_body).shift().fillna(0).astype(float)


class CDLINNECK_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<40: raise ValueError("Not enough rows for INNECK")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
