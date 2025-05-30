"""
CDL3BLACKCROWS Agent
====================

"Three Black Crows" bearish pattern:

* Three consecutive long black candles
* Each opens inside prior body and closes lower

Feature set identical to CDL2CROWS.
"""

from __future__ import annotations
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression


def _is_3black(df):
    o, c = df["open"], df["close"]
    black1 = c.shift(2) < o.shift(2)
    black2 = c.shift(1) < o.shift(1)
    black3 = c < o
    open2_in1 = o.shift(1) < o.shift(2)
    open3_in2 = o < o.shift(1)
    lower2 = c.shift(1) < c.shift(2)
    lower3 = c < c.shift(1)
    return (black1 & black2 & black3 &
            open2_in1 & open3_in2 &
            lower2 & lower3).astype(float)


class CDL3BLACKCROWS_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        flag=_is_3black(df).shift()
        d=df.copy(); d["flag"]=flag; d["roc3"]=d["close"].pct_change(3)
        d[["flag","roc3"]]=d[["flag","roc3"]].fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df); 
        if len(d)<40: raise ValueError("Not enough rows for CDL3BLACKCROWS_Agent")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
