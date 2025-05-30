"""
CDLHIKKAKEMOD Agent
===================

Modified Hikkake – same as Hikkake but confirmation occurs one bar later
(Bar-0 **breaks** Bar-1 extreme).

Implementation
--------------
* Detect classic Hikkake on Bars 3-1.
* Bar-0 **posts a new extreme** past Bar-1 high/low → confirmation.
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    h, l = df["high"], df["low"]

    inside = (h.shift(3) < h.shift(4)) & (l.shift(3) > l.shift(4))
    break1 = (h.shift(2) > h.shift(3)) | (l.shift(2) < l.shift(3))

    confirm = (h > h.shift(2)) | (l < l.shift(2))

    return (inside & break1 & confirm).shift().fillna(0).astype(float)


class CDLHIKKAKEMOD_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<70: raise ValueError("Not enough rows for HIKKAKEMOD")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
