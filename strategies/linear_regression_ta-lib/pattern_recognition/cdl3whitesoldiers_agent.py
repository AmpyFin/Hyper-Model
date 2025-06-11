"""
CDL3WHITESOLDIERS Agent
=======================

Three Advancing White Soldiers – strong bullish continuation.

Rules (simplified)
------------------
* Three consecutive **white** candles (close > open)
* Each opens within prior body.
* Each closes higher than prior close.
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag_3soldiers(df):
    o, c = df["open"], df["close"]

    white1 = c.shift(2) > o.shift(2)
    white2 = c.shift(1) > o.shift(1)
    white3 = c > o

    open2_in1 = o.shift(1).between(o.shift(2), c.shift(2))
    open3_in2 = o.between(o.shift(1), c.shift(1))

    higher2 = c.shift(1) > c.shift(2)
    higher3 = c > c.shift(1)

    return (white1 & white2 & white3 &
            open2_in1 & open3_in2 &
            higher2 & higher3).astype(float)


class CDL3WHITESOLDIERS_Agent:
    def __init__(self):
        self.model=LogisticRegression(max_iter=1000); self.fitted=False
    def _features(self,df):
        flag=_flag_3soldiers(df).shift()
        d=df.copy(); d["flag"]=flag; d["roc3"]=d["close"].pct_change(3)
        d[["flag","roc3"]]=d[["flag","roc3"]].fillna(0)
        return d.dropna()
    def fit(self,df):
        data=self._features(df)
        if len(data)<40: raise ValueError("Not enough rows for CDL3WHITESOLDIERS_Agent")
        X=data[["flag","roc3"]][:-1]; y=(data["close"].shift(-1)>data["close"]).astype(int)[:-1]
        self.model.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._features(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.model.predict_proba(last)[0,1]-1
