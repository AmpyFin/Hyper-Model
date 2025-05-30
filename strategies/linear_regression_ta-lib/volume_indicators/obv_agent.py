"""
OBV Agent
=========

On-Balance Volume:

    OBV_t = OBV_{t-1} + volume · sign(close_t − close_{t-1})

Features
--------
* **OBV_norm**   – OBV / cumulative volume
* **OBV Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _obv(df):
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()

class OBV_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self, df):
        if {"close","volume"}.issubset(df.columns) is False:
            raise ValueError("Need close and volume")
        obv=_obv(df)
        cum_vol=df["volume"].cumsum().replace(0,np.nan)
        d=df.copy()
        d["obv"]=obv/cum_vol
        d["slope"]=d["obv"].diff()
        d["roc3"]=d["close"].pct_change(3)
        feats=["obv","slope","roc3"]
        d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df)
        if len(d)<30: raise ValueError("Not enough rows for OBV_Agent")
        X=d[["obv","slope","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["obv","slope","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
