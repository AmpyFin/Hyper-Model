"""
WILLR Agent
===========

Williams %R (period 14):

    %R = −100 * (High_n − Close) / (High_n − Low_n)

Features
--------
* **%R / −100**   (scales to 0…1)
* **%R Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression

def _willr(df,n=14):
    high_n=df["high"].rolling(n).max()
    low_n =df["low"] .rolling(n).min()
    return -100 * (high_n - df["close"]) / (high_n - low_n)

class WILLR_Agent:
    def __init__(self, period=14):
        self.n=period; self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        req={"high","low","close"}; 
        if not req.issubset(df.columns): raise ValueError("Need OHLC")
        wr=_willr(df,self.n)
        d=df.copy(); d["wr"]=wr/ -100.0  # scale 0..1
        d["slope"]=d["wr"].diff()
        d["roc3"]=d["close"].pct_change(3)
        feats=["wr","slope","roc3"]
        d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df)
        if len(d)<self.n+10: raise ValueError("Not enough rows for WILLR_Agent")
        X=d[["wr","slope","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["wr","slope","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
