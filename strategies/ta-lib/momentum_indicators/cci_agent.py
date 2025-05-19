"""
CCI Agent
=========

Commodity Channel Index (CCI, default 20).

Features
--------
* **CCI / 200**      – scaled to ~–1…+1
* **CCI Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _cci(df,n=20):
    tp=(df["high"]+df["low"]+df["close"])/3
    sma=tp.rolling(n).mean()
    md =(tp-sma).abs().rolling(n).mean()
    return (tp-sma)/(0.015*md)

class CCI_Agent:
    def __init__(self,period:int=20):
        self.n=period; self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        req={"high","low","close"}; 
        if not req.issubset(df.columns): raise ValueError("Missing OHLC")
        d=df.copy()
        cci=_cci(d,self.n)
        d["cci"]=cci/200.0
        d["cci_slope"]=d["cci"].diff()
        d["roc3"]=d["close"].pct_change(3)
        feats=["cci","cci_slope","roc3"]
        d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)[feats+["close"]]
    def fit(self,df):
        d=self._feat(df); X=d[["cci","cci_slope","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        if len(X)<self.n+10: raise ValueError("Not enough rows for CCI_Agent")
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["cci","cci_slope","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
