"""
MINUS_DI Agent
==============

Minus Directional Indicator (−DI, default 14).

Features
--------
* **−DI / 100**
* **−DI Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _minus_di(df,n=14):
    up=df["high"].diff(); dn=-df["low"].diff()
    minus_dm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([df["high"]-df["low"],
                  (df["high"]-df["close"].shift()).abs(),
                  (df["low"]-df["close"].shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(n).mean().replace(0,np.nan)
    return 100*pd.Series(minus_dm,index=df.index).rolling(n).sum()/atr

class MINUS_DI_Agent:
    def __init__(self,period=14):
        self.n=period; self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        req={"high","low","close"}; 
        if not req.issubset(df.columns): raise ValueError("Missing OHLC")
        mdi=_minus_di(df,self.n).fillna(0)
        d=df.copy()
        d["mdi"]=mdi/100.0
        d["mdi_slope"]=d["mdi"].diff()
        d["roc3"]=d["close"].pct_change(3)
        feats=["mdi","mdi_slope","roc3"]
        d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df); 
        if len(d)<self.n+10: raise ValueError("Not enough rows for MINUS_DI_Agent")
        X=d[["mdi","mdi_slope","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["mdi","mdi_slope","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
