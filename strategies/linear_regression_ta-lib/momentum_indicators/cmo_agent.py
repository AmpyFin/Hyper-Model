"""
CMO Agent
=========

Chande Momentum Oscillator (CMO, default 14).

    CMO = 100 * (SumUp − SumDown) / (SumUp + SumDown)

Features
--------
* **CMO / 100**    – scaled to –1…+1
* **CMO Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _cmo(close,n=14):
    diff=close.diff()
    up =  diff.clip(lower=0).rolling(n).sum()
    dn = -diff.clip(upper=0).rolling(n).sum()
    denom=up+dn
    return np.where(denom==0,0,100*(up-dn)/denom)

class CMO_Agent:
    def __init__(self,period:int=14):
        self.n=period; self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        if "close" not in df: raise ValueError("Need close")
        d=df.copy()
        cmo=_cmo(d["close"],self.n)
        d["cmo"]=cmo/100.0
        d["cmo_slope"]=d["cmo"].diff()
        d["roc3"]=d["close"].pct_change(3)
        feats=["cmo","cmo_slope","roc3"]
        d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)[feats+["close"]]
    def fit(self,df):
        d=self._feat(df); X=d[["cmo","cmo_slope","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        if len(X)<self.n+10: raise ValueError("Not enough rows for CMO_Agent")
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["cmo","cmo_slope","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
