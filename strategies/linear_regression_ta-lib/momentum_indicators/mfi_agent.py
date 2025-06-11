"""
MFI Agent
=========

Money Flow Index (MFI-14).

Features
--------
* **MFI / 100**   – scaled 0-1
* **MFI Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _mfi(df,n=14):
    tp=(df["high"]+df["low"]+df["close"])/3
    mf=tp*df["volume"]
    pos_mf=np.where(tp>tp.shift(),mf,0)
    neg_mf=np.where(tp<tp.shift(),mf,0)
    pos= pd.Series(pos_mf,index=df.index).rolling(n).sum()
    neg= pd.Series(neg_mf,index=df.index).rolling(n).sum()
    return np.where((pos+neg)==0,0,100*pos/(pos+neg))

class MFI_Agent:
    def __init__(self,period:int=14):
        self.n=period; self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        req={"high","low","close","volume"}
        if not req.issubset(df.columns): raise ValueError("Need OHLCV")
        mfi=_mfi(df,self.n)
        d=df.copy()
        d["mfi"]=mfi/100.0
        d["mfi_slope"]=d["mfi"].diff()
        d["roc3"]=d["close"].pct_change(3)
        feats=["mfi","mfi_slope","roc3"]
        d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df)
        if len(d)<self.n+10: raise ValueError("Not enough rows for MFI_Agent")
        X=d[["mfi","mfi_slope","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["mfi","mfi_slope","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
