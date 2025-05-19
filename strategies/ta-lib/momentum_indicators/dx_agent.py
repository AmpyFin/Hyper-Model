"""
DX Agent
========

Directional Movement Index (single-period DX, default 14 for DI calcs).

    DX = 100 * |+DI − −DI| / (+DI + −DI)

Features
--------
* **DX / 100**     – trend discrimination
* **DI Spread**    – (+DI − −DI) / 100
* **ADX-1**        – 1-period lagged ADX (smoothed DX)
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _dm_parts(df,n=14):
    up=df["high"].diff(); dn=-df["low"].diff()
    plus_dm=np.where((up>dn)&(up>0),up,0.0)
    minus_dm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([df["high"]-df["low"],
                  (df["high"]-df["close"].shift()).abs(),
                  (df["low"]-df["close"].shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(n).mean().replace(0,np.nan)
    plus_di = 100*pd.Series(plus_dm,index=df.index).rolling(n).sum()/atr
    minus_di=100*pd.Series(minus_dm,index=df.index).rolling(n).sum()/atr
    den=plus_di+minus_di
    dx=np.where(den==0,0, abs(plus_di-minus_di)/den*100)
    adx=pd.Series(dx,index=df.index).rolling(n).mean()
    return plus_di.fillna(0),minus_di.fillna(0),pd.Series(dx,index=df.index).fillna(0),adx.fillna(0)

class DX_Agent:
    def __init__(self,period:int=14):
        self.n=period; self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        req={"high","low","close"}; 
        if not req.issubset(df.columns): raise ValueError("Missing OHLC")
        pdi,mdi,dx,adx=_dm_parts(df,self.n)
        d=df[["close"]].copy()
        d["dx"]=dx/100.0
        d["spread"]=(pdi-mdi)/100.0
        d["adx1"]=adx.shift(1)/100.0
        feats=["dx","spread","adx1"]
        d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df); X=d[["dx","spread","adx1"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        if len(X)<self.n+10: raise ValueError("Not enough rows for DX_Agent")
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["dx","spread","adx1"]]
        return 2*self.m.predict_proba(last)[0,1]-1
