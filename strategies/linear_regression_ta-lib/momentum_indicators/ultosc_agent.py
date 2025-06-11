"""
ULTOSC Agent
===========

Ultimate Oscillator: blend of 7-, 14-, 28-period BP/ATR averages.

    BP = close − min(low, prev_close)
    TR = max(high, prev_close) − min(low, prev_close)

Weights: 4, 2, 1.

Features
--------
* **UO / 100**
* **UO Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _uo(df):
    cl, hi, lo = df["close"], df["high"], df["low"]
    prev_cl = cl.shift()
    bp = cl - np.minimum(lo, prev_cl)
    tr = np.maximum(hi, prev_cl) - np.minimum(lo, prev_cl)
    avg7  = bp.rolling(7).sum()  / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    return 100 * (4*avg7 + 2*avg14 + avg28) / 7

class ULTOSC_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        req={"high","low","close"}; 
        if not req.issubset(df.columns): raise ValueError("Need OHLC")
        uo=_uo(df)
        d=df.copy()
        d["uo"]=uo/100.0
        d["slope"]=d["uo"].diff()
        d["roc3"]=d["close"].pct_change(3)
        feats=["uo","slope","roc3"]
        d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df)
        if len(d)<40: raise ValueError("Not enough rows for ULTOSC_Agent")
        X=d[["uo","slope","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["uo","slope","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
