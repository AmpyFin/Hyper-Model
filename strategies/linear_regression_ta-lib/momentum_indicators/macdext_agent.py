"""
MACDEXT Agent
=============

Extended MACD with user-selectable MA types.

Default:
    fast=12 (EMA), slow=26 (EMA), signal=9 (SMA)

Supported ma_type: "ema", "sma", "wma" (weighted).
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _ma(series, span, ma_type):
    if ma_type=="ema": return series.ewm(span=span, adjust=False).mean()
    if ma_type=="sma": return series.rolling(span).mean()
    if ma_type=="wma":
        w=np.arange(1,span+1)
        return series.rolling(span).apply(lambda x: np.dot(x,w)/w.sum(), raw=True)
    raise ValueError("ma_type must be ema, sma, or wma")

class MACDEXT_Agent:
    def __init__(self, fast=12, slow=26, signal=9,
                 fast_ma="ema", slow_ma="ema", sig_ma="sma"):
        self.f,self.s,self.sig=fast,slow,signal
        self.ft,self.st,self.sigt = fast_ma, slow_ma, sig_ma
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        c=df["close"]
        macd=_ma(c,self.f,self.ft)-_ma(c,self.s,self.st)
        signal=_ma(macd,self.sig,self.sigt)
        hist=macd-signal
        d=df.copy()
        d["hist_n"]=hist/c
        d["macd_slope"]=macd.diff()/c
        d["roc5"]=c.pct_change(5)
        feats=["hist_n","macd_slope","roc5"]
        d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df)
        if len(d)<self.s+self.sig+10: raise ValueError("Not enough rows for MACDEXT_Agent")
        X=d[["hist_n","macd_slope","roc5"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["hist_n","macd_slope","roc5"]]
        return 2*self.m.predict_proba(last)[0,1]-1
