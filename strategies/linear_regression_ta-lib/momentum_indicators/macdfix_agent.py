"""
MACDFIX Agent
=============

MACD Fix 12/26 (EMA) with 9-period signal (EMA).

This is TA-Lib's `MACDFIX`.
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression

def _ema(s,span): return s.ewm(span=span,adjust=False).mean()

class MACDFIX_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        c=df["close"]
        macd=_ema(c,12)-_ema(c,26)
        signal=_ema(macd,9)
        hist=macd-signal
        d=df.copy()
        d["hist"]=hist/c
        d["signal_div"]=signal/c
        d["roc5"]=c.pct_change(5)
        feats=["hist","signal_div","roc5"]
        d[feats]=d[feats].replace([float("inf"),float("-inf")],pd.NA).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df)
        if len(d)<50: raise ValueError("Not enough rows for MACDFIX_Agent")
        X=d[["hist","signal_div","roc5"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["hist","signal_div","roc5"]]
        return 2*self.m.predict_proba(last)[0,1]-1
