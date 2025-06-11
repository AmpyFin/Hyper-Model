"""
TRIX Agent
==========

TRIX = 100 * EMA3(close) / EMA3(close).shift(1) − 100
(where EMA3 = EMA( EMA( EMA(close) ) ))

Default period = 15.

Features
--------
* **TRIX / 100**
* **TRIX Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression

def _ema(s, span): return s.ewm(span=span, adjust=False).mean()
def _trix(close, n=15):
    e1 = _ema(close, n)
    e2 = _ema(e1, n)
    e3 = _ema(e2, n)
    return 100 * (e3 / e3.shift() - 1)

class TRIX_Agent:
    def __init__(self, period=15):
        self.n=period; self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        if "close" not in df: raise ValueError("Need close")
        trix=_trix(df["close"],self.n)
        d=df.copy(); d["trix"]=trix/100.0
        d["slope"]=d["trix"].diff()
        d["roc3"]=d["close"].pct_change(3)
        feats=["trix","slope","roc3"]
        d[feats]=d[feats].replace([float("inf"),float("-inf")],pd.NA).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df)
        if len(d)<self.n*3+10: raise ValueError("Not enough rows for TRIX_Agent")
        X=d[["trix","slope","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["trix","slope","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
