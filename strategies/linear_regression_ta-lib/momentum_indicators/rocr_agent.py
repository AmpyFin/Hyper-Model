"""
ROCR Agent
==========

ROCR = price / prevPrice  (period 10)

Features
--------
* **ROCR**
* **ROCR Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression

class ROCR_Agent:
    def __init__(self, period=10):
        self.n=period; self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        c=df["close"]; rocr=c / c.shift(self.n)
        d=df.copy(); d["rocr"]=rocr
        d["slope"]=d["rocr"].diff()
        d["roc3"]=c.pct_change(3)
        feats=["rocr","slope","roc3"]
        d[feats]=d[feats].replace([float("inf"),float("-inf")],pd.NA).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df)
        if len(d)<self.n+10: raise ValueError("Not enough rows for ROCR_Agent")
        X=d[["rocr","slope","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["rocr","slope","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
