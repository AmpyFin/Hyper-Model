"""
ROCP Agent
==========

ROCP = (price − prevPrice) / prevPrice  (period 10)

Features
--------
* **ROCP**
* **ROCP Slope**
* **Price ROC-3**   (shorter context)
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression

class ROCP_Agent:
    def __init__(self, period=10):
        self.n=period; self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        c=df["close"]; rocp=(c-c.shift(self.n))/c.shift(self.n)
        d=df.copy(); d["rocp"]=rocp
        d["slope"]=d["rocp"].diff()
        d["roc3"]=c.pct_change(3)
        feats=["rocp","slope","roc3"]
        d[feats]=d[feats].replace([float("inf"),float("-inf")],pd.NA).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df)
        if len(d)<self.n+10: raise ValueError("Not enough rows for ROCP_Agent")
        X=d[["rocp","slope","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["rocp","slope","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
