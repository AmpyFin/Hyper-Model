"""
MOM Agent
=========

Classic Momentum indicator:

    MOM_n = close_t − close_{t-n}

Defaults to n = 10.

Features
--------
* **MOM / close**
* **MOM Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression

class MOM_Agent:
    def __init__(self, period=10):
        self.n=period; self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        if "close" not in df: raise ValueError("Need close column")
        d=df.copy()
        mom=d["close"].diff(self.n)
        d["mom_n"]=mom/d["close"]
        d["mom_slope"]=mom.diff()/d["close"]
        d["roc3"]=d["close"].pct_change(3)
        feats=["mom_n","mom_slope","roc3"]
        d[feats]=d[feats].replace([float("inf"),float("-inf")],pd.NA).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df)
        if len(d)<self.n+10: raise ValueError("Not enough rows for MOM_Agent")
        X=d[["mom_n","mom_slope","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["mom_n","mom_slope","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
