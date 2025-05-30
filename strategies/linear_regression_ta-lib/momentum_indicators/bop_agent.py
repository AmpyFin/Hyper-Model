"""
BOP Agent
=========

Balance of Power (BOP):

    BOP = (close − open) / (high − low)

Features
--------
* **BOP**         – raw BOP (already ~–1…+1)
* **BOP SMA-5**   – 5-bar SMA of BOP
* **BOP Slope**   – 1-bar slope of BOP
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

class BOP_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000); self.fitted=False
    # ---------------- feature builder ---------------- #
    def _feat(self,df):
        req={"open","high","low","close"}; 
        if not req.issubset(df.columns): raise ValueError("Missing OHLC")
        d=df.copy()
        rng = (d["high"]-d["low"]).replace(0,np.nan)
        d["bop"]       = (d["close"]-d["open"])/rng
        d["bop_sma5"]  = d["bop"].rolling(5).mean()
        d["bop_slope"] = d["bop"].diff()
        feats=["bop","bop_sma5","bop_slope"]
        d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)[feats+["close"]]
    # ---------------- train / predict ---------------- #
    def fit(self,df):
        d=self._feat(df); 
        X=d[["bop","bop_sma5","bop_slope"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        if len(X)<30: raise ValueError("Not enough rows for BOP_Agent")
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["bop","bop_sma5","bop_slope"]]
        return 2*self.m.predict_proba(last)[0,1]-1
