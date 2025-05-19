"""
AD Agent
========

Chaikin Accumulation / Distribution Line (A/D).

    MFM  = ((close − low) − (high − close)) / (high − low)
    MFV  = MFM · volume
    AD   = cumulative Σ(MFV)

Features
--------
* **AD_norm**   – AD divided by cumulative volume (keeps scale stable)
* **AD Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _ad(df):
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        (df["high"] - df["low"]).replace(0, np.nan)
    )
    mfv = mfm * df["volume"]
    return mfv.cumsum()

class AD_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self, df):
        req = {"high","low","close","volume"}
        if not req.issubset(df.columns): raise ValueError("Need OHLCV")
        ad = _ad(df)
        cum_vol = df["volume"].cumsum().replace(0, np.nan)
        d = df.copy()
        d["ad_norm"] = ad / cum_vol
        d["slope"] = d["ad_norm"].diff()
        d["roc3"] = d["close"].pct_change(3)
        feats = ["ad_norm", "slope", "roc3"]
        d[feats] = d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self, df):
        d=self._feat(df); 
        if len(d)<30: raise ValueError("Not enough rows for AD_Agent")
        X=d[["ad_norm","slope","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["ad_norm","slope","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
