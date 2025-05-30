"""
CDLGAPSIDESIDEWHITE Agent
=========================

Up/Down-Gap Side-by-Side White Lines (bullish continuation).

Simplified rules
----------------
* Gap up relative to Bar-2 high.
* Candle-1 and Candle-0 are both white.
* Opens equal (within tick) and bodies similar (<= 10 % difference).

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c, h = df["open"], df["close"], df["high"]

    white1 = c.shift(1) > o.shift(1)
    white0 = c > o
    gap_up = o.shift(1) > h.shift(2)

    open_equal = (o - o.shift(1)).abs() <= (df["high"] - df["low"]).median() * 0.01
    
    # Fix the max() function to work with Series
    body_current = abs(c - o)
    body_prev = abs(c.shift(1) - o.shift(1))
    # Use pandas' pmax() to get element-wise maximum
    body_max = pd.DataFrame({'current': body_current, 'prev': body_prev}).max(axis=1)
    body_equal = (body_current - body_prev).abs() <= body_max * 0.1

    # Handle each condition separately, filling NAs with False
    result = (
        white1.fillna(False) & 
        white0.fillna(False) & 
        gap_up.fillna(False) & 
        open_equal.fillna(False) & 
        body_equal.fillna(False)
    )
    
    return result.shift().fillna(0).astype(float)


class CDLGAPSIDESIDEWHITE_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<40: raise ValueError("Not enough rows for GAPSIDESIDEWHITE")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
