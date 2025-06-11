"""
CDLEVENINGSTAR Agent
====================

Evening Star (bearish, 3-bar):

* Bar-1: long white.
* Bar-2: small body (<= 50 % Bar-1) that **gaps up** above Bar-1 high.
* Bar-3: black, closes below midpoint of Bar-1 body.

Features
--------
* flag  (1 when pattern completed at t-1)
* roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df: pd.DataFrame) -> pd.Series:
    o, c, h = df["open"], df["close"], df["high"]
    body1 = (c.shift(2) - o.shift(2)).abs()
    body2 = (c.shift(1) - o.shift(1)).abs()

    long_white1 = c.shift(2) > o.shift(2)
    small2      = body2 <= body1 * 0.5
    gap_up2     = o.shift(1) > h.shift(2)

    black3      = c < o
    close_below = c < (o.shift(2) + c.shift(2)) / 2

    return (long_white1 & small2 & gap_up2 & black3 & close_below) \
              .shift().fillna(0).astype(float)


class CDLEVENINGSTAR_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self, df):
        d = df.copy()
        d["flag"] = _flag(df)
        d["roc3"] = d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self, df):
        d=self._feat(df)
        if len(d)<60: raise ValueError("Not enough rows for EVENINGSTAR")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
