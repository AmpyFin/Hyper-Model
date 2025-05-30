"""
CDLADVANCEBLOCK Agent
=====================

Advance Block – bearish warning after an up-move.

Simplified rules
----------------
* Three consecutive **white** candles.
* Each opens within prior body and closes higher.
* **Body size shrinks** each day.
* Upper wick length grows (profit-taking).

Feature set
-----------
* flag  (1 when pattern ended t-1)
* roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df: pd.DataFrame) -> pd.Series:
    o, c, h = df["open"], df["close"], df["high"]
    body = (c - o).abs()

    white1 = c.shift(2) > o.shift(2)
    white2 = c.shift(1) > o.shift(1)
    white3 = c > o

    open2_in1 = o.shift(1).between(o.shift(2), c.shift(2))
    open3_in2 = o.between(o.shift(1), c.shift(1))

    higher2 = c.shift(1) > c.shift(2)
    higher3 = c > c.shift(1)

    body_shrink = (body.shift(2) > body.shift(1)) & (body.shift(1) > body)
    wick_up = (h.shift(2) - c.shift(2) < h.shift(1) - c.shift(1)) & \
              (h.shift(1) - c.shift(1) < h - c)

    return (white1 & white2 & white3 &
            open2_in1 & open3_in2 &
            higher2 & higher3 &
            body_shrink & wick_up).shift().fillna(0).astype(float)


class CDLADVANCEBLOCK_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False
    def _feat(self, df):
        d = df.copy()
        d["flag"] = _flag(df)
        d["roc3"] = d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self, df):
        d=self._feat(df)
        if len(d)<60: raise ValueError("Not enough rows for ADVANCEBLOCK")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
