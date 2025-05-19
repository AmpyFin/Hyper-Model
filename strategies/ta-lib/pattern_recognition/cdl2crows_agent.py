"""
CDL2CROWS Agent
===============

"Two Crows" bearish-reversal (3-bar) pattern:

* bar-1: long **white** (close > open)
* bar-2: gaps up, **black**, closes above bar-1 close
* bar-3: **black**, opens above bar-2 open, closes inside bar-1 body

Feature
-------
* **flag**  – 1 if pattern at *t-1*, else 0
* **ROC-3**

LogisticRegression maps to score ∈ [-1, +1].
"""

from __future__ import annotations
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression


def _is_cdl2crows(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    white1 = c.shift(2) > o.shift(2)
    black2 = c.shift(1) < o.shift(1)
    black3 = c < o

    gap_up2 = o.shift(1) > c.shift(2)
    bar2_above = c.shift(1) > c.shift(2)

    open3_above2 = o > o.shift(1)
    close3_inside1 = (c < c.shift(2)) & (c > o.shift(2))

    return (white1 & black2 & black3 &
            gap_up2 & bar2_above &
            open3_above2 & close3_inside1).astype(float)


class CDL2CROWS_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False

    def _feat(self, df):
        flag = _is_cdl2crows(df).shift()  # signal formed at t-1
        d = df.copy()
        d["flag"] = flag
        d["roc3"] = d["close"].pct_change(3)
        feats = ["flag", "roc3"]
        d[feats] = d[feats].fillna(0)
        return d.dropna()

    def fit(self, df):
        d = self._feat(df)
        if len(d) < 40:
            raise ValueError("Not enough rows for CDL2CROWS_Agent")
        X = d[["flag", "roc3"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y)
        self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["flag", "roc3"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
