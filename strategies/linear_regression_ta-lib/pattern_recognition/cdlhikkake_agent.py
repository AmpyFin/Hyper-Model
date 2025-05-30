"""
CDLHIKKAKE Agent
================

Hikkake pattern (inside bar → false breakout).

Simplified rules
----------------
* Bar-2 is an **inside bar** (high<prev high & low>prev low).
* Bar-1 breaks the inside range (high>bar-2 high or low<bar-2 low).
* Bar-0 closes **back inside** the inside-bar range → “trap”.

Features
--------
* flag
* roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    inside  = (h.shift(2) < h.shift(3)) & (l.shift(2) > l.shift(3))

    up_break   = h.shift(1) > h.shift(2)
    down_break = l.shift(1) < l.shift(2)

    close_back = c.between(l.shift(2), h.shift(2))     # bar-0 close inside

    hk = inside & (up_break | down_break) & close_back
    return hk.shift().fillna(0).astype(float)


class CDLHIKKAKE_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000); self.fitted = False
    def _feat(self, df):
        d = df.copy()
        d["flag"] = _flag(df)
        d["roc3"] = d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self, df):
        d = self._feat(df)
        if len(d) < 60: raise ValueError("Not enough rows for HIKKAKE")
        X = d[["flag", "roc3"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True
    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["flag", "roc3"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
