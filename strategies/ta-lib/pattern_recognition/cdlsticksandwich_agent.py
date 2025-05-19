"""
CDLSTICKSANDWICH Agent
======================

Stick Sandwich – bullish reversal (simplified):

* Bar-2: long **black**.
* Bar-1: **white** – closes above Bar-2 close.
* Bar-0: **black** – closes *equal* (±1 tick) to Bar-2 close.

Features
--------
* flag
* roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _body(i, df): return (df["close"].shift(i) - df["open"].shift(i)).abs()
def _range(i, df): return (df["high"].shift(i) - df["low"].shift(i)).abs()
def _long(i, df):  return _body(i, df) >= _range(i, df) * 0.5


def _flag(df):
    o, c = df["open"], df["close"]
    tick = (df["high"] - df["low"]).median() * 0.01
    black2 = c.shift(2) < o.shift(2) & _long(2, df)
    white1 = c.shift(1) > o.shift(1)
    close_equal = (c - c.shift(2)).abs() <= tick
    black0 = c < o
    return (black2 & white1 & black0 & close_equal).shift().fillna(0).astype(float)


class CDLSTICKSANDWICH_Agent:
    def __init__(self): self.m = LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self, df):
        d = df.copy()
        d["flag"] = _flag(df)
        d["roc3"] = d["close"].pct_change(3).fillna(0)
        return d.dropna()

    def fit(self, df):
        d = self._feat(df)
        if len(d) < 60: raise ValueError("Not enough rows for STICKSANDWICH")
        X, y = d[["flag", "roc3"]][:-1], (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["flag", "roc3"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
