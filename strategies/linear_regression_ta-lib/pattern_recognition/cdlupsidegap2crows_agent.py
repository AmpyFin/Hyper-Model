"""
CDLUPSIDEGAP2CROWS Agent
========================

Upside Gap Two Crows – bearish reversal:

* Bar-2: long **white**.
* Bar-1: **black**, gaps **up** above Bar-2 high, closes lower.
* Bar-0: **black**, opens above Bar-1 open, closes **below** Bar-1 close
  but **still above** Bar-2 close (fills part of the gap).

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c, h = df["open"], df["close"], df["high"]

    white2 = c.shift(2) > o.shift(2)
    gap_up1 = o.shift(1) > h.shift(2)
    black1 = c.shift(1) < o.shift(1)
    black0 = c < o
    open_above1 = o > o.shift(1)
    close_lower = c < c.shift(1)
    above_white_close = c > c.shift(2)

    return (white2 & gap_up1 & black1 & black0 &
            open_above1 & close_lower & above_white_close)\
           .shift().fillna(0).astype(float)


class CDLUPSIDEGAP2CROWS_Agent:
    def __init__(self): self.m = LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self, df):
        d = df.copy()
        d["flag"] = _flag(df)
        d["roc3"] = d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self, df):
        d = self._feat(df)
        if len(d) < 60:
            raise ValueError("Not enough rows for UPSIDEGAP2CROWS")
        X, y = d[["flag", "roc3"]][:-1], (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True
    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["flag", "roc3"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
