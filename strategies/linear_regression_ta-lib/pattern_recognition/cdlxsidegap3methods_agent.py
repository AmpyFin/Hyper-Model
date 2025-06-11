"""
CDLXSIDEGAP3METHODS Agent
=========================

Upside/Downside Gap Three Methods (TA-Lib “XSideGap3Methods”):

Bullish Upside variant
----------------------
* Bars-3 & -2 : two **white** with gap **up** between them.
* Bars-1 & 0  : two **black** falling but stay **above** first gap – 
  close does **not** fill the gap.

Bearish Downside variant = colours reversed with gap down.

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]

    # Bullish upside gap 3-methods
    bull = (
        (c.shift(3) > o.shift(3)) & (c.shift(2) > o.shift(2)) &     # two white
        (o.shift(2) > h.shift(3)) &                                 # gap up
        (c.shift(1) < o.shift(1)) & (c < o) &                       # two black
        (l.shift(1) > h.shift(3)) & (l > h.shift(3)) &              # stay above gap
        (c > l.shift(1))                                            # mild rise last bar
    )

    # Bearish downside gap 3-methods
    bear = (
        (c.shift(3) < o.shift(3)) & (c.shift(2) < o.shift(2)) &     # two black
        (o.shift(2) < l.shift(3)) &                                 # gap down
        (c.shift(1) > o.shift(1)) & (c > o) &                       # two white
        (h.shift(1) < l.shift(3)) & (h < l.shift(3)) &              # stay below gap
        (c < h.shift(1))
    )
    return (bull | bear).shift().fillna(0).astype(float)


class CDLXSIDEGAP3METHODS_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self, df):
        d = df.copy()
        d["flag"] = _flag(df)
        d["roc3"] = d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self, df):
        d = self._feat(df)
        if len(d) < 80:
            raise ValueError("Not enough rows for XSIDEGAP3METHODS")
        X, y = d[["flag", "roc3"]][:-1], (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True
    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["flag", "roc3"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
