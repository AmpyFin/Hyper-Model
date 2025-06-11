"""
CDLUNIQUE3RIVER Agent
=====================

Unique Three River – rare bullish reversal (simplified):

* Bar-2: long **black**.
* Bar-1: **hammer/dragonfly** gaps **down** (long lower shadow).
* Bar-0: small **black**, opens above Bar-1 open, 
  closes above Bar-1 close but **below** Bar-2 close.

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _is_hammer(i, df):
    o, c, h, l = df.iloc[i][["open", "close", "high", "low"]]
    body = abs(c - o)
    rng  = h - l
    lower = min(o, c) - l
    upper = h - max(o, c)
    return rng > 0 and body / rng <= 0.3 and lower >= body * 2 and upper <= body * 0.2


def _flag(df):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    idx = df.index
    f = pd.Series(0.0, index=idx)

    for i in range(2, len(df)):
        long_black = c.iat[i-2] < o.iat[i-2] and \
                     abs(o.iat[i-2]-c.iat[i-2]) >= (h.iat[i-2]-l.iat[i-2]) * 0.6
        if not long_black:
            continue
        if not _is_hammer(i-1, df):
            continue
        gap_down = o.iat[i-1] < l.iat[i-2]
        small_black = c.iat[i] < o.iat[i] and \
                      abs(c.iat[i]-o.iat[i]) <= (h.iat[i]-l.iat[i]) * 0.4
        cond = gap_down and small_black and \
               o.iat[i] > o.iat[i-1] and c.iat[i] > c.iat[i-1] and \
               c.iat[i] < c.iat[i-2]
        if cond:
            f.iat[i] = 1.0
    return f.shift().fillna(0)


class CDLUNIQUE3RIVER_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000); self.fitted = False
    def _feat(self, df):
        d = df.copy()
        d["flag"] = _flag(df)
        d["roc3"] = d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self, df):
        d = self._feat(df)
        if len(d) < 100:
            raise ValueError("Not enough rows for UNIQUE3RIVER")
        X, y = d[["flag", "roc3"]][:-1], (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True
    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["flag", "roc3"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
