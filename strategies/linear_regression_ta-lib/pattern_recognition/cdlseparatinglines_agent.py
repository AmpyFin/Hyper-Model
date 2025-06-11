"""
CDLSEPARATINGLINES Agent
========================

Separating Lines (continuation, 2-bar):

* Prev trend candle (Bar-1) long.
* Bar-0 opens **equal** to Bar-1 open and closes strongly
  in the **same direction** (white after up-bar, black after down-bar).

Detection (simplified)
----------------------
* Bar-1 body ≥ 60 % of range.
* |open₀ − open₁| ≤ one tick (1 % median range).
* Colours identical and body long.

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _body(i, df): return (df["close"].shift(i) - df["open"].shift(i)).abs()
def _range(i, df): return (df["high"].shift(i) - df["low"].shift(i)).abs()
def _long(i, df):  return _body(i, df) >= _range(i, df) * 0.6


def _flag(df):
    o, c = df["open"], df["close"]
    rng_tick = (df["high"] - df["low"]).median() * 0.01

    same_color = ((c.shift(1) > o.shift(1)) & (c > o)) | \
                 ((c.shift(1) < o.shift(1)) & (c < o))

    open_equal = (o - o.shift(1)).abs() <= rng_tick

    return (_long(1, df) & _long(0, df) & same_color & open_equal) \
           .shift().fillna(0).astype(float)


class CDLSEPARATINGLINES_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000); self.fitted = False
    def _feat(self, df):
        d = df.copy()
        d["flag"] = _flag(df)
        d["roc3"] = d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self, df):
        d = self._feat(df)
        if len(d) < 40: raise ValueError("Not enough rows for SEPARATINGLINES")
        X = d[["flag", "roc3"]][:-1]; y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True
    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["flag", "roc3"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
