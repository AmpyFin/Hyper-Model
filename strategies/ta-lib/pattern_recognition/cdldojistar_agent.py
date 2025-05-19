"""
CDLDOJISTAR Agent
=================

Doji Star (neutral warning):

* Bar-1: long candle.
* Bar-0: **doji** gapped above (bull) or below (bear) bar-1 body.

Features
--------
* flag
* roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _is_doji(idx, df):
    body = abs(df["close"].iat[idx] - df["open"].iat[idx])
    rng = df["high"].iat[idx] - df["low"].iat[idx]
    return rng > 0 and body / rng <= 0.1


def _flag(df: pd.DataFrame) -> pd.Series:
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    flags = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        if not _is_doji(i, df):
            continue
        long1 = abs(c.iat[i - 1] - o.iat[i - 1]) > (h.iat[i - 1] - l.iat[i - 1]) * 0.5
        gap_up = l.iat[i] > h.iat[i - 1]
        gap_down = h.iat[i] < l.iat[i - 1]
        if long1 and (gap_up or gap_down):
            flags.iat[i] = 1.0
    return flags.shift().fillna(0)


class CDLDOJISTAR_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False
    def _feat(self, df):
        d = df.copy()
        d["flag"] = _flag(df)
        d["roc3"] = d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self, df):
        d = self._feat(df)
        if len(d) < 40: raise ValueError("Not enough rows for DOJISTAR")
        X = d[["flag", "roc3"]][:-1]; y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True
    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["flag", "roc3"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
