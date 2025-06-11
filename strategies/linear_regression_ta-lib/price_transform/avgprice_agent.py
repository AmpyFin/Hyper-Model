"""
AVGPRICE Agent
==============

Average Price:

    AP = (open + high + low + close) / 4

Features
--------
* Divergence = (close − AP) / AP
* AP Slope   = AP.pct_change()
* Price ROC-3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


class AVGPRICE_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False

    def _feat(self, df: pd.DataFrame):
        req = {"open", "high", "low", "close"}
        if not req.issubset(df.columns):
            raise ValueError("Need OHLC")
        ap = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        d = df.copy()
        d["div"] = (df["close"] - ap) / ap
        d["slope"] = ap.pct_change()
        d["roc3"] = df["close"].pct_change(3)
        feats = ["div", "slope", "roc3"]
        d[feats] = d[feats].ffill().bfill()
        return d.dropna(subset=feats)

    def fit(self, df):
        d = self._feat(df)
        if len(d) < 30:
            raise ValueError("Not enough rows for AVGPRICE_Agent")
        X = d[["div", "slope", "roc3"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y)
        self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["div", "slope", "roc3"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
