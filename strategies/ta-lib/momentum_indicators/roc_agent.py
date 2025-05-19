"""
ROC Agent
=========

Rate of Change (ROC-10):

    ROC10 = 100 * (close / close_{t-10} − 1)

Features
--------
* **ROC10 / 100**
* **ROC10 Slope**
* **Price ROC-3**   (shorter momentum)
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression

class ROC_Agent:
    def __init__(self, period=10):
        self.n = period
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False
    def _feat(self, df):
        if "close" not in df: raise ValueError("Need close column")
        c = df["close"]
        roc = 100 * (c / c.shift(self.n) - 1)
        d = df.copy()
        d["roc_n"] = roc / 100.0
        d["roc_slope"] = d["roc_n"].diff()
        d["roc3"] = c.pct_change(3)
        feats = ["roc_n", "roc_slope", "roc3"]
        d[feats] = d[feats].replace([float("inf"), float("-inf")], pd.NA).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows for ROC_Agent")
        X = d[["roc_n", "roc_slope", "roc3"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True
    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["roc_n", "roc_slope", "roc3"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
