"""
PPO Agent
=========

Percentage Price Oscillator:

    PPO = 100 * (EMA_fast − EMA_slow) / EMA_slow

Defaults: fast = 12, slow = 26.

Features
--------
* **PPO / 100**      – already scaled
* **PPO Slope**
* **Price ROC-5**
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression

def _ema(s, span): return s.ewm(span=span, adjust=False).mean()

class PPO_Agent:
    def __init__(self, fast=12, slow=26):
        self.f, self.s = fast, slow
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False
    def _feat(self, df):
        c = df["close"]
        ppo = 100 * (_ema(c, self.f) - _ema(c, self.s)) / _ema(c, self.s)
        d = df.copy()
        d["ppo"] = ppo / 100.0
        d["ppo_slope"] = d["ppo"].diff()
        d["roc5"] = c.pct_change(5)
        feats = ["ppo", "ppo_slope", "roc5"]
        d[feats] = d[feats].replace([float("inf"), float("-inf")], pd.NA).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.s + 10:
            raise ValueError("Not enough rows for PPO_Agent")
        X = d[["ppo", "ppo_slope", "roc5"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True
    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["ppo", "ppo_slope", "roc5"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
