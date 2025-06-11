"""
APO Agent
=========

Absolute Price Oscillator:

    APO = EMA_fast − EMA_slow        (defaults 12/26)

Features
--------
* **APO / close**
* **APO Slope**
* **Price ROC-5**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from ..utils import BaseAgent

def _ema(s, span): return s.ewm(span=span, adjust=False).mean()

class APO_Agent(BaseAgent):
    def __init__(self, fast: int = 12, slow: int = 26):
        super().__init__()
        self.fast, self.slow = fast, slow

    def _feat(self, df):
        if "close" not in df: raise ValueError("Need close column")
        d = df.copy()
        apo = _ema(d["close"], self.fast) - _ema(d["close"], self.slow)
        d["apo_n"] = apo / d["close"]
        d["apo_slope"] = apo.pct_change()
        d["roc5"] = d["close"].pct_change(5)
        feats = ["apo_n", "apo_slope", "roc5"]
        d = self.replace_inf_with_nan(d, feats)
        return d.dropna(subset=feats)

    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.slow + 10:
            raise ValueError("Not enough rows for APO_Agent")
        X = d[["apo_n", "apo_slope", "roc5"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        
        X_scaled = self._scale_features(X)
        self.m.fit(X_scaled, y)
        self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted: self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["apo_n", "apo_slope", "roc5"]]
        
        last_scaled = self._scale_features(last)
        return 2 * self.m.predict_proba(last_scaled)[0, 1] - 1
