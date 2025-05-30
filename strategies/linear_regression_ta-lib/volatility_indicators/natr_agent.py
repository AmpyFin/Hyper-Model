"""
NATR Agent
==========

Normalized ATR:

    NATR = 100 · ATR / close
    (ATR period = 14)

Features
--------
* **NATR / 100**   (0-1)
* **NATR Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from ..utils import BaseAgent


class NATR_Agent(BaseAgent):
    def __init__(self, period: int = 14):
        super().__init__()
        self.n = period

    def _feat(self, df):
        req = {"high", "low", "close"}
        if not req.issubset(df.columns):
            raise ValueError("Need OHLC")
        hi, lo, cl = df["high"], df["low"], df["close"]
        tr = pd.concat(
            [hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(self.n).mean()
        natr = 100 * atr / cl
        d = df.copy()
        d["natr"] = natr / 100.0
        d["slope"] = d["natr"].diff()
        d["roc3"] = cl.pct_change(3)
        feats = ["natr", "slope", "roc3"]
        d = self.replace_inf_with_nan(d, feats)
        return d.dropna(subset=feats)

    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows for NATR_Agent")
        X = d[["natr", "slope", "roc3"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        
        # Scale features before fitting
        X_scaled = self._scale_features(X)
        self.m.fit(X_scaled, y)
        self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["natr", "slope", "roc3"]]
        
        # Scale features before prediction
        last_scaled = self._scale_features(last)
        return 2 * self.m.predict_proba(last_scaled)[0, 1] - 1
