"""
ATR Agent
=========

Average True Range (ATR-14).

Features
--------
* **ATR / close**     (0-1 scale)
* **ATR Slope**       – pct_change of ATR
* **Price ROC-3**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from ..utils import BaseAgent


class ATR_Agent(BaseAgent):
    def __init__(self, period: int = 14):
        super().__init__()
        self.n = period

    # ---------------- feature builder ---------------- #
    def _feat(self, df: pd.DataFrame):
        req = {"high", "low", "close"}
        if not req.issubset(df.columns):
            raise ValueError("Need OHLC")
        hi, lo, cl = df["high"], df["low"], df["close"]
        tr = pd.concat(
            [hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(self.n).mean()
        d = df.copy()
        d["atr_pct"] = atr / cl
        d["slope"] = d["atr_pct"].pct_change()
        d["roc3"] = cl.pct_change(3)
        feats = ["atr_pct", "slope", "roc3"]
        # Handle inf values properly before ffill/bfill
        d = self.replace_inf_with_nan(d, feats)
        return d.dropna(subset=feats)

    # ---------------- train / predict ---------------- #
    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows for ATR_Agent")
        X = d[["atr_pct", "slope", "roc3"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        
        # Scale features before fitting
        X_scaled = self._scale_features(X)
        self.m.fit(X_scaled, y)
        self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["atr_pct", "slope", "roc3"]]
        
        # Scale features before prediction
        last_scaled = self._scale_features(last)
        return 2 * self.m.predict_proba(last_scaled)[0, 1] - 1
