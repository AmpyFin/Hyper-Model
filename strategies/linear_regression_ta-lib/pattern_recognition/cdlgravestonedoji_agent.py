"""
CDLGRAVESTONEDOJI Agent
=======================

Gravestone Doji:

* open ≈ close ≈ low
* long upper shadow (≥ 2 × total range)

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from ..utils import PatternAgent


class CDLGRAVESTONEDOJI_Agent(PatternAgent):
    def __init__(self):
        super().__init__()
    
    def _flag(self, df):
        """Calculate the gravestone doji pattern flag."""
        o, c, h, l = df["open"], df["close"], df["high"], df["low"]
        
        # Calculate ranges with proper handling of zero values
        rng = (h - l)
        body = (c - o).abs()
        upper = h - c
        lower = o - l
        
        # Avoid division by zero
        valid_range = rng > 0
        body_ratio = pd.Series(np.inf, index=df.index)
        body_ratio[valid_range] = body[valid_range] / rng[valid_range]
        
        # Safe conditions using our helper methods
        small_body = body_ratio <= 0.1
        small_lower = self.series_compare(lower, '<=', rng * 0.05)
        large_upper = self.series_compare(upper, '>=', rng * 0.7)
        
        # Combine conditions safely
        gravestone = self.safe_and(small_body, small_lower, large_upper)
        
        # Shift pattern to match the next candle and handle NAs
        return gravestone.shift().fillna(0).astype(float)

    def _feat(self, df):
        d = df.copy()
        d["flag"] = self._flag(df)
        d["roc3"] = d["close"].pct_change(3).fillna(0)
        return d

    def fit(self, df):
        d = self._feat(df)
        if len(d) < 30:
            raise ValueError("Not enough rows for GRAVESTONEDOJI")
        
        X = d[["flag", "roc3"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        
        # Scale features before fitting
        X_scaled = self._scale_features(X)
        self.m.fit(X_scaled, y)
        self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
            
        last = self._feat(historical_df).iloc[-1:][["flag", "roc3"]]
        
        # Scale features before prediction
        last_scaled = self._scale_features(last) 
        return 2 * self.m.predict_proba(last_scaled)[0, 1] - 1
