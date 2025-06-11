"""
CDLTRISTAR Agent
================

Tristar (three-doji reversal):

Bullish
-------
* Three consecutive **doji**.
* First two gap **down**, third gaps **up** and closes above mid of
  gap between first two.

Bearish = gaps up then final doji gaps down.

Simplified detection: just require three doji with alternating gaps.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from ..utils import PatternAgent


class CDLTRISTAR_Agent(PatternAgent):
    def __init__(self):
        super().__init__()

    def _flag(self, df):
        """Calculate the tristar pattern flag with vectorized operations."""
        # Pre-calculate is_doji for all bars at once (vectorized)
        doji = self.is_doji(df, 0, doji_threshold=0.1)
        doji_1 = doji.shift(1)
        doji_2 = doji.shift(2)
        
        # Get high/low values with proper alignment
        h = df["high"]
        l = df["low"]
        h_1 = h.shift(1)
        l_1 = l.shift(1)
        h_2 = h.shift(2)
        l_2 = l.shift(2)
        
        # Check for three consecutive doji (vectorized)
        three_doji = doji & doji_1 & doji_2
        
        # Check gaps (vectorized)
        bullish_gaps = (l_2 > h_1) & (l_1 > h)
        bearish_gaps = (h_2 < l_1) & (h_1 < l)
        
        # Combine conditions (vectorized)
        pattern = three_doji & (bullish_gaps | bearish_gaps)
        
        # Shift for prediction and return
        return pattern.shift().fillna(0).astype(float)

    def _feat(self, df):
        d = df.copy()
        d["flag"] = self._flag(df)
        d["roc3"] = d["close"].pct_change(3).fillna(0)
        return d

    def fit(self, df):
        d = self._feat(df)
        if len(d) < 90:
            raise ValueError("Not enough rows for TRISTAR")
            
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
