"""
CDLSTALLEDPATTERN Agent
=======================

Stalled Pattern (bearish 3-bar – simplified):

* Bars-2 & -1: two advancing long **white** candles with higher highs.
* Bodies **shrink** and Bar-0 is a small white candle gapping up, 
  closing inside Bar-1 body.

Features: flag + roc3
"""

from __future__ import annotations
import pandas as pd
from ..utils import PatternAgent


class CDLSTALLEDPATTERN_Agent(PatternAgent):
    def __init__(self):
        super().__init__()

    def _body(self, i, df):
        return (df["close"].shift(i) - df["open"].shift(i)).abs()
        
    def _range(self, i, df):
        return (df["high"].shift(i) - df["low"].shift(i)).abs()
        
    def _long(self, i, df):
        body = self._body(i, df)
        rng = self._range(i, df)
        return self.series_compare(body, '>=', rng * 0.5)
        
    def _small(self, i, df):
        body = self._body(i, df)
        rng = self._range(i, df)
        return self.series_compare(body, '<=', rng * 0.3)

    def _flag(self, df):
        o, c = df["open"], df["close"]

        # Replace bitwise & with safe_and
        white2 = self.safe_and(
            self.series_compare(c.shift(2), '>', o.shift(2)),
            self._long(2, df)
        )
        
        white1 = self.safe_and(
            self.series_compare(c.shift(1), '>', o.shift(1)),
            self._long(1, df)
        )
        
        higher = self.safe_and(
            self.series_compare(c.shift(1), '>', c.shift(2)),
            self.series_compare(o.shift(1), '>', o.shift(2))
        )
        
        shrink = self.series_compare(self._body(1, df), '<', self._body(2, df))

        small0 = self.safe_and(
            self.series_compare(c, '>', o),
            self._small(0, df)
        )
        
        gap_up = self.series_compare(o, '>', c.shift(1))
        close_inside = self.series_compare(c, '<', c.shift(1))

        # Combine all conditions safely
        pattern = self.safe_and(
            white2, white1, higher, shrink, small0, gap_up, close_inside
        )
        
        return pattern.shift().fillna(0).astype(float)

    def _feat(self, df):
        d = df.copy()
        d["flag"] = self._flag(df)
        d["roc3"] = d["close"].pct_change(3).fillna(0)
        return d

    def fit(self, df):
        d = self._feat(df)
        if len(d) < 80:
            raise ValueError("Not enough rows for STALLEDPATTERN")
            
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
