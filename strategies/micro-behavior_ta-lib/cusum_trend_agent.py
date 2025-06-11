"""
CUSUM Trend Agent
~~~~~~~~~~~~~~~~~
Implements the Page-Hinkley cumulative-sum filter to isolate persistent
drift in log-returns.  Positive CUSUM → BUY, negative → SELL, scaled to
(-1, +1).

Input : OHLCV DataFrame.  Output ∈ (-1, +1).
"""

from __future__ import annotations
import numpy as np, pandas as pd

class CUSUM_Trend_Agent:
    def __init__(self, threshold: float = 0.0005, drift: float = 0.0):
        self.h  = threshold     # sensitivity ~ half a bp on equities
        self.k  = drift         # allowance
    def _cusum(self, r: np.ndarray) -> float:
        g_pos = g_neg = 0.0
        for x in r:
            g_pos = max(0.0, g_pos + x - self.k)
            g_neg = min(0.0, g_neg + x + self.k)
        return g_pos if abs(g_pos) > abs(g_neg) else g_neg
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        # No training needed for this rule-based agent
        pass
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        if len(historical_df) < 5:
            raise ValueError("Need at least 5 rows")
        log_ret = np.log(historical_df["close"]).diff().dropna().values
        c = self._cusum(log_ret[-250:])        # last ~ one day of 1-min bars
        return float(np.sign(c) * (1 - np.exp(-abs(c)/self.h)))
