"""
MACD Crossover Agent
~~~~~~~~~~~~~~~~~~~
Detects crossovers between the MACD line and its signal line to identify
momentum shifts. Incorporates the strength of the crossover and histogram 
magnitude to provide signal strength.

Logic:
1. Calculate MACD (12, 26, 9) - fast EMA, slow EMA, signal EMA
2. Detect crossovers between MACD line and signal line
3. Calculate signal strength based on:
   - Direction of crossover (bullish/bearish)
   - Magnitude of the MACD histogram
   - Trend confirmation based on MACD line direction
4. Normalize output to range [-1, +1]

Input: OHLCV DataFrame. Output ∈ [-1, +1].

Dependencies
~~~~~~~~~~~~
pip install ta
"""

from __future__ import annotations
import numpy as np
import pandas as pd
try:
    import ta
except ModuleNotFoundError as e:
    raise ImportError('Install ta: pip install ta') from e

class MACD_Crossover_Agent:
    def __init__(
        self, 
        fast_window: int = 12,
        slow_window: int = 26,
        signal_window: int = 9,
        smoothing_window: int = 3,
    ):
        self.fast = fast_window
        self.slow = slow_window
        self.signal = signal_window
        self.smooth = smoothing_window
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        # No training needed for this rule-based agent
        pass
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        min_length = max(self.fast, self.slow, self.signal) + 10
        if len(historical_df) < min_length:
            raise ValueError(f"Need at least {min_length} rows")
        
        close = historical_df["close"]
        
        # Calculate MACD
        macd_indicator = ta.trend.MACD(
            close=close,
            window_fast=self.fast,
            window_slow=self.slow,
            window_sign=self.signal
        )
        
        macd_line = macd_indicator.macd()
        signal_line = macd_indicator.macd_signal()
        histogram = macd_indicator.macd_diff()
        
        # Get the most recent values
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        
        # Detect crossover by checking current and previous relationship
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        
        # Calculate standard deviations for normalization
        hist_std = histogram.iloc[-30:].std() or 1e-6  # Moved outside conditional
        macd_std = macd_line.iloc[-30:].std() or 1e-6
        
        # Check if there's a crossover
        bullish_cross = prev_macd < prev_signal and current_macd > current_signal
        bearish_cross = prev_macd > prev_signal and current_macd < current_signal
        
        # Base score calculation
        if not (bullish_cross or bearish_cross):
            # No recent crossover - score based on histogram
            base_score = np.tanh(current_hist / (2 * hist_std))
            
            # Add trend bias
            macd_slope = (current_macd - macd_line.iloc[-3]) / 3  # Shorter for minute data
            trend_strength = np.tanh(macd_slope / macd_std)
            base_score = base_score * (1 + 0.2 * trend_strength)  # Small trend bias
        else:
            # Recent crossover detected - stronger signal
            direction = 1.0 if bullish_cross else -1.0
            
            # Get trend strength from MACD line slope (shorter for minute data)
            macd_slope = (current_macd - macd_line.iloc[-3]) / 3
            trend_strength = np.tanh(macd_slope / macd_std)
            
            # Stronger signal for crossover in direction of trend
            if np.sign(trend_strength) == np.sign(direction):
                base_score = direction * (0.8 + 0.2 * abs(trend_strength))  # More weight on crossover
            else:
                base_score = direction * 0.6  # Weaker if against trend
        
        # Apply light smoothing to avoid extreme swings
        if len(historical_df) > self.smooth + min_length:
            prev_scores = []
            for i in range(1, self.smooth + 1):
                prev_idx = -i - 1
                p_hist = histogram.iloc[prev_idx]
                p_score = np.tanh(p_hist / (2 * hist_std))
                prev_scores.append(p_score)
                
            # Average with higher weight on current score
            weights = np.array([0.7] + [0.3 / self.smooth] * self.smooth)
            base_score = base_score * weights[0] + sum(s * w for s, w in zip(prev_scores, weights[1:]))
        
        return float(np.clip(base_score, -1.0, 1.0)) 