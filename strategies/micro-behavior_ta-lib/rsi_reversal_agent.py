"""
RSI Reversal Agent
~~~~~~~~~~~~~~~~~
Detects overbought/oversold conditions using Relative Strength Index (RSI)
and scores potential price reversals. Generates stronger signals when RSI
reaches extreme values or shows divergence with price.

Logic:
1. Calculate RSI(14) for price series
2. Assign scores based on RSI levels:
   - RSI > 70 → SELL signal (negative score)
   - RSI < 30 → BUY signal (positive score)
3. Amplify signals when RSI diverges from price (RSI makes lower high while 
   price makes higher high, or RSI makes higher low while price makes lower low)
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

class RSI_Reversal_Agent:
    def __init__(
        self, 
        rsi_window: int = 14,
        overbought_threshold: float = 70.0,
        oversold_threshold: float = 30.0,
        divergence_lookback: int = 5
    ):
        self.rsi_window = rsi_window
        self.overbought = overbought_threshold
        self.oversold = oversold_threshold
        self.div_lookback = divergence_lookback
    
    def _check_divergence(self, close: pd.Series, rsi: pd.Series) -> tuple[bool, bool]:
        """Check for bullish or bearish divergence in the last few bars"""
        if len(close) < self.div_lookback * 2:
            return False, False
            
        # Get local extrema (peaks and troughs)
        window = self.div_lookback
        price_highs = close.rolling(window=window, center=True).max()
        price_lows = close.rolling(window=window, center=True).min()
        rsi_highs = rsi.rolling(window=window, center=True).max()
        rsi_lows = rsi.rolling(window=window, center=True).min()
        
        # Last window bars
        last_idx = len(close) - 1
        end_idx = last_idx
        start_idx = max(0, end_idx - (window * 2))
        
        # Check for bearish divergence (price higher high, RSI lower high)
        bearish = False
        if (close.iloc[end_idx] >= price_highs.iloc[start_idx:end_idx].max() and 
            rsi.iloc[end_idx] < rsi_highs.iloc[start_idx:end_idx].max()):
            bearish = True
            
        # Check for bullish divergence (price lower low, RSI higher low)
        bullish = False
        if (close.iloc[end_idx] <= price_lows.iloc[start_idx:end_idx].min() and
            rsi.iloc[end_idx] > rsi_lows.iloc[start_idx:end_idx].min()):
            bullish = True
            
        return bullish, bearish
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        # No training needed for this rule-based agent
        pass
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        if len(historical_df) < self.rsi_window + self.div_lookback:
            raise ValueError(f"Need at least {self.rsi_window + self.div_lookback} rows")
        
        # Calculate RSI
        close = historical_df["close"]
        rsi_indicator = ta.momentum.RSIIndicator(close, window=self.rsi_window)
        rsi = rsi_indicator.rsi()
        
        # Get latest RSI value
        current_rsi = rsi.iloc[-1]
        
        # Base score from RSI levels
        if current_rsi > self.overbought:
            # Overbought - bearish
            # Normalize to [-1, 0] range based on how extreme RSI is
            base_score = -np.tanh((current_rsi - self.overbought) / 15.0)
        elif current_rsi < self.oversold:
            # Oversold - bullish
            # Normalize to [0, 1] range based on how extreme RSI is
            base_score = np.tanh((self.oversold - current_rsi) / 15.0)
        else:
            # Neutral zone - weak signal proportional to distance from 50
            base_score = (50 - current_rsi) / 50.0 * 0.5
            
        # Check for divergence to amplify signals
        bullish_div, bearish_div = self._check_divergence(close, rsi)
        
        # Amplify signals when divergence is detected
        if bearish_div and base_score < 0:
            base_score *= 1.5  # Amplify bearish signal
        elif bullish_div and base_score > 0:
            base_score *= 1.5  # Amplify bullish signal
            
        # Ensure output is within [-1, +1]
        return float(np.clip(base_score, -1.0, 1.0)) 