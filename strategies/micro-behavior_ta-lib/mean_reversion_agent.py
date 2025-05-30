"""
Mean Reversion Agent
~~~~~~~~~~~~~~~~~~~
Identifies statistical extremes in price action relative to historical
volatility bands and scores potential mean reversion opportunities.
Uses multiple timeframes to detect overextended price moves that are
likely to revert back to the mean.

Logic:
1. Calculate price distance from moving averages (20, 50, 200 period)
2. Compute z-score of current price relative to historical volatility
3. Generate signals when:
   - Price is significantly extended from moving averages (z-score > 2)
   - Rate of change is extreme relative to historical norms
   - The extension is not supported by volume or momentum
4. Scale signals based on:
   - Degree of extension (more extreme = stronger signal)
   - Presence of reversal patterns
   - Volume and momentum confirmation
   - Volatility regime

Dependencies:
- pandas
- numpy
- ta-lib
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

class Mean_Reversion_Agent:
    def __init__(
        self,
        lookback: int = 20,  # Base lookback period
        z_score_threshold: float = 2.0,  # Z-score threshold for signals
        vol_lookback: int = 10,  # Volatility lookback
        noise_reduction: float = 0.02  # Base noise level for randomization
    ):
        self.lookback = lookback
        self.z_threshold = z_score_threshold
        self.vol_lookback = vol_lookback
        self.noise_reduction = noise_reduction
        
    def _calculate_zscore(
        self,
        series: pd.Series,
        lookback: int,
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """Calculate rolling z-score with proper error handling"""
        if min_periods is None:
            min_periods = lookback // 2
            
        # Calculate rolling mean and std with minimum periods
        rolling_mean = series.rolling(lookback, min_periods=min_periods).mean()
        rolling_std = series.rolling(lookback, min_periods=min_periods).std()
        
        # Handle zero/nan standard deviation
        rolling_std = rolling_std.replace(0, np.nan)
        
        # Calculate z-score
        z_score = (series - rolling_mean) / rolling_std
        
        # Replace infinite values with max/min
        z_score = z_score.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill small gaps using ffill() instead of deprecated fillna(method='ffill')
        z_score = z_score.ffill(limit=3)
        
        return z_score
        
    def _calculate_volatility_regime(
        self,
        returns: pd.Series,
        current_vol: float,
        hist_vol: float
    ) -> float:
        """Calculate volatility regime score [-1, 1]"""
        if pd.isna(current_vol) or pd.isna(hist_vol) or hist_vol == 0:
            return 0.0
            
        # Compare current to historical volatility
        vol_ratio = current_vol / hist_vol
        
        # Score from -1 (low vol) to 1 (high vol)
        regime_score = np.tanh(vol_ratio - 1)
        
        return regime_score
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        # No training needed for this rule-based agent
        pass
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        if len(historical_df) < self.lookback:
            return 0.0
            
        # Calculate returns
        returns = historical_df['close'].pct_change().fillna(0)
        
        # Calculate volatility measures
        current_vol = returns.rolling(self.vol_lookback).std().iloc[-1]
        hist_vol = returns.rolling(self.lookback).std().iloc[-1]
        
        # Calculate regime score
        regime_score = self._calculate_volatility_regime(returns, current_vol, hist_vol)
        
        # Calculate z-scores for different lookbacks
        z_scores = {}
        for period in [self.lookback // 2, self.lookback, self.lookback * 2]:
            z_scores[period] = self._calculate_zscore(
                historical_df['close'],
                period,
                min_periods=max(5, period // 4)
            )
            
        # Get current z-scores
        current_z = {
            period: z_score.iloc[-1]
            for period, z_score in z_scores.items()
            if not pd.isna(z_score.iloc[-1])
        }
        
        if not current_z:
            print("No valid z-scores calculated")
            return 0.0
            
        # Calculate weighted average z-score
        weights = {
            self.lookback // 2: 0.5,  # Short-term
            self.lookback: 0.3,      # Medium-term
            self.lookback * 2: 0.2   # Long-term
        }
        
        weighted_z = 0.0
        total_weight = 0.0
        
        for period, z in current_z.items():
            weight = weights.get(period, 0.0)
            weighted_z += z * weight
            total_weight += weight
            
        if total_weight > 0:
            weighted_z /= total_weight
        else:
            print("No valid weights for z-score calculation")
            return 0.0
            
        print(f"Weighted z-score: {weighted_z:.4f}")
        
        # Only generate signals for significant deviations
        if abs(weighted_z) < self.z_threshold:
            print("Z-score within normal range")
            return 0.0
            
        # Calculate base signal from z-score
        base_signal = -np.tanh(weighted_z / self.z_threshold)  # Reverse z-score direction
        print(f"Base signal: {base_signal:.4f}")
        
        # Adjust signal strength based on regime
        regime_factor = 1.0 - abs(regime_score) * 0.5  # Reduce signals in extreme regimes
        signal = base_signal * regime_factor
        print(f"Regime adjusted signal: {signal:.4f}")
        
        # Add momentum confirmation
        momentum = returns.iloc[-5:].mean()  # 5-period momentum
        if np.sign(momentum) == np.sign(signal):
            signal *= 1.2  # Boost aligned signals
        else:
            signal *= 0.8  # Reduce contrary signals
            
        print(f"Momentum adjusted signal: {signal:.4f}")
        
        # Add volume confirmation if available
        if 'volume' in historical_df.columns:
            vol_ratio = (
                historical_df['volume'].iloc[-5:].mean() /
                historical_df['volume'].iloc[-20:-5].mean()
            )
            if vol_ratio > 1.1:  # Volume increasing
                signal *= 1.1
            elif vol_ratio < 0.9:  # Volume decreasing
                signal *= 0.9
                
            print(f"Volume adjusted signal: {signal:.4f}")
            
        # Add candlestick pattern confirmation
        last_candle = historical_df.iloc[-1]
        body = abs(last_candle['close'] - last_candle['open'])
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        
        # Check for reversal candles
        if signal > 0:  # Bullish signal
            if last_candle['close'] > last_candle['open'] and lower_wick > body:
                signal *= 1.2  # Hammer pattern
        else:  # Bearish signal
            if last_candle['close'] < last_candle['open'] and upper_wick > body:
                signal *= 1.2  # Shooting star pattern
                
        print(f"Pattern adjusted signal: {signal:.4f}")
        
        # Add controlled randomness
        noise = np.random.normal(0, self.noise_reduction)
        signal *= (1.0 + noise)
        
        # Add ticker-specific randomization
        ticker_hash = sum(ord(c) for c in historical_df.iloc[-1].name) if isinstance(historical_df.iloc[-1].name, str) else 0
        np.random.seed(ticker_hash)
        ticker_noise = np.random.normal(0, 0.01)  # 1% ticker-specific noise
        signal *= (1.0 + ticker_noise)
        
        print(f"Final signal: {signal:.4f}")
        return float(np.clip(signal, -1.0, 1.0)) 