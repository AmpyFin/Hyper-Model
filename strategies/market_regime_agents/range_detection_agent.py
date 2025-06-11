"""
Range Detection Agent
~~~~~~~~~~~~~~~~~
Detects whether the market is in a ranging (sideways) or trending regime.
During ranging markets, prices move within a defined range without a clear directional bias,
while trending markets show a persistent directional move.

Logic:
1. Identify price channels and trading ranges
2. Calculate directional efficiency ratio (DER)
3. Apply range detection algorithms (Bollinger BandWidth, ADX)
4. Compare recent highs and lows to detect range boundaries
5. Determine the probability of range continuation vs breakout

Input: OHLCV DataFrame. Output ∈ [-1, +1] where:
* Values near 0: Strong ranging market
* Values near +1: Strong uptrend
* Values near -1: Strong downtrend
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict


class RangeDetectionAgent:
    def __init__(
        self,
        range_period: int = 20,        # Period for range detection
        bollinger_width: int = 20,     # Period for Bollinger BandWidth
        bollinger_stdev: float = 2.0,  # Standard deviation for Bollinger Bands
        der_period: int = 14,          # Period for Directional Efficiency Ratio
        adx_period: int = 14,          # Period for ADX calculation
        range_threshold: float = 0.3,   # Threshold to identify ranging markets
        trend_threshold: float = 0.6    # Threshold to identify trending markets
    ):
        self.range_period = range_period
        self.bollinger_width = bollinger_width
        self.bollinger_stdev = bollinger_stdev
        self.der_period = der_period
        self.adx_period = adx_period
        self.range_threshold = range_threshold
        self.trend_threshold = trend_threshold
        self.latest_signal = 0.0
        self.latest_metrics = {}
        
    def _calculate_range_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate various range detection metrics"""
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        result = {}
        
        # Calculate Bollinger Bands
        if len(df_copy) >= self.bollinger_width:
            df_copy['bb_middle'] = df_copy['close'].rolling(window=self.bollinger_width).mean()
            bb_stdev = df_copy['close'].rolling(window=self.bollinger_width).std()
            df_copy['bb_upper'] = df_copy['bb_middle'] + (self.bollinger_stdev * bb_stdev)
            df_copy['bb_lower'] = df_copy['bb_middle'] - (self.bollinger_stdev * bb_stdev)
            
            # Bollinger BandWidth (normalized)
            df_copy['bb_width'] = (df_copy['bb_upper'] - df_copy['bb_lower']) / df_copy['bb_middle']
            
            # Bandwidth percentile over a longer period
            lookback = min(len(df_copy), 100)
            bandwidth_series = df_copy['bb_width'].iloc[-lookback:]
            result['bb_width'] = df_copy['bb_width'].iloc[-1]
            result['bb_width_percentile'] = sum(bandwidth_series <= result['bb_width']) / len(bandwidth_series)
            
            # Price position within bands (0 = lower band, 1 = upper band)
            df_copy['bb_position'] = (df_copy['close'] - df_copy['bb_lower']) / (df_copy['bb_upper'] - df_copy['bb_lower'])
            result['bb_position'] = df_copy['bb_position'].iloc[-1]
        
        # Calculate Directional Efficiency Ratio (DER)
        # Measures how efficiently price moves in a direction
        if len(df_copy) >= self.der_period:
            # Net price movement over period
            net_movement = abs(df_copy['close'].iloc[-1] - df_copy['close'].iloc[-self.der_period])
            
            # Sum of individual price movements
            price_series = df_copy['close'].iloc[-self.der_period:]
            path_length = sum(abs(price_series.diff().dropna()))
            
            # DER = net movement / path length
            # Range: [0, 1] where 1 = perfect trend, 0 = perfect range
            if path_length > 0:
                result['der'] = net_movement / path_length
            else:
                result['der'] = 0.0
        
        # Calculate ADX for trend strength
        if len(df_copy) >= self.adx_period * 2:
            # True Range
            df_copy['high_minus_low'] = df_copy['high'] - df_copy['low']
            df_copy['high_minus_prev_close'] = abs(df_copy['high'] - df_copy['close'].shift(1))
            df_copy['low_minus_prev_close'] = abs(df_copy['low'] - df_copy['close'].shift(1))
            df_copy['tr'] = df_copy[['high_minus_low', 'high_minus_prev_close', 'low_minus_prev_close']].max(axis=1)
            
            # Directional Movement
            df_copy['up_move'] = df_copy['high'] - df_copy['high'].shift(1)
            df_copy['down_move'] = df_copy['low'].shift(1) - df_copy['low']
            
            # Positive and negative DM
            df_copy['+dm'] = np.where(
                (df_copy['up_move'] > df_copy['down_move']) & (df_copy['up_move'] > 0),
                df_copy['up_move'],
                0
            )
            df_copy['-dm'] = np.where(
                (df_copy['down_move'] > df_copy['up_move']) & (df_copy['down_move'] > 0),
                df_copy['down_move'],
                0
            )
            
            # Smoothed TR and DM
            df_copy['tr_smooth'] = df_copy['tr'].rolling(window=self.adx_period).mean()
            df_copy['+dm_smooth'] = df_copy['+dm'].rolling(window=self.adx_period).mean()
            df_copy['-dm_smooth'] = df_copy['-dm'].rolling(window=self.adx_period).mean()
            
            # Directional Indicators
            df_copy['+di'] = 100 * (df_copy['+dm_smooth'] / df_copy['tr_smooth'])
            df_copy['-di'] = 100 * (df_copy['-dm_smooth'] / df_copy['tr_smooth'])
            
            # Directional Index
            df_copy['di_diff'] = abs(df_copy['+di'] - df_copy['-di'])
            df_copy['di_sum'] = df_copy['+di'] + df_copy['-di']
            df_copy['dx'] = 100 * (df_copy['di_diff'] / df_copy['di_sum'])
            
            # Average Directional Index
            df_copy['adx'] = df_copy['dx'].rolling(window=self.adx_period).mean()
            
            # Store latest values
            result['adx'] = df_copy['adx'].iloc[-1]
            result['+di'] = df_copy['+di'].iloc[-1]
            result['-di'] = df_copy['-di'].iloc[-1]
            
            # Trend direction based on DI comparison
            result['trend_direction'] = 1 if result['+di'] > result['-di'] else -1
        
        # Check range stability
        if len(df_copy) >= self.range_period:
            # Get highest high and lowest low in range period
            period_high = df_copy['high'].iloc[-self.range_period:].max()
            period_low = df_copy['low'].iloc[-self.range_period:].min()
            
            # Range height as percentage
            result['range_height'] = (period_high - period_low) / period_low
            
            # Check if price is forming higher lows and lower highs
            # (characteristic of range contraction)
            half_period = self.range_period // 2
            if len(df_copy) >= self.range_period + half_period:
                early_range_high = df_copy['high'].iloc[-self.range_period:-half_period].max()
                early_range_low = df_copy['low'].iloc[-self.range_period:-half_period].min()
                late_range_high = df_copy['high'].iloc[-half_period:].max()
                late_range_low = df_copy['low'].iloc[-half_period:].min()
                
                result['contracting_range'] = (late_range_high < early_range_high and 
                                               late_range_low > early_range_low)
        
        return result
    
    def _evaluate_range_regime(self, metrics: Dict) -> float:
        """
        Evaluate the range vs trend regime based on calculated metrics
        Returns a signal from -1 (strong downtrend) to +1 (strong uptrend)
        with 0 indicating a ranging market
        """
        # Default to neutral if insufficient data
        if not metrics:
            return 0.0
        
        # Collect evidence for ranging market
        range_evidence = 0.0
        range_count = 0
        
        # Low DER suggests ranging market
        if 'der' in metrics:
            range_score = 1.0 - metrics['der']  # Invert DER (1 = perfect range)
            range_evidence += range_score
            range_count += 1
            
        # Low ADX suggests ranging market
        if 'adx' in metrics:
            # Scale ADX from [0, 50] to [1, 0] (lower ADX = stronger range)
            range_score = max(0.0, 1.0 - (metrics['adx'] / 50.0))
            range_evidence += range_score
            range_count += 1
            
        # Low Bollinger BandWidth suggests ranging market
        if 'bb_width_percentile' in metrics:
            # Lower percentiles suggest narrowing bands (range)
            range_score = 1.0 - metrics['bb_width_percentile']
            range_evidence += range_score
            range_count += 1
            
        # Contracting range suggests consolidation
        if 'contracting_range' in metrics and metrics['contracting_range']:
            range_evidence += 1.0
            range_count += 1
        
        # Calculate average range evidence
        range_score = range_evidence / max(1, range_count)
        
        # Determine regime
        if range_score > self.range_threshold:
            # Strong ranging market
            return 0.0
        elif 'adx' in metrics and metrics['adx'] > 25 and range_score < self.trend_threshold:
            # Trending market - determine direction
            if 'trend_direction' in metrics:
                # Direction strength based on ADX and trend threshold
                trend_strength = min(1.0, metrics['adx'] / 50.0)
                return metrics['trend_direction'] * trend_strength
        
        # Weak signal, biased slightly toward range
        return 0.0
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to identify the current range vs trend regime
        """
        # Need enough bars for calculation
        required_bars = max(self.range_period, self.bollinger_width, self.der_period, self.adx_period * 2)
        if len(historical_df) < required_bars:
            self.latest_signal = 0.0
            return
        
        # Calculate range metrics
        self.latest_metrics = self._calculate_range_metrics(historical_df)
        
        # Evaluate range vs trend regime
        self.latest_signal = self._evaluate_range_regime(self.latest_metrics)
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict the range/trend regime signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Values near 0 indicate a ranging market
          * Positive values indicate an uptrend
          * Negative values indicate a downtrend
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Range Detection Agent" 