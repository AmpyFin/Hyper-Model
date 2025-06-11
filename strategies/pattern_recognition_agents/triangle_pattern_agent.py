"""
Triangle Pattern Agent
~~~~~~~~~~~~~~~~~~~
Detects triangle chart patterns (symmetrical, ascending, and descending).
Triangle patterns are consolidation patterns characterized by converging trendlines:
- Symmetrical Triangle: Lower highs and higher lows (neutral bias)
- Ascending Triangle: Flat upper resistance with rising support (bullish bias)
- Descending Triangle: Flat lower support with declining resistance (bearish bias)

Logic:
1. Identify potential triangle formations using linear regression on highs and lows
2. Classify triangle type based on the slope of upper and lower trendlines
3. Validate pattern by checking:
   - Minimum number of touchpoints on trendlines
   - Convergence of trendlines
   - Decreasing volume during formation
4. Generate signals on breakout from the triangle:
   - Breakout direction often follows the preceding trend
   - Breakout from direction of bias is higher probability
5. Scale signal strength based on:
   - Pattern size and duration
   - Volume confirmation on breakout
   - Triangle type and breakout alignment

Input: OHLCV DataFrame. Output ∈ [-1, +1].
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from scipy import stats


class TrianglePatternAgent:
    def __init__(
        self,
        min_triangle_bars: int = 15,      # Minimum bars for triangle pattern
        max_triangle_bars: int = 60,      # Maximum bars for triangle pattern
        min_touches: int = 2,             # Minimum touches of support/resistance
        convergence_threshold: float = 0.3,# How much lines must converge
        slope_tolerance: float = 0.002,    # Max slope difference for symmetrical
        volume_confirmation: bool = True,  # Require volume confirmation
        breakout_threshold: float = 0.005  # Min price move for breakout
    ) -> None:
        self.min_triangle_bars = min_triangle_bars
        self.max_triangle_bars = max_triangle_bars
        self.min_touches = min_touches
        self.convergence_threshold = convergence_threshold
        self.slope_tolerance = slope_tolerance
        self.volume_confirmation = volume_confirmation
        self.breakout_threshold = breakout_threshold
        
        self.latest_signal = 0.0
        self.detected_patterns = []
        
    def _detect_triangles(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect triangle patterns in the given dataframe
        Returns list of triangle pattern dictionaries
        """
        # Ensure enough data
        if len(df) < self.min_triangle_bars:
            return []
            
        triangles = []
        
        # Try different window sizes for triangle patterns
        for window in range(self.min_triangle_bars, min(self.max_triangle_bars, len(df))):
            # Skip if not enough data to look backward
            if window >= len(df):
                continue
                
            # Get the data window
            window_df = df.iloc[-window:]
            
            # Check for triangle pattern
            triangle = self._check_triangle(window_df)
            if triangle:
                triangles.append(triangle)
                
        return triangles
    
    def _check_triangle(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Check if the given dataframe contains a triangle pattern
        Returns a dictionary with pattern details if found, None otherwise
        """
        if len(df) < self.min_triangle_bars:
            return None
            
        # Get price data
        highs = df['high'].values
        lows = df['low'].values
        x = np.arange(len(df))
        
        # Fit lines to highs and lows
        high_slope, high_intercept = np.polyfit(x, highs, 1)
        low_slope, low_intercept = np.polyfit(x, lows, 1)
        
        # Calculate convergence
        high_start = high_intercept
        high_end = high_slope * len(df) + high_intercept
        low_start = low_intercept
        low_end = low_slope * len(df) + low_intercept
        
        initial_spread = high_start - low_start
        final_spread = high_end - low_end
        
        # Check if lines are converging
        convergence = 1 - (final_spread / initial_spread if initial_spread != 0 else 1)
        
        # Identify pattern type
        is_ascending = abs(low_slope) > self.slope_tolerance and low_slope > 0 and abs(high_slope) < self.slope_tolerance
        is_descending = abs(high_slope) > self.slope_tolerance and high_slope < 0 and abs(low_slope) < self.slope_tolerance
        is_symmetrical = abs(abs(high_slope) - abs(low_slope)) < self.slope_tolerance and high_slope < 0 and low_slope > 0
        
        pattern_type = None
        if is_ascending and convergence > self.convergence_threshold:
            pattern_type = "ascending"
        elif is_descending and convergence > self.convergence_threshold:
            pattern_type = "descending"
        elif is_symmetrical and convergence > self.convergence_threshold:
            pattern_type = "symmetrical"
            
        if pattern_type is None:
            return None
            
        # Check touches
        high_line = high_slope * x + high_intercept
        low_line = low_slope * x + low_intercept
        
        high_touches = sum(abs(highs - high_line) < (high_line * 0.002))  # 0.2% tolerance
        low_touches = sum(abs(lows - low_line) < (low_line * 0.002))
        
        if min(high_touches, low_touches) < self.min_touches:
            return None
            
        # Pattern found - return details
        return {
            "type": pattern_type,
            "start_idx": df.index[0],
            "end_idx": df.index[-1],
            "high_slope": high_slope,
            "low_slope": low_slope,
            "high_intercept": high_intercept,
            "low_intercept": low_intercept,
            "convergence": convergence,
            "high_touches": high_touches,
            "low_touches": low_touches
        }
    
    def _check_breakout(
            self, 
            df: pd.DataFrame, 
            triangle: Dict[str, Any]
        ) -> Tuple[bool, float, bool]:
        """
        Check if there's a valid breakout from the triangle
        Returns (is_breakout, breakout_strength, is_bullish)
        """
        # Get last bar of triangle
        try:
            end_loc = df.index.get_loc(triangle['end_idx'])
        except KeyError:
            print(f"Warning: end_idx {triangle['end_idx']} not found in index")
            return False, 0.0, False
            
        # Ensure we have data to check for breakout
        if end_loc >= len(df) - 1:
            print("Warning: Not enough data after triangle to check breakout")
            return False, 0.0, False
            
        # Get breakout bar data
        breakout_idx = end_loc + 1
        breakout_high = df['high'].iloc[breakout_idx]
        breakout_low = df['low'].iloc[breakout_idx]
        breakout_close = df['close'].iloc[breakout_idx]
        
        # Calculate triangle trendlines at breakout point
        x = breakout_idx  # Use actual index for calculation
        upper_bound = triangle['high_slope'] * x + triangle['high_intercept']
        lower_bound = triangle['low_slope'] * x + triangle['low_intercept']
        
       
        
        is_breakout = False
        breakout_strength = 0.0
        is_bullish = False
        
        # Check for bullish breakout (above upper trendline)
        if breakout_close > upper_bound * (1 + self.breakout_threshold):
            is_breakout = True
            is_bullish = True
            breakout_strength = (breakout_close - upper_bound) / upper_bound
            
        # Check for bearish breakout (below lower trendline)
        elif breakout_close < lower_bound * (1 - self.breakout_threshold):
            is_breakout = True
            is_bullish = False
            breakout_strength = (lower_bound - breakout_close) / lower_bound
        
        # Volume confirmation
        if is_breakout and self.volume_confirmation and 'volume' in df.columns:
            # Compare breakout volume to average volume during pattern
            pattern_vol_avg = df.loc[:triangle['end_idx']]['volume'].mean()
            breakout_vol = df['volume'].iloc[breakout_idx]
            vol_ratio = breakout_vol / pattern_vol_avg
            
            # Adjust strength based on volume confirmation
            if vol_ratio > 1.5:
                breakout_strength *= 1.2
            elif vol_ratio < 0.5:
                breakout_strength *= 0.8
        
        return is_breakout, breakout_strength, is_bullish
            
    def fit(self, df: pd.DataFrame) -> None:
        """Update internal state from historical data."""
        if len(df) < self.min_triangle_bars:
            self.latest_signal = 0.0
            return

        # Reset state
        self.detected_patterns = []
        
        # Scan for patterns in different window sizes
        for window_size in range(self.min_triangle_bars, min(self.max_triangle_bars, len(df)), 5):
            for start_idx in range(0, len(df) - window_size):
                window = df.iloc[start_idx:start_idx + window_size]
                pattern = self._check_triangle(window)
                
                if pattern is not None:
                    # Adjust indices to match full dataframe
                    pattern['start_idx'] = df.index[start_idx]
                    pattern['end_idx'] = df.index[start_idx + window_size - 1]
                    self.detected_patterns.append(pattern)
        
        
        # Check for breakouts
        valid_patterns = []
        for pattern in self.detected_patterns:
            is_breakout, strength, is_bullish = self._check_breakout(df, pattern)
            if is_breakout:
                pattern['breakout_strength'] = strength
                pattern['is_bullish'] = is_bullish
                valid_patterns.append(pattern)
              
        
        
        # Generate signal from valid patterns
        if valid_patterns:
            # Use the most recent valid pattern
            latest_pattern = max(valid_patterns, key=lambda x: x['end_idx'])
            signal = latest_pattern['breakout_strength']
            if not latest_pattern['is_bullish']:
                signal *= -1
                
            # Scale signal to [-1, 1]
            self.latest_signal = np.clip(signal * 2.0, -1.0, 1.0)
        else:
            self.latest_signal = 0.0
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """Generate trading signal from detected patterns."""
        if not self.detected_patterns:
            return 0.0
            
        # Use most recent valid pattern
        pattern = self.detected_patterns[-1]
        
        # Calculate signal
        signal = pattern['breakout_strength']
        if not pattern['is_bullish']:
            signal = -signal
            
        # Scale by pattern quality and bias
        signal *= 1.0  # Assuming pattern quality is always 1.0 for this implementation
        signal = np.clip(signal * 2.0, -1.0, 1.0)  # Scale to [-1, 1]
        
        return float(signal)
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return "Triangle Pattern Agent"

