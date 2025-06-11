"""
Flag Pattern Agent
~~~~~~~~~~~~~~~~
Detects flag chart patterns (both bullish and bearish).
A flag is a continuation pattern that occurs after a strong price movement (the pole),
followed by a consolidation period (the flag) that typically moves counter to the 
preceding trend, before continuing in the original direction.

Logic:
1. Identify strong directional moves (poles) using rate of change
2. Look for consolidation periods (flags) following the pole
3. Detect the characteristic parallel channel of the flag
4. Generate signals on breakout from the flag pattern:
   - Bullish flag: Upward breakout from downward-sloping parallel channel after uptrend
   - Bearish flag: Downward breakout from upward-sloping parallel channel after downtrend
5. Scale signal strength based on:
   - Pole height (strength of the preceding move)
   - Flag quality (clear boundaries, decreasing volume)
   - Breakout characteristics (volume expansion, price velocity)

Input: OHLCV DataFrame. Output ∈ [-1, +1].
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy import stats


class FlagPatternAgent:
    def __init__(
        self,
        pole_roc_threshold: float = 0.002,  # Min rate of change to qualify as pole (0.2% for intraday)
        flag_max_bars: int = 30,          # Max bars a flag consolidation can last (increased for intraday)
        flag_min_bars: int = 10,          # Min bars a flag consolidation must last (increased for intraday)
        channel_tolerance: float = 0.15,   # Max variance from parallel channel lines (relaxed for intraday)
        volume_confirmation: bool = True,  # Use volume in signal generation
        breakout_threshold: float = 0.001  # Min breakout % to confirm pattern completion (0.1% for intraday)
    ):
        self.pole_roc_threshold = pole_roc_threshold
        self.flag_max_bars = flag_max_bars
        self.flag_min_bars = flag_min_bars
        self.channel_tolerance = channel_tolerance
        self.use_volume = volume_confirmation
        self.breakout_threshold = breakout_threshold
        self.detected_patterns = []
        
    def _find_pole(self, df: pd.DataFrame) -> List[Tuple[int, int, bool]]:
        """
        Find potential poles (strong directional moves)
        Returns list of (start_idx, end_idx, is_bullish)
        """
        poles = []
        
        # Ensure enough data
        if len(df) < 5:
            return poles
            
        # Calculate returns and volume ratios
        returns = df['close'].pct_change()
        volumes = df['volume'] if 'volume' in df.columns else None
        
        # Look for strong moves in the first half of the data
        max_start = len(df) // 2  # Only look in first half
        
        # Look for strong moves
        for i in range(min(max_start, len(df) - 5)):
            # Calculate cumulative return over 5 bars
            cum_return = (df['close'].iloc[i+5] / df['close'].iloc[i]) - 1
            
            # Check if move is strong enough
            if abs(cum_return) >= self.pole_roc_threshold:
                # Check volume trend
                vol_trend = 1.0
                if volumes is not None:
                    # Compare average volume during potential pole to previous period
                    pole_vol = volumes.iloc[i:i+5].mean()
                    prev_vol = volumes.iloc[max(0, i-5):i].mean() if i > 0 else pole_vol
                    vol_trend = pole_vol / prev_vol
                
                # Relaxed volume confirmation for intraday (only require 10% increase)
                if vol_trend > 1.1:  
                    is_bullish = cum_return > 0
                    poles.append((i, i+5, is_bullish))
        
        # Return top 3 strongest poles instead of just one
        if poles:
            sorted_poles = sorted(poles, key=lambda p: abs(
                (df['close'].iloc[p[1]] / df['close'].iloc[p[0]]) - 1
            ), reverse=True)
            return sorted_poles[:3]  # Return up to 3 strongest poles
            
        return poles
    
    def _find_flag_channel(
            self, 
            df: pd.DataFrame, 
            start_idx: int, 
            is_bullish: bool
        ) -> Optional[Tuple[int, List[float], List[float], float]]:
        """
        Find flag channel after pole
        Returns (end_idx, upper_channel, lower_channel, channel_quality)
        """
        # Ensure we have enough data
        if start_idx + self.flag_min_bars >= len(df):
            return None
            
        max_end = min(start_idx + self.flag_max_bars, len(df))
        
        # Get appropriate price series
        highs = df['high'].values[start_idx:max_end]
        lows = df['low'].values[start_idx:max_end]
        closes = df['close'].values[start_idx:max_end]
        
        best_quality = 0
        best_result = None
        
        # Try different potential flag lengths
        for end_idx in range(start_idx + self.flag_min_bars, max_end):
            # Calculate flag length
            flag_length = end_idx - start_idx
            
            # For a bullish flag, we expect a downward-sloping channel
            # For a bearish flag, we expect an upward-sloping channel
            expected_slope = -1 if is_bullish else 1
            
            # Fit lines to highs and lows
            x = np.array(range(flag_length))
            
            # Linear regression for upper channel (highs)
            high_slope, high_intercept, high_r, _, _ = stats.linregress(x, highs[:flag_length])
            
            # Linear regression for lower channel (lows)
            low_slope, low_intercept, low_r, _, _ = stats.linregress(x, lows[:flag_length])
            
            # Calculate channel lines
            upper_channel = high_intercept + high_slope * x
            lower_channel = low_intercept + low_slope * x
            
            # More flexible slope direction check for intraday
            if is_bullish:
                # For bullish flags, allow slightly upward slopes but prefer downward
                if high_slope > 0.001 or low_slope > 0.001:  # Small positive threshold
                    continue
            else:
                # For bearish flags, allow slightly downward slopes but prefer upward
                if high_slope < -0.001 or low_slope < -0.001:  # Small negative threshold
                    continue
                
            # Check if channel lines are roughly parallel (more flexible for intraday)
            slope_diff = abs(high_slope - low_slope)
            avg_slope = (abs(high_slope) + abs(low_slope)) / 2
            if slope_diff / (avg_slope + 1e-6) > self.channel_tolerance:  # Added small epsilon to avoid division by zero
                continue
                
            # Calculate how well prices stay within channel (more flexible)
            prices_in_channel = np.sum((highs[:flag_length] <= upper_channel * 1.002) &  # Allow 0.2% breach
                                     (lows[:flag_length] >= lower_channel * 0.998))   # Allow 0.2% breach
            containment_ratio = prices_in_channel / flag_length
            
            # Calculate channel quality score
            channel_quality = containment_ratio
            
            # Check volume characteristics if enabled (more flexible)
            if self.use_volume and 'volume' in df.columns:
                # Volume should generally decrease in flag but be more flexible
                volumes = df['volume'].values[start_idx:end_idx]
                volume_slope = np.polyfit(x, volumes, 1)[0]
                
                # Less penalty for non-decreasing volume
                if volume_slope >= 0:
                    channel_quality *= 0.9  # Reduced penalty
            
            # Check if price range is narrowing (more flexible)
            price_ranges = highs[:flag_length] - lows[:flag_length]
            range_slope = np.polyfit(x, price_ranges, 1)[0]
            if range_slope > 0:  # Range should be decreasing but be more flexible
                channel_quality *= 0.9  # Reduced penalty
            
            # Check for potential breakout after flag
            if end_idx < len(df) - 5:
                next_5_closes = df['close'].values[end_idx:end_idx+5]
                next_5_volumes = df['volume'].values[end_idx:end_idx+5]
                
                # Price should move in expected direction after flag
                price_change = (next_5_closes[-1] - next_5_closes[0]) / next_5_closes[0]
                if (is_bullish and price_change > 0) or (not is_bullish and price_change < 0):
                    channel_quality *= 1.1  # Reduced reward
                
                # Volume should increase after flag
                flag_vol_avg = np.mean(volumes[-5:])
                breakout_vol_avg = np.mean(next_5_volumes)
                if breakout_vol_avg > flag_vol_avg:
                    channel_quality *= 1.1  # Reduced reward
            
            # Update best result if this is better
            if channel_quality > best_quality:
                best_quality = channel_quality
                best_result = (end_idx, upper_channel.tolist(), lower_channel.tolist(), channel_quality)
                
        return best_result
    
    def _check_breakout(
            self, 
            df: pd.DataFrame, 
            flag_end: int, 
            upper_channel: List[float], 
            lower_channel: List[float], 
            is_bullish: bool
        ) -> Tuple[bool, float]:
        """
        Check if there's a valid breakout from the flag
        Returns (is_breakout, breakout_strength)
        """
        # Ensure we have data to check for breakout
        if flag_end >= len(df) - 1:
            return False, 0.0
            
        # Look at next few bars after flag
        lookforward = min(5, len(df) - flag_end - 1)
        if lookforward < 1:
            return False, 0.0
            
        # Get channel boundaries at end of flag
        upper_bound = upper_channel[-1]
        lower_bound = lower_channel[-1]
        
        # Get breakout bar data
        breakout_data = df.iloc[flag_end:flag_end+lookforward]
        breakout_close = breakout_data['close'].iloc[-1]
        max_high = breakout_data['high'].max()
        min_low = breakout_data['low'].min()
     
        
        is_breakout = False
        breakout_strength = 0.0
        
        if is_bullish:
            # Bullish breakout should be above upper channel
            if max_high > upper_bound * (1 + self.breakout_threshold):
                is_breakout = True
                # Calculate breakout strength as percentage above channel
                breakout_strength = (breakout_close - upper_bound) / upper_bound
        else:
            # Bearish breakout should be below lower channel
            if min_low < lower_bound * (1 - self.breakout_threshold):
                is_breakout = True
                # Calculate breakout strength as percentage below channel
                breakout_strength = (lower_bound - breakout_close) / lower_bound
        
        # Check volume confirmation if enabled
        if is_breakout and self.use_volume and 'volume' in df.columns:
            # Compare breakout volume to flag volume
            flag_start = flag_end - len(upper_channel)
            avg_flag_volume = df['volume'].iloc[flag_start:flag_end].mean()
            breakout_volume = breakout_data['volume'].mean()
            
            # Volume should increase on breakout
            vol_ratio = breakout_volume / avg_flag_volume
            
            if vol_ratio > 1.5:
                # Strong volume confirmation
                breakout_strength *= 1.2
            elif vol_ratio < 1.0:
                # Weak volume
                breakout_strength *= 0.8
        
        return is_breakout, breakout_strength
            
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Detect patterns in the latest window."""
        if historical_df.empty:
            print("Warning: Empty DataFrame provided")
            self.detected_patterns = []
            return

        
        # Find potential poles
        poles = self._find_pole(historical_df)
        
        # For each pole, look for flag pattern
        for pole_start, pole_end, is_bullish in poles:
            pole_return = (historical_df['close'].iloc[pole_end] / historical_df['close'].iloc[pole_start]) - 1
          
            # Look for flag channel after pole
            flag_result = self._find_flag_channel(historical_df, pole_end, is_bullish)
            
            if flag_result:
                flag_end, upper_channel, lower_channel, channel_quality = flag_result
             
                # Check for breakout
                is_breakout, breakout_strength = self._check_breakout(
                    historical_df, flag_end, upper_channel, lower_channel, is_bullish
                )
                
                if is_breakout:
                    pattern = {
                        'pole_start': pole_start,
                        'pole_end': pole_end,
                        'flag_end': flag_end,
                        'is_bullish': is_bullish,
                        'channel_quality': channel_quality,
                        'breakout_strength': breakout_strength
                    }
                    self.detected_patterns.append(pattern)
                
            
        
   )
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict signal based on current price and historical data.
        Returns a float in the range [-1, 1] where:
          * Positive values indicate bullish signals
          * Negative values indicate bearish signals
          * Magnitude indicates signal strength
        """
        # Fit on the provided historical data
        self.fit(historical_df)
        
        # If no patterns detected, return neutral
        if not self.detected_patterns:
            return 0.0
            
        # Focus on the most recent pattern
        latest_pattern = max(self.detected_patterns, key=lambda p: p['flag_end'])
        
        # Calculate signal strength [0-1]
        signal_strength = latest_pattern['channel_quality'] * latest_pattern['breakout_strength']
        
        # Cap at 1.0
        signal_strength = min(signal_strength, 1.0)
        
        # Apply direction
        if latest_pattern['is_bullish']:
            return signal_strength
        else:
            return -signal_strength
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return "Flag Pattern Agent"

