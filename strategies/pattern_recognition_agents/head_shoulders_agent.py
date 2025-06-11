"""
Head and Shoulders Pattern Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Detects Head and Shoulders chart patterns (both regular/bearish and inverse/bullish).
These are reversal patterns consisting of three peaks, with the middle peak (head)
being the highest and the two outer peaks (shoulders) being lower and at roughly the 
same level, connected by a neckline.

Logic:
1. Identify local highs/lows using peak detection algorithm
2. Scan for potential pattern formations:
   - Regular H&S: Three peaks with the middle one highest, on a common neckline
   - Inverse H&S: Three troughs with the middle one lowest, on a common neckline
3. Confirm pattern using strict criteria:
   - Proper sequence and relative heights of peaks/troughs
   - Clear neckline with at least two touch points
   - Volume typically decreases toward the head and increases on breakout
4. Generate signals on neckline breakout:
   - Regular H&S: Bearish signal on neckline break down
   - Inverse H&S: Bullish signal on neckline break up
5. Scale signal strength based on:
   - Pattern size (% of price)
   - Volume confirmation
   - Clean formation (symmetry, clear neckline)
   - Following vs. against prevailing trend

Input: OHLCV DataFrame. Output ∈ [-1, +1].
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.signal import find_peaks


class HeadShouldersAgent:
    def __init__(
        self,
        window_size: int = 120,          # Increased window for better context
        peak_distance: int = 5,          # Minimum distance between peaks
        peak_prominence: float = 0.003,   # Reduced from 0.01 for intraday
        shoulder_tolerance: float = 0.1,  # Reduced from 0.2 for tighter formations
        volume_confirmation: bool = True, # Whether to use volume for confirmation
        breakout_threshold: float = 0.002 # Reduced from 0.01 for intraday
    ):
        self.window = window_size
        self.peak_distance = peak_distance
        self.peak_prominence = peak_prominence
        self.shoulder_tolerance = shoulder_tolerance
        self.use_volume = volume_confirmation
        self.breakout_threshold = breakout_threshold
        self.detected_patterns = []
        
    def _find_peaks_and_troughs(self, prices: np.ndarray) -> Tuple[List[int], List[int]]:
        """Find significant peaks and troughs in the price series"""
        # Calculate ATR-based prominence for adaptive threshold
        high_low = np.max(prices) - np.min(prices)
        typical_move = high_low / len(prices)  # Average price move per bar
        prominence = max(self.peak_prominence, typical_move * 0.5)  # At least 50% of typical move
        
        # Find peaks (for regular H&S)
        peaks, _ = find_peaks(
            prices, 
            distance=self.peak_distance, 
            prominence=prominence
        )
        
        # Find troughs (for inverse H&S)
        troughs, _ = find_peaks(
            -prices,  # Invert to find troughs
            distance=self.peak_distance, 
            prominence=prominence
        )
        
        return peaks.tolist(), troughs.tolist()
        
    def _check_head_shoulders(
            self, 
            df: pd.DataFrame, 
            peaks: List[int], 
            inverse: bool = False
        ) -> List[Tuple[int, int, float, float]]:
        """
        Check for head and shoulders patterns in the given peaks
        Returns list of (start_idx, end_idx, neckline, pattern_quality)
        """
        if len(peaks) < 3:
            return []
            
        patterns = []
        
        # Get the relevant price series
        price_series = df['low'] if inverse else df['high']
        
        # Analyze each potential head and shoulders pattern
        for i in range(len(peaks) - 2):
            left_idx, head_idx, right_idx = peaks[i], peaks[i+1], peaks[i+2]
            
            # Skip if indices are too close
            if head_idx - left_idx < self.peak_distance or right_idx - head_idx < self.peak_distance:
                continue
                
            # Get prices
            left_price = price_series.iloc[left_idx]
            head_price = price_series.iloc[head_idx]
            right_price = price_series.iloc[right_idx]
            
            # Check pattern criteria
            if inverse:
                # Inverse H&S: Head should be lower than shoulders
                if not (head_price < left_price and head_price < right_price):
                    continue
                    
                # Shoulders should be at similar levels
                shoulder_diff = abs(left_price - right_price) / left_price
                if shoulder_diff > self.shoulder_tolerance:
                    continue
                    
                # Find neckline connecting the highs between troughs
                left_neck_idx = df['high'].iloc[left_idx:head_idx].idxmax()
                right_neck_idx = df['high'].iloc[head_idx:right_idx].idxmax()
                left_neck = df['high'].loc[left_neck_idx]
                right_neck = df['high'].loc[right_neck_idx]
                
            else:
                # Regular H&S: Head should be higher than shoulders
                if not (head_price > left_price and head_price > right_price):
                    continue
                    
                # Shoulders should be at similar levels
                shoulder_diff = abs(left_price - right_price) / left_price
                if shoulder_diff > self.shoulder_tolerance:
                    continue
                    
                # Find neckline connecting the lows between peaks
                left_neck_idx = df['low'].iloc[left_idx:head_idx].idxmin()
                right_neck_idx = df['low'].iloc[head_idx:right_idx].idxmin()
                left_neck = df['low'].loc[left_neck_idx]
                right_neck = df['low'].loc[right_neck_idx]
            
            # Calculate neckline
            x1 = df.index.get_loc(left_neck_idx)
            x2 = df.index.get_loc(right_neck_idx)
            y1 = left_neck
            y2 = right_neck
            
            # Calculate neckline slope
            neckline_slope = (y2 - y1) / (x2 - x1)
            
            # More flexible slope check for intraday
            if abs(neckline_slope) > 0.002:  # Allow 0.2% slope per bar
                continue
                
            # Calculate average neckline level
            neckline = (y1 + y2) / 2
            
            # Check volume confirmation if enabled
            volume_confirmed = True
            if self.use_volume and 'volume' in df.columns:
                # Compare volume patterns
                left_vol = df['volume'].iloc[left_idx:head_idx].mean()
                head_vol = df['volume'].iloc[head_idx]
                right_vol = df['volume'].iloc[head_idx:right_idx].mean()
                
                # More flexible volume criteria for intraday
                if head_vol > left_vol * 1.5:  # Head volume shouldn't be too high
                    volume_confirmed = False
                
                # Check potential breakout volume
                if right_idx + 1 < len(df):
                    breakout_vol = df['volume'].iloc[right_idx+1]
                    if breakout_vol < right_vol * 0.8:  # Allow some flexibility
                        volume_confirmed = False
            
            # Calculate pattern quality score [0-1]
            symmetry = 1.0 - shoulder_diff
            neckline_clarity = 1.0 - abs(neckline_slope) * 500  # Scale slope to [0-1]
            
            quality = (symmetry * 0.5) + (neckline_clarity * 0.5)
            if self.use_volume:
                quality = quality * 0.8 + (0.2 if volume_confirmed else 0.0)
            
            # Add to patterns
            pattern = (left_idx, right_idx, neckline, quality)
            patterns.append(pattern)
            
        return patterns
        
    def _check_breakout(
            self, 
            df: pd.DataFrame, 
            pattern: Tuple[int, int, float, float],
            inverse: bool = False
        ) -> Tuple[bool, float]:
        """
        Check if there's a breakout of the neckline
        Returns (is_breakout, breakout_strength)
        """
        start_idx, end_idx, neckline, quality = pattern
        
        # No room for breakout confirmation
        if end_idx + 1 >= len(df):
            return False, 0.0
            
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Check for breakout
        if inverse:
            # Bullish breakout: Price breaks above neckline
            breakout = current_price > neckline
            # Calculate breakout strength
            if breakout:
                # Percent above neckline
                strength = (current_price - neckline) / neckline
                # Normalize to typical range
                strength = min(1.0, strength / self.breakout_threshold)
            else:
                strength = 0.0
        else:
            # Bearish breakout: Price breaks below neckline
            breakout = current_price < neckline
            # Calculate breakout strength
            if breakout:
                # Percent below neckline
                strength = (neckline - current_price) / neckline
                # Normalize to typical range
                strength = min(1.0, strength / self.breakout_threshold)
            else:
                strength = 0.0
                
        return breakout, strength
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Analyze historical data to find head and shoulders patterns"""
        # Reset detected patterns
        self.detected_patterns = []
        
        # Need enough data to detect patterns
        if len(historical_df) < self.window:
            return
            
        # Get recent window of data
        df = historical_df.iloc[-self.window:].copy()
        
        # Find peaks and troughs
        peaks, troughs = self._find_peaks_and_troughs(df['close'].values)
        
        # Check for regular head and shoulders (bearish)
        regular_patterns = self._check_head_shoulders(df, peaks, inverse=False)
        
        # Check for inverse head and shoulders (bullish)
        inverse_patterns = self._check_head_shoulders(df, troughs, inverse=True)
        
        # Store detected patterns with type
        for pattern in regular_patterns:
            self.detected_patterns.append(("regular", pattern))
            
        for pattern in inverse_patterns:
            self.detected_patterns.append(("inverse", pattern))
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """Generate signal based on head and shoulders pattern detection"""
        # Refit to detect any new patterns
        self.fit(historical_df)
        
        # No patterns detected
        if not self.detected_patterns:
            return 0.0
            
        # Get strongest confirmed pattern
        best_score = 0.0
        best_pattern = None
        best_type = None
        
        for pattern_type, pattern in self.detected_patterns:
            # Check if the pattern has a breakout
            is_breakout, strength = self._check_breakout(
                historical_df, 
                pattern,
                inverse=(pattern_type == "inverse")
            )
            
            if is_breakout:
                # Calculate score based on pattern quality and breakout strength
                _, _, _, quality = pattern
                score = quality * strength
                
                # Track best pattern
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
                    best_type = pattern_type
        
        # No confirmed breakout patterns
        if best_pattern is None:
            return 0.0
            
        # Generate signal
        if best_type == "regular":
            # Bearish signal
            return -best_score
        else:
            # Bullish signal
            return best_score
            
    def __str__(self) -> str:
        return "Head and Shoulders Pattern Agent" 