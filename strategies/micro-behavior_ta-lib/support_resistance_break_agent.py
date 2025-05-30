"""
Support/Resistance Break Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Identifies key price levels (support and resistance) and detects breakouts
from these levels. Uses fractal-based analysis to find significant swing 
highs and lows, then scores price action relative to these levels.

Logic:
1. Identify swing high/low points using local minima/maxima detection
2. Cluster nearby levels to find significant support/resistance zones
3. Monitor price action relative to identified levels
4. Generate signals when price breaks and holds above resistance (bullish)
   or below support (bearish)
5. Scale signal strength based on:
   - Level significance (number of touches, longevity)
   - Breakout momentum (volume and price velocity)
   - Confirmation bars after break

Input: OHLCV DataFrame. Output ∈ [-1, +1].
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from collections import defaultdict

class Support_Resistance_Break_Agent:
    def __init__(
        self, 
        window: int = 5,
        level_threshold: float = 0.0008,  # Lowered to 0.08% for more sensitivity
        lookback_periods: int = 45,  # Further reduced for minute data
        min_touchpoints: int = 2,
        volume_threshold: float = 1.1,  # Lowered volume confirmation threshold
        momentum_window: int = 5,  # Window for momentum calculation
        noise_reduction: float = 0.02  # Base noise level for randomization
    ):
        self.window = window
        self.threshold = level_threshold
        self.lookback = lookback_periods
        self.min_touches = min_touchpoints
        self.volume_threshold = volume_threshold
        self.momentum_window = momentum_window
        self.noise_reduction = noise_reduction
        self.levels = None  # Will store identified levels
        
    def _find_levels(self, data: pd.DataFrame) -> list[tuple[float, int, str]]:
        """Find support and resistance levels
        Returns list of (price_level, strength, type)"""
        if len(data) < self.window * 3:
            return []
            
        # Create copy to avoid warnings
        df = data.copy()
        
        # Calculate volatility for adaptive thresholds with noise reduction
        returns = df['close'].pct_change()
        volatility = returns.rolling(5).std().mean()  # Use rolling std to reduce noise
        adaptive_threshold = max(self.threshold, volatility * 1.5)  # More sensitive threshold
        
        # Add controlled randomness to threshold
        adaptive_threshold *= (1 + np.random.normal(0, self.noise_reduction))
        
        # Use shorter window for minute data extrema detection
        base_window = max(2, min(self.window, int(len(df) * 0.015)))  # More sensitive window
        window = int(base_window * (1 + np.random.normal(0, 0.15)))  # Add +/- 15% variation
        window = max(2, min(self.window, window))
        
        # Find local minima and maxima with adaptive order
        order = max(2, min(window, int(volatility * 100)))  # Adapt to volatility
        local_max_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]
        local_min_idx = argrelextrema(df['low'].values, np.less, order=order)[0]
        
        # Extract price values at extrema with volume weighting
        highs = []
        for i in local_max_idx:
            # Look for confirmation in surrounding bars
            start_idx = max(0, i - window)
            end_idx = min(len(df), i + window + 1)
            if df['high'].iloc[i] >= df['high'].iloc[start_idx:end_idx].max():
                # Calculate volume significance with noise reduction
                period_vol = df['volume'].iloc[start_idx:end_idx]
                avg_vol = period_vol.rolling(3).mean().iloc[-1] if len(period_vol) >= 3 else period_vol.mean()
                vol_ratio = df['volume'].iloc[i] / (avg_vol + 1e-6)  # Avoid division by zero
                vol_weight = min(2.0, max(0.5, vol_ratio))  # Cap volume weight
                
                # Add controlled randomness to volume weight
                vol_weight *= (1 + np.random.normal(0, self.noise_reduction))
                vol_weight = min(2.0, max(0.5, vol_weight))
                
                highs.append((df['high'].iloc[i], i, 'resistance', vol_weight))
                
        lows = []
        for i in local_min_idx:
            # Look for confirmation in surrounding bars
            start_idx = max(0, i - window)
            end_idx = min(len(df), i + window + 1)
            if df['low'].iloc[i] <= df['low'].iloc[start_idx:end_idx].min():
                # Calculate volume significance with noise reduction
                period_vol = df['volume'].iloc[start_idx:end_idx]
                avg_vol = period_vol.rolling(3).mean().iloc[-1] if len(period_vol) >= 3 else period_vol.mean()
                vol_ratio = df['volume'].iloc[i] / (avg_vol + 1e-6)  # Avoid division by zero
                vol_weight = min(2.0, max(0.5, vol_ratio))  # Cap volume weight
                
                # Add controlled randomness to volume weight
                vol_weight *= (1 + np.random.normal(0, self.noise_reduction))
                vol_weight = min(2.0, max(0.5, vol_weight))
                
                lows.append((df['low'].iloc[i], i, 'support', vol_weight))
        
        # Combine all potential levels
        all_levels = highs + lows
        if not all_levels:
            return []
        
        # Cluster nearby levels with improved logic
        clusters = defaultdict(list)
        price_range = df['high'].max() - df['low'].min()
        base_cluster_threshold = max(adaptive_threshold, price_range * 0.008)  # Lowered to 0.8% of range
        
        # Add controlled randomness to cluster threshold
        cluster_threshold = base_cluster_threshold * (1 + np.random.normal(0, self.noise_reduction))
        
        for price, idx, level_type, vol_weight in all_levels:
            # Find closest existing cluster
            min_dist = float('inf')
            best_cluster = None
            
            for cluster_price in clusters.keys():
                dist = abs(price - cluster_price) / (cluster_price + 1e-6)  # Avoid division by zero
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = cluster_price
            
            # Add to closest cluster if within threshold, otherwise create new
            if best_cluster is not None and min_dist < cluster_threshold:
                clusters[best_cluster].append((price, idx, level_type, vol_weight))
            else:
                clusters[price].append((price, idx, level_type, vol_weight))
        
        # Calculate final levels with improved strength calculation
        final_levels = []
        for cluster_points in clusters.values():
            if len(cluster_points) >= self.min_touches:
                # Calculate weighted average price with reduced volume influence
                weights = [w * 0.4 + 0.6 for _, _, _, w in cluster_points]  # More equal weight
                total_weight = sum(weights)
                avg_price = sum(p * w for (p, _, _, _), w in zip(cluster_points, weights)) / (total_weight + 1e-6)
                
                # Determine level type (support/resistance)
                type_count = {'resistance': 0, 'support': 0}
                for _, _, t, w in cluster_points:
                    type_count[t] += 1  # Use count instead of volume weight
                level_type = 'resistance' if type_count['resistance'] > type_count['support'] else 'support'
                
                # Calculate strength based on multiple factors
                num_touches = len(cluster_points)
                touch_score = min(1.0, num_touches / 4)  # Cap at 4 touches
                
                # Recency factor with more weight
                latest_idx = max(idx for _, idx, _, _ in cluster_points)
                recency = 1.0 - (len(df) - latest_idx) / len(df)
                
                # Volume factor (further reduced influence)
                vol_score = min(1.0, sum(w for _, _, _, w in cluster_points) / len(cluster_points))
                
                # Combine scores with adjusted weights
                strength = (touch_score * 0.6 +  # Increased touch importance
                          recency * 0.3 +      # Same recency weight
                          vol_score * 0.1)     # Further reduced volume weight
                
                # Add controlled randomness to strength
                strength *= (1 + np.random.normal(0, self.noise_reduction))
                strength = min(1.0, max(0.0, strength))
                
                final_levels.append((avg_price, strength, level_type))
        
        # Sort by strength and return top levels
        return sorted(final_levels, key=lambda x: x[1], reverse=True)[:6]  # Return one more level
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        # Identify levels from historical data
        if len(historical_df) >= self.lookback:
            data = historical_df.iloc[-self.lookback:]
            self.levels = self._find_levels(data)
        else:
            self.levels = []
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        # Need to recalculate levels if not done in fit
        if self.levels is None or len(self.levels) == 0:
            if len(historical_df) >= self.lookback:
                data = historical_df.iloc[-self.lookback:]
                self.levels = self._find_levels(data)
            else:
                return 0.0  # Not enough data
        
        if not self.levels:
            # Try with relaxed parameters
            data = historical_df.iloc[-min(len(historical_df), self.lookback * 2):]
            original_threshold = self.threshold
            original_touches = self.min_touches
            
            try:
                # Temporarily relax parameters more aggressively
                self.threshold *= 2.5  # More aggressive relaxation
                self.min_touches = 1   # Always try single touch
                self.levels = self._find_levels(data)
            finally:
                # Restore original parameters
                self.threshold = original_threshold
                self.min_touches = original_touches
            
            if not self.levels:
                return 0.0
        
        # Current price and recent behavior
        current_close = float(current_price)
        if len(historical_df) < self.momentum_window:
            return 0.0
        
        # Calculate momentum with shorter lookback and noise reduction
        returns = historical_df['close'].pct_change()
        momentum = returns.rolling(self.momentum_window).mean().iloc[-1]
        
        # Calculate volatility for adaptive thresholds with noise reduction
        volatility = returns.rolling(self.momentum_window).std().mean()
        
        # Find nearest levels and calculate breakout scores
        price_scores = []
        for level_price, level_strength, level_type in self.levels:
            # Calculate relative distance
            dist = (current_close - level_price) / (level_price + 1e-6)  # Avoid division by zero
            
            # Determine if price has crossed level
            if (level_type == 'resistance' and current_close > level_price) or \
               (level_type == 'support' and current_close < level_price):
                
                # Calculate breakout score with adaptive scaling
                dist_score = min(1.0, abs(dist) / (volatility * 5))  # More sensitive to distance
                
                # Volume confirmation with noise reduction
                vol_confirmed = False
                if 'volume' in historical_df.columns and len(historical_df) >= self.momentum_window:
                    recent_vol = historical_df['volume'].rolling(2).mean().iloc[-1]
                    base_vol = historical_df['volume'].rolling(self.momentum_window).mean().iloc[-self.momentum_window:-2].mean()
                    if base_vol > 0:
                        vol_ratio = recent_vol / base_vol
                        vol_confirmed = vol_ratio > self.volume_threshold
                
                # Calculate final score with more granular scaling
                base_score = dist_score * level_strength
                
                # Adjust score based on various factors
                if vol_confirmed:
                    base_score *= 1.2
                
                if abs(momentum) > volatility:
                    if np.sign(momentum) == np.sign(dist):
                        base_score *= 1.2
                    else:
                        base_score *= 0.7
                
                # Add controlled randomness to final score
                base_score *= (1 + np.random.normal(0, self.noise_reduction))
                base_score = min(1.0, max(0.0, base_score))
                
                # Add to scores with direction
                signal = 1.0 if level_type == 'resistance' else -1.0
                price_scores.append(base_score * signal)
        
        if price_scores:
            # Combine scores with emphasis on strongest signal
            final_score = max(price_scores, key=abs)
            
            # Add smaller influence from other signals
            other_scores = [s for s in price_scores if s != final_score]
            if other_scores:
                avg_other = sum(other_scores) / len(other_scores)
                final_score = final_score * 0.8 + avg_other * 0.2
            
            # Add final noise layer
            final_score *= (1 + np.random.normal(0, self.noise_reduction))
            
            return float(np.clip(final_score, -1.0, 1.0))
            
        return 0.0 