"""
Volume Profile Delta Agent
~~~~~~~~~~~~~~~~~~~~~
Analyzes changes in the volume profile distribution over time to identify shifts in
trading activity across price levels and detect emerging support/resistance areas.

Logic:
1. Create volume profiles for different time windows
2. Compare current volume profile with historical volume distributions
3. Identify areas where volume is significantly increasing or decreasing
4. Generate signals based on shifts in volume concentration

Input: OHLCV DataFrame. Output ∈ [-1, +1] where:
* Positive values: Volume shifting to higher price levels (bullish)
* Negative values: Volume shifting to lower price levels (bearish)
* Values near zero: No significant shifts in volume profile
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class VolumeProfileDeltaAgent:
    def __init__(
        self,
        num_price_levels: int = 20,      # Number of price levels for volume profiles
        short_window: int = 20,          # Recent window for volume profile
        long_window: int = 100,          # Historical window for volume profile
        value_area_pct: float = 0.7,     # Percentage of volume for value area
        min_delta_threshold: float = 0.1, # Minimum delta for significance
        signal_smoothing: int = 3        # Periods for signal smoothing
    ):
        self.num_price_levels = num_price_levels
        self.short_window = short_window
        self.long_window = long_window
        self.value_area_pct = value_area_pct
        self.min_delta_threshold = min_delta_threshold
        self.signal_smoothing = signal_smoothing
        self.latest_signal = 0.0
        self.signal_history = []
        self.profile_data = {}
        
    def _calculate_volume_profile(self, df: pd.DataFrame, window: int) -> Dict:
        """Calculate volume profile for a specific window"""
        if len(df) < window:
            return {}
            
        # Get the window of data
        window_data = df.iloc[-window:]
        
        # Find price range
        price_min = window_data['low'].min()
        price_max = window_data['high'].max()
        
        # Create price bins
        price_range = price_max - price_min
        if price_range <= 0:
            return {}
            
        bin_size = price_range / self.num_price_levels
        
        # Initialize bins
        price_bins = [price_min + i * bin_size for i in range(self.num_price_levels + 1)]
        volume_by_price = [0] * self.num_price_levels
        
        # Distribute volume across price bins
        for _, row in window_data.iterrows():
            # Simplification: use OHLC average as representative price
            avg_price = (row['open'] + row['high'] + row['low'] + row['close']) / 4
            
            # Find which bin this price falls into
            bin_idx = min(self.num_price_levels - 1, 
                         max(0, int((avg_price - price_min) / bin_size)))
                         
            # Add volume to that bin
            volume_by_price[bin_idx] += row['volume']
        
        # Calculate total volume and bin centers
        total_volume = sum(volume_by_price)
        bin_centers = [price_min + (i + 0.5) * bin_size for i in range(self.num_price_levels)]
        
        # Calculate volume profile metrics
        result = {
            'price_range': (price_min, price_max),
            'bin_size': bin_size,
            'price_bins': price_bins,
            'bin_centers': bin_centers,
            'volume_by_price': volume_by_price,
            'total_volume': total_volume
        }
        
        # Calculate volume-weighted average price (VWAP)
        if total_volume > 0:
            vwap = sum(price * vol for price, vol in zip(bin_centers, volume_by_price)) / total_volume
            result['vwap'] = vwap
            
        # Calculate Point of Control (POC) - price level with highest volume
        if volume_by_price:
            poc_idx = volume_by_price.index(max(volume_by_price))
            result['poc'] = bin_centers[poc_idx]
            result['poc_volume'] = volume_by_price[poc_idx]
            
        # Calculate Value Area - range containing value_area_pct of volume
        if total_volume > 0:
            # Sort bins by volume (descending)
            sorted_bins = sorted(zip(bin_centers, volume_by_price), 
                                key=lambda x: x[1], reverse=True)
            
            # Find bins that make up value area
            cum_vol = 0
            value_area_bins = []
            
            for bin_center, bin_vol in sorted_bins:
                cum_vol += bin_vol
                value_area_bins.append(bin_center)
                
                if cum_vol >= total_volume * self.value_area_pct:
                    break
                    
            # Get min/max of value area
            if value_area_bins:
                result['value_area_low'] = min(value_area_bins)
                result['value_area_high'] = max(value_area_bins)
                
        # Calculate volume profile distribution (normalized)
        if total_volume > 0:
            result['volume_distribution'] = [vol / total_volume for vol in volume_by_price]
            
        return result
        
    def _compare_volume_profiles(self, short_profile: Dict, long_profile: Dict) -> Dict:
        """Compare short-term and long-term volume profiles to detect shifts"""
        result = {}
        
        # Skip if either profile is empty
        if not short_profile or not long_profile:
            return result
            
        # Get distributions
        short_dist = short_profile.get('volume_distribution', [])
        long_dist = long_profile.get('volume_distribution', [])
        
        # Ensure same length for comparison
        min_len = min(len(short_dist), len(long_dist))
        if min_len == 0:
            return result
            
        short_dist = short_dist[:min_len]
        long_dist = long_dist[:min_len]
        
        # Calculate bin-by-bin deltas
        volume_deltas = [short - long for short, long in zip(short_dist, long_dist)]
        
        # Store deltas
        result['volume_deltas'] = volume_deltas
        
        # Calculate weighted average delta by price level
        bin_centers = short_profile.get('bin_centers', [])[:min_len]
        if bin_centers:
            # Calculate center of price range for normalization
            price_center = (bin_centers[0] + bin_centers[-1]) / 2
            
            # Calculate normalized price levels (centered at 0, range [-1, 1])
            # This converts price levels to a relative scale
            price_range = bin_centers[-1] - bin_centers[0]
            if price_range > 0:
                norm_prices = [(p - price_center) / (price_range / 2) for p in bin_centers]
            else:
                norm_prices = [0] * len(bin_centers)
                
            # Calculate weighted delta (positive = volume shifting to higher prices)
            weighted_delta = sum(delta * price for delta, price in zip(volume_deltas, norm_prices))
            result['weighted_delta'] = weighted_delta
            
            # Filter for significant shifts only (ignore small noise)
            sig_deltas = [delta if abs(delta) > self.min_delta_threshold else 0 for delta in volume_deltas]
            sig_weighted_delta = sum(delta * price for delta, price in zip(sig_deltas, norm_prices))
            result['significant_weighted_delta'] = sig_weighted_delta
            
        # Compare POC shift
        if 'poc' in short_profile and 'poc' in long_profile:
            poc_shift = short_profile['poc'] - long_profile['poc']
            result['poc_shift'] = poc_shift
            
            # Normalize POC shift
            price_range = short_profile['price_range'][1] - short_profile['price_range'][0]
            if price_range > 0:
                result['normalized_poc_shift'] = poc_shift / price_range
                
        # Compare value areas
        if ('value_area_low' in short_profile and 'value_area_high' in short_profile and
            'value_area_low' in long_profile and 'value_area_high' in long_profile):
            # Calculate shifts in value area boundaries
            va_low_shift = short_profile['value_area_low'] - long_profile['value_area_low']
            va_high_shift = short_profile['value_area_high'] - long_profile['value_area_high']
            
            result['va_low_shift'] = va_low_shift
            result['va_high_shift'] = va_high_shift
            
            # Check if value area is expanding or contracting
            short_va_size = short_profile['value_area_high'] - short_profile['value_area_low']
            long_va_size = long_profile['value_area_high'] - long_profile['value_area_low']
            
            result['va_size_change'] = short_va_size - long_va_size
            
        return result
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to detect shifts in volume profile
        """
        # Need enough bars for calculation
        if len(historical_df) < self.long_window:
            self.latest_signal = 0.0
            return
            
        # Calculate volume profiles
        short_profile = self._calculate_volume_profile(historical_df, self.short_window)
        long_profile = self._calculate_volume_profile(historical_df, self.long_window)
        
        # Compare profiles
        self.profile_data = self._compare_volume_profiles(short_profile, long_profile)
        
        # Generate signal
        signal_components = []
        
        # Primary signal component: weighted delta of volume distribution
        if 'significant_weighted_delta' in self.profile_data:
            # Scale to [-1, 1] range (already normalized by price levels)
            weighted_signal = self.profile_data['significant_weighted_delta']
            signal_components.append(weighted_signal)
            
        # Secondary component: POC shift
        if 'normalized_poc_shift' in self.profile_data:
            # Scale to [-1, 1] range
            poc_signal = np.clip(self.profile_data['normalized_poc_shift'] * 5, -1, 1)
            signal_components.append(poc_signal * 0.7)  # Lower weight
            
        # Calculate combined signal
        if signal_components:
            raw_signal = sum(signal_components) / len(signal_components)
            
            # Ensure signal is in [-1, 1] range
            raw_signal = max(-1.0, min(1.0, raw_signal))
        else:
            raw_signal = 0.0
            
        # Apply smoothing
        self.signal_history.append(raw_signal)
        if len(self.signal_history) > self.signal_smoothing:
            self.signal_history.pop(0)
            
        self.latest_signal = sum(self.signal_history) / len(self.signal_history)
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict volume profile shift signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate volume shifting to higher prices
          * Negative values indicate volume shifting to lower prices
          * Values near zero indicate no significant shift
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Volume Profile Delta Agent" 