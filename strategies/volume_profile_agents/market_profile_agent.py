"""
Market Profile (TPO) Agent
~~~~~~~~~~~~~~~~~~~~~~~~~
Implements Market Profile analysis using Time Price Opportunity (TPO) charts
to identify value areas and potential reversal points. Market Profile helps
visualize price distribution over time to determine where the market spent
the most time (value area) and extremes that may act as support/resistance.

Logic:
1. Construct TPO profile by tracking price levels visited across time
2. Identify key areas:
   - Point of Control (POC): Price with most TPOs
   - Value Area: Range containing 70% of TPO activity
   - High/Low Volume Nodes: Clusters and gaps in the profile
3. Generate signals when:
   - Price moves outside value area and shows rejection (mean reversion)
   - Price breaks through value area high/low with momentum (trend)
   - Price approaches POC after moving away (magnet effect)
4. Scale signals based on:
   - Profile shape (normal, skewed, bimodal)
   - Recent value area evolution (expanding, contracting, rotating)
   - Time spent building the profile (more significant with more data)

Input: OHLCV DataFrame with DateTimeIndex. Output ∈ [-1, +1].
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional
from enum import Enum


class ProfileTimeframe(Enum):
    SESSION = 1    # Daily session
    MULTI_DAY = 2  # Rolling multi-day
    WEEKLY = 3     # Weekly profile
    CUSTOM = 4     # Custom timeframe


class ProfileShape(Enum):
    NORMAL = 1     # Bell-shaped
    SKEWED_UP = 2  # Skewed to higher prices
    SKEWED_DOWN = 3  # Skewed to lower prices
    BIMODAL = 4    # Two distinct POCs 
    FLAT = 5       # Wide, flat distribution


class MarketProfileAgent:
    def __init__(
        self,
        timeframe: ProfileTimeframe = ProfileTimeframe.SESSION,
        num_days: int = 1,           # Days to include in profile
        num_price_levels: int = 100,  # Number of price levels in profile
        value_area_pct: float = 0.7,  # % of TPOs in value area (typically 70%)
        tpo_interval: int = 30,       # Minutes per TPO
        market_open: time = time(9, 30),
        market_close: time = time(16, 0)
    ):
        self.timeframe = timeframe
        self.days = num_days
        self.num_levels = num_price_levels
        self.va_pct = value_area_pct
        self.tpo_interval = tpo_interval
        self.open_time = market_open
        self.close_time = market_close
        self.profile = None
        self.last_update = None
        
    def _is_update_needed(self, current_time: datetime) -> bool:
        """Check if we need to rebuild the profile"""
        if self.last_update is None:
            return True
            
        # For session profile, rebuild if it's a new day
        if self.timeframe == ProfileTimeframe.SESSION:
            return current_time.date() != self.last_update.date()
            
        # For multi-day profile, rebuild if oldest day drops out
        if self.timeframe == ProfileTimeframe.MULTI_DAY:
            threshold = current_time - timedelta(days=self.days)
            return self.last_update < threshold
            
        # For weekly profile, rebuild if it's a new week
        if self.timeframe == ProfileTimeframe.WEEKLY:
            current_week = current_time.isocalendar()[1]
            last_week = self.last_update.isocalendar()[1]
            return (current_time.year != self.last_update.year or 
                   current_week != last_week)
                   
        # Default to rebuilding every day
        return current_time.date() != self.last_update.date()
        
    def _get_profile_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract data for building the profile based on timeframe"""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for Market Profile")
            
        current_time = df.index[-1]
        
        if self.timeframe == ProfileTimeframe.SESSION:
            # Current day only
            return df[df.index.date == current_time.date()]
            
        elif self.timeframe == ProfileTimeframe.MULTI_DAY:
            # Last N days
            start_date = current_time.date() - timedelta(days=self.days-1)
            return df[df.index.date >= start_date]
            
        elif self.timeframe == ProfileTimeframe.WEEKLY:
            # Current week
            current_week = current_time.isocalendar()[1]
            return df[df.index.isocalendar().week == current_week]
            
        # Default to last N days
        start_date = current_time.date() - timedelta(days=self.days-1)
        return df[df.index.date >= start_date]
        
    def _build_profile(self, df: pd.DataFrame) -> Dict:
        """Construct TPO profile from price data"""
        if len(df) < 2:
            return None
            
        # Get price range
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        # Ensure minimum range for very tight consolidation
        if price_max - price_min < price_min * 0.001:
            avg_price = (price_min + price_max) / 2
            price_min = avg_price * 0.999
            price_max = avg_price * 1.001
            
        # Create price bins
        price_bins = np.linspace(price_min, price_max, self.num_levels + 1)
        bin_width = (price_max - price_min) / self.num_levels
        bin_centers = price_bins[:-1] + bin_width / 2
        
        # Initialize TPO counts
        tpo_counts = np.zeros(self.num_levels, dtype=int)
        
        # Group bars into TPO intervals
        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        df_copy = df.copy()
        df_copy.loc[:, 'tpo_group'] = df_copy.index.to_series().dt.floor(f'{self.tpo_interval}min')
        tpo_groups = df_copy.groupby('tpo_group')
        
        # For each TPO interval, count price levels visited
        for _, group in tpo_groups:
            # Get price range for this TPO
            tpo_high = group['high'].max()
            tpo_low = group['low'].min()
            
            # Find bins this TPO visited
            low_bin = max(0, int((tpo_low - price_min) / bin_width))
            high_bin = min(self.num_levels - 1, int((tpo_high - price_min) / bin_width))
            
            # Add TPO to each bin
            for i in range(low_bin, high_bin + 1):
                tpo_counts[i] += 1
                
        # Calculate total TPOs
        total_tpos = tpo_counts.sum()
        
        # Find point of control (price level with most TPOs)
        poc_idx = np.argmax(tpo_counts)
        poc_price = bin_centers[poc_idx]
        
        # Calculate value area (70% of TPOs)
        target_tpos = int(total_tpos * self.va_pct)
        current_tpos = tpo_counts[poc_idx]
        va_indices = [poc_idx]
        
        # Expand value area outward from POC
        upper_idx = poc_idx
        lower_idx = poc_idx
        
        while current_tpos < target_tpos and (upper_idx < len(tpo_counts) - 1 or lower_idx > 0):
            # Check upper and lower adjacent levels
            upper_tpos = tpo_counts[upper_idx + 1] if upper_idx < len(tpo_counts) - 1 else 0
            lower_tpos = tpo_counts[lower_idx - 1] if lower_idx > 0 else 0
            
            # Add level with more TPOs first
            if upper_tpos >= lower_tpos and upper_idx < len(tpo_counts) - 1:
                upper_idx += 1
                va_indices.append(upper_idx)
                current_tpos += upper_tpos
            elif lower_idx > 0:
                lower_idx -= 1
                va_indices.append(lower_idx)
                current_tpos += lower_tpos
                
        # Calculate value area high and low
        va_high = bin_centers[max(va_indices)]
        va_low = bin_centers[min(va_indices)]
        
        # Determine profile shape
        profile_shape = self._determine_profile_shape(tpo_counts, poc_idx)
        
        # Return profile data
        profile = {
            'price_levels': bin_centers,
            'tpo_counts': tpo_counts,
            'poc': poc_price,
            'va_high': va_high,
            'va_low': va_low,
            'shape': profile_shape,
            'total_tpos': total_tpos
        }
        
        return profile
        
    def _determine_profile_shape(self, tpo_counts: np.ndarray, poc_idx: int) -> ProfileShape:
        """Analyze TPO distribution to determine profile shape"""
        # Calculate first and second moments
        total = tpo_counts.sum()
        if total == 0:
            return ProfileShape.FLAT
            
        # Calculate mean (1st moment)
        weighted_sum = 0
        for i, count in enumerate(tpo_counts):
            weighted_sum += i * count
        mean_idx = weighted_sum / total
        
        # Calculate standard deviation (2nd moment)
        variance = 0
        for i, count in enumerate(tpo_counts):
            variance += ((i - mean_idx) ** 2) * count
        variance /= total
        std_dev = np.sqrt(variance)
        
        # Look for secondary POC
        sorted_idxs = np.argsort(tpo_counts)[::-1]  # Descending
        max_idx = sorted_idxs[0]
        if len(sorted_idxs) > 1:
            second_max_idx = sorted_idxs[1]
            # Check if secondary POC is significant
            if (tpo_counts[second_max_idx] > 0.7 * tpo_counts[max_idx] and 
                abs(second_max_idx - max_idx) > std_dev):
                return ProfileShape.BIMODAL
                
        # Check for skew
        skew = 0
        for i, count in enumerate(tpo_counts):
            skew += ((i - mean_idx) ** 3) * count
        skew = skew / (total * (std_dev ** 3)) if std_dev > 0 else 0
        
        # Determine shape based on skew
        if abs(skew) < 0.5:
            return ProfileShape.NORMAL
        elif skew > 0.5:
            return ProfileShape.SKEWED_UP
        elif skew < -0.5:
            return ProfileShape.SKEWED_DOWN
        
        # Check if profile is flat (low standard deviation)
        if std_dev < 0.1 * len(tpo_counts):
            return ProfileShape.FLAT
            
        return ProfileShape.NORMAL
        
    def _find_low_volume_nodes(self, tpo_counts: np.ndarray, price_levels: np.ndarray) -> List[float]:
        """Find significant low volume nodes (LVNs) in the profile"""
        if len(tpo_counts) < 5:
            return []
            
        # Calculate moving average of TPO counts
        window = 3
        smooth_counts = np.convolve(tpo_counts, np.ones(window)/window, mode='valid')
        
        # Find local minima
        lvn_indices = []
        for i in range(1, len(smooth_counts) - 1):
            if smooth_counts[i] < smooth_counts[i-1] and smooth_counts[i] < smooth_counts[i+1]:
                # Local minimum
                lvn_indices.append(i + window//2)  # Adjust for convolution offset
                
        # Filter significant LVNs (low relative to neighbors)
        significant_lvns = []
        for idx in lvn_indices:
            if idx > 0 and idx < len(tpo_counts) - 1:
                # Check if significantly lower than neighbors
                neighbors_avg = (tpo_counts[idx-1] + tpo_counts[idx+1]) / 2
                if tpo_counts[idx] < 0.5 * neighbors_avg:
                    significant_lvns.append(price_levels[idx])
                    
        return significant_lvns
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Reset internal state - no actual training needed"""
        self.profile = None
        self.last_update = None
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """Generate signal based on market profile analysis"""
        if len(historical_df) < 10:
            return 0.0  # Not enough data
            
        if not isinstance(historical_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for Market Profile")
            
        # Get current time
        current_time = historical_df.index[-1]
        
        # Check if profile needs updating
        if self.profile is None or self._is_update_needed(current_time):
            # Get data for profile
            profile_df = self._get_profile_data(historical_df)
            
            # Build new profile
            self.profile = self._build_profile(profile_df)
            self.last_update = current_time
            
        # If we still don't have a valid profile, return neutral
        if self.profile is None:
            return 0.0
            
        # Get current price
        price = float(current_price)
        
        # Get profile parameters
        poc = self.profile['poc']
        va_high = self.profile['va_high']
        va_low = self.profile['va_low']
        profile_shape = self.profile['shape']
        
        # Identify low volume nodes
        lvns = self._find_low_volume_nodes(
            self.profile['tpo_counts'], 
            self.profile['price_levels']
        )
        
        # Initialize score
        score = 0.0
        
        # Calculate relative position to value area
        va_range = va_high - va_low
        if va_range == 0:
            va_range = poc * 0.01  # Prevent division by zero
            
        # Relative position within value area (-1 to +1)
        if price >= va_low and price <= va_high:
            va_position = 2 * (price - va_low) / va_range - 1  # -1 at VA low, +1 at VA high
        elif price < va_low:
            # Below value area
            va_position = -1 - (va_low - price) / va_range  # Less than -1
        else:
            # Above value area
            va_position = 1 + (price - va_high) / va_range  # Greater than +1
        
        # 1. POC Magnet Effect - tendency for price to be drawn to POC
        poc_distance = abs(price - poc) / va_range
        if poc_distance < 0.5:
            # Close to POC, potential magnet effect
            poc_score = 0.3 * np.sign(poc - price)  # Pull toward POC
            score += poc_score * (1 - poc_distance * 2)  # Stronger when closer
            
        # 2. Value Area Signals
        if abs(va_position) > 1.0:
            # Outside value area - potential mean reversion
            # Calculate strength - stronger the further outside VA
            reversion_strength = min(1.0, (abs(va_position) - 1.0) * 2)
            
            # Mean reversion signal - pull back to value area
            reversion_score = -np.sign(va_position) * reversion_strength * 0.7
            
            # Adjust based on profile shape
            if profile_shape == ProfileShape.SKEWED_UP and va_position > 1:
                # Less likely to revert from above in up-skewed profile
                reversion_score *= 0.5
            elif profile_shape == ProfileShape.SKEWED_DOWN and va_position < -1:
                # Less likely to revert from below in down-skewed profile
                reversion_score *= 0.5
                
            score += reversion_score
            
        # 3. Low Volume Node signals
        if lvns:
            # Find closest LVN
            closest_lvn = min(lvns, key=lambda x: abs(x - price))
            lvn_distance = abs(price - closest_lvn) / va_range
            
            if lvn_distance < 0.2:
                # Near a low volume node - potential acceleration through
                lvn_score = 0.4 * np.sign(price - closest_lvn)  # Direction to move away from LVN
                score += lvn_score * (1 - lvn_distance * 5)  # Stronger when closer
                
        # 4. Profile shape influence
        if profile_shape == ProfileShape.BIMODAL:
            # Bimodal profile often leads to larger moves - amplify signal
            score *= 1.2
            
        elif profile_shape == ProfileShape.FLAT:
            # Flat profile suggests indecision - reduce signal
            score *= 0.7
            
        # 5. Previous bar momentum confirmation
        if len(historical_df) > 1:
            prev_close = historical_df['close'].iloc[-2]
            curr_close = historical_df['close'].iloc[-1]
            
            # If recent momentum matches signal direction, amplify
            if (score > 0 and curr_close > prev_close) or (score < 0 and curr_close < prev_close):
                score *= 1.15
                
        return float(np.clip(score, -1.0, 1.0))
        
    def __str__(self) -> str:
        return "Market Profile (TPO) Agent" 