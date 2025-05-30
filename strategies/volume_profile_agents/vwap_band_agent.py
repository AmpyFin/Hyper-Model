"""
VWAP Band Agent
~~~~~~~~~~~~~~
Extends VWAP (Volume Weighted Average Price) analysis with standard deviation
bands to identify when price is potentially overextended and likely to revert
to the mean. VWAP serves as the center line, with upper and lower bands at
1, 2, and 3 standard deviations of price from VWAP.

Logic:
1. Calculate VWAP from the start of the specified timeframe (session, week, month)
2. Compute standard deviation bands around VWAP
3. Generate signals when:
   - Price moves outside outer bands (+/-3 SD) → mean reversion signal
   - Price tests and bounces from a key band → trend continuation signal
   - Price breaks through VWAP from above/below → trend change signal
4. Scale signals based on:
   - Distance from VWAP (z-score)
   - Recent volatility
   - VWAP slope
   - Volume profile at price level

Input: OHLCV DataFrame with DateTimeIndex. Output ∈ [-1, +1].
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Tuple, Optional


class VWAPTimeframe(Enum):
    SESSION = 1    # Intraday (from market open)
    DAILY = 2      # From daily open
    WEEKLY = 3     # From weekly open
    MONTHLY = 4    # From monthly open


class VWAPBandAgent:
    def __init__(
        self,
        timeframe: VWAPTimeframe = VWAPTimeframe.DAILY,
        deviation_levels: list[float] = [1.0, 2.0, 3.0],  # Standard deviation levels for bands
        market_open: time = time(9, 30),
        market_close: time = time(16, 0),
        session_cutoff: time = time(11, 0),  # Time after which VWAP becomes more significant
        band_memory: int = 20  # How many bars to remember band touches/breaks
    ):
        self.timeframe = timeframe
        self.deviation_levels = sorted(deviation_levels)
        self.open_time = market_open
        self.close_time = market_close
        self.cutoff = session_cutoff
        self.memory = band_memory
        self.vwap_data = None
        self.last_reset = None
        
    def _reset_needed(self, timestamp: datetime) -> bool:
        """Check if VWAP calculation needs to be reset based on timeframe"""
        if self.last_reset is None:
            return True
            
        current_date = timestamp.date()
        last_date = self.last_reset.date()
        
        if self.timeframe == VWAPTimeframe.SESSION or self.timeframe == VWAPTimeframe.DAILY:
            # Reset if new day
            return current_date > last_date
            
        elif self.timeframe == VWAPTimeframe.WEEKLY:
            # Reset if new week
            last_week = self.last_reset.isocalendar()[1]
            current_week = timestamp.isocalendar()[1]
            return (current_date.year > last_date.year or 
                   (current_date.year == last_date.year and current_week > last_week))
                   
        elif self.timeframe == VWAPTimeframe.MONTHLY:
            # Reset if new month
            return (current_date.year > last_date.year or 
                   (current_date.year == last_date.year and current_date.month > last_date.month))
                   
        return False
        
    def _get_session_start(self, df: pd.DataFrame, current_timestamp: datetime) -> int:
        """Get index for start of relevant timeframe"""
        current_date = current_timestamp.date()
        
        if self.timeframe == VWAPTimeframe.SESSION or self.timeframe == VWAPTimeframe.DAILY:
            # Get start of current day session
            session_start = df[df.index.date == current_date].index[0]
            return df.index.get_loc(session_start)
            
        elif self.timeframe == VWAPTimeframe.WEEKLY:
            # Get start of current week
            current_week = current_timestamp.isocalendar()[1]
            week_start = None
            for idx, ts in enumerate(df.index):
                if ts.isocalendar()[1] == current_week and (week_start is None or ts < week_start):
                    week_start = ts
            if week_start:
                return df.index.get_loc(week_start)
            else:
                # Fallback to first available bar
                return 0
                
        elif self.timeframe == VWAPTimeframe.MONTHLY:
            # Get start of current month
            month_start = None
            for idx, ts in enumerate(df.index):
                if ts.month == current_timestamp.month and ts.year == current_timestamp.year:
                    if month_start is None or ts < month_start:
                        month_start = ts
            if month_start:
                return df.index.get_loc(month_start)
            else:
                # Fallback to first available bar
                return 0
        
        # Default to beginning of available data
        return 0
        
    def _calculate_vwap(self, df: pd.DataFrame, start_idx: int) -> Tuple[float, float, list[Tuple[float, float]]]:
        """Calculate VWAP and standard deviation bands"""
        if start_idx >= len(df):
            return 0.0, 0.0, []
            
        # Get data from start to current
        subset = df.iloc[start_idx:].copy()
        
        # Typical price for each bar
        subset['tp'] = (subset['high'] + subset['low'] + subset['close']) / 3.0
        
        # Running sum of (typical price * volume) and volume
        subset['tp_vol'] = subset['tp'] * subset['volume']
        cumulative_tp_vol = subset['tp_vol'].cumsum()
        cumulative_vol = subset['volume'].cumsum()
        
        # Calculate VWAP
        vwap = cumulative_tp_vol / cumulative_vol
        
        # Calculate price deviation from VWAP
        subset['dev'] = subset['tp'] - vwap
        subset['dev2'] = subset['dev'] ** 2
        subset['dev2_vol'] = subset['dev2'] * subset['volume']
        
        # Standard deviation of price from VWAP, weighted by volume
        cumulative_dev2_vol = subset['dev2_vol'].cumsum()
        std_dev = np.sqrt(cumulative_dev2_vol / cumulative_vol)
        
        # Get current values
        current_vwap = vwap.iloc[-1]
        current_std = std_dev.iloc[-1]
        
        # Calculate bands
        bands = []
        for multiplier in self.deviation_levels:
            upper = current_vwap + (current_std * multiplier)
            lower = current_vwap - (current_std * multiplier)
            bands.append((upper, lower))
            
        return current_vwap, current_std, bands
        
    def _calculate_z_score(self, price: float, vwap: float, std_dev: float) -> float:
        """Calculate z-score of price relative to VWAP"""
        if std_dev == 0:
            return 0.0
        return (price - vwap) / std_dev
        
    def _check_band_break(self, df: pd.DataFrame, current_idx: int, bands: list[Tuple[float, float]]) -> Tuple[list[bool], list[bool]]:
        """Check for crosses through VWAP bands in recent history"""
        if current_idx < self.memory:
            lookback = current_idx
        else:
            lookback = self.memory
            
        recent_bars = df.iloc[current_idx-lookback:current_idx+1]
        
        breaks_up = []  # Breaks from below to above
        breaks_down = []  # Breaks from above to below
        
        for upper, lower in bands:
            # Check if we've broken through upper band recently
            break_up = False
            for i in range(1, len(recent_bars)):
                prev_close = recent_bars.iloc[i-1]['close']
                curr_close = recent_bars.iloc[i]['close']
                if prev_close < upper and curr_close > upper:
                    break_up = True
                    break
            breaks_up.append(break_up)
            
            # Check if we've broken through lower band recently
            break_down = False
            for i in range(1, len(recent_bars)):
                prev_close = recent_bars.iloc[i-1]['close']
                curr_close = recent_bars.iloc[i]['close']
                if prev_close > lower and curr_close < lower:
                    break_down = True
                    break
            breaks_down.append(break_down)
            
        return breaks_up, breaks_down
    
    def _check_band_touch(self, price: float, bands: list[Tuple[float, float]], window: pd.DataFrame) -> Tuple[Optional[int], str]:
        """Check if current price is touching a band and returning"""
        if len(window) < 3:
            return None, ""
            
        # Get recent price action
        high = window['high'].iloc[-1]
        low = window['low'].iloc[-1]
        close = window['close'].iloc[-1]
        open_price = window['open'].iloc[-1]
        
        # Check touches for each band
        for i, (upper, lower) in enumerate(bands):
            # Check for upper band touch and rejection
            if high >= upper and close < upper:
                # Price touched upper band but closed below
                # Stronger signal if close is lower than open (bearish bar)
                if close < open_price:
                    return i, "upper_reject"
                    
            # Check for lower band touch and bounce
            if low <= lower and close > lower:
                # Price touched lower band but closed above
                # Stronger signal if close is higher than open (bullish bar)
                if close > open_price:
                    return i, "lower_bounce"
                    
        return None, ""
        
    def _calculate_vwap_slope(self, df: pd.DataFrame, start_idx: int, bars: int = 10) -> float:
        """Calculate normalized slope of VWAP over recent bars"""
        if start_idx + bars >= len(df):
            return 0.0
            
        subset = df.iloc[start_idx:start_idx + bars + 1].copy()
        
        # Calculate VWAP for each bar
        subset['tp'] = (subset['high'] + subset['low'] + subset['close']) / 3.0
        subset['tp_vol'] = subset['tp'] * subset['volume']
        
        # Running sum of (typical price * volume) and volume
        cum_tp_vol = subset['tp_vol'].cumsum()
        cum_vol = subset['volume'].cumsum()
        
        # VWAP series
        vwap_series = cum_tp_vol / cum_vol
        
        # Get first and last VWAP value
        start_vwap = vwap_series.iloc[0]
        end_vwap = vwap_series.iloc[-1]
        
        # Calculate slope as percentage change
        if start_vwap == 0:
            return 0.0
            
        slope = (end_vwap - start_vwap) / start_vwap
        return slope
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Reset internal state - no actual training needed"""
        self.vwap_data = None
        self.last_reset = None
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """Generate signal based on price relative to VWAP bands"""
        if len(historical_df) < 20:
            return 0.0  # Not enough data
            
        if not isinstance(historical_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for VWAP calculation")
            
        # Get current timestamp
        current_timestamp = historical_df.index[-1]
        
        # Check if we need to reset VWAP calculation
        if self._reset_needed(current_timestamp):
            self.vwap_data = None
            self.last_reset = current_timestamp
            
        # Get starting index for VWAP calculation
        start_idx = self._get_session_start(historical_df, current_timestamp)
        
        # Calculate VWAP and bands
        vwap, std_dev, bands = self._calculate_vwap(historical_df, start_idx)
        
        if vwap == 0.0:
            return 0.0  # Could not calculate VWAP
            
        # Store calculated values
        self.vwap_data = {
            'vwap': vwap,
            'std_dev': std_dev,
            'bands': bands
        }
        
        # Current price
        price = float(current_price)
        
        # Calculate z-score
        z_score = self._calculate_z_score(price, vwap, std_dev)
        
        # Get current index in dataframe
        current_idx = len(historical_df) - 1
        
        # Check for recent band breaks
        breaks_up, breaks_down = self._check_band_break(historical_df, current_idx, bands)
        
        # Check for band touches
        recent_window = historical_df.iloc[-5:]
        band_touch, touch_type = self._check_band_touch(price, bands, recent_window)
        
        # Calculate VWAP slope (trend direction and strength)
        vwap_slope = self._calculate_vwap_slope(historical_df, start_idx)
        
        # Initialize score
        score = 0.0
        
        # Time-based significance factor (VWAP becomes more significant later in session)
        significance = 1.0
        if self.timeframe == VWAPTimeframe.SESSION or self.timeframe == VWAPTimeframe.DAILY:
            current_time = current_timestamp.time()
            if current_time < self.cutoff:
                # Early in session, VWAP is less reliable
                elapsed = (current_timestamp - historical_df.iloc[start_idx].name).total_seconds() / 3600.0  # Hours
                significance = min(1.0, elapsed / 2.0)  # Gradually increase significance over first 2 hours
        
        # 1. Mean reversion signals for extreme z-scores
        if abs(z_score) > self.deviation_levels[-1]:
            # Beyond outer band - strong mean reversion signal
            reversion_score = -np.tanh(z_score / 2.0)  # Negative for high z-score, positive for low
            score += reversion_score * 0.7 * significance
            
        # 2. Band touch/rejection signals
        if band_touch is not None:
            band_level = self.deviation_levels[band_touch]
            if touch_type == "upper_reject":
                # Bearish signal on upper band rejection
                touch_score = -0.5 * band_level / self.deviation_levels[-1]
                score += touch_score * significance
            elif touch_type == "lower_bounce":
                # Bullish signal on lower band bounce
                touch_score = 0.5 * band_level / self.deviation_levels[-1]
                score += touch_score * significance
                
        # 3. Band break signals
        for i, (break_up, break_down) in enumerate(zip(breaks_up, breaks_down)):
            if break_up or break_down:
                # More significant breaks at higher bands
                band_level = self.deviation_levels[i]
                weight = band_level / self.deviation_levels[-1]
                
                if break_up:
                    # Bullish break above band
                    break_score = 0.4 * weight
                    score += break_score * significance
                    
                if break_down:
                    # Bearish break below band
                    break_score = -0.4 * weight
                    score += break_score * significance
                    
        # 4. VWAP cross signals
        if len(historical_df) > start_idx + 1:
            prev_close = historical_df.iloc[-2]['close']
            curr_close = historical_df.iloc[-1]['close']
            
            # Check for cross above/below VWAP
            if prev_close < vwap and curr_close > vwap:
                # Bullish cross above VWAP
                score += 0.3 * significance
                
            elif prev_close > vwap and curr_close < vwap:
                # Bearish cross below VWAP
                score -= 0.3 * significance
                
        # 5. Trend confirmation/contradiction
        # If VWAP slope and z-score have the same sign, trend is strong
        # If they have opposite sign, potential reversal
        if abs(vwap_slope) > 0.001:  # Non-flat VWAP
            if np.sign(vwap_slope) == np.sign(z_score):
                # Trend confirmation - amplify score
                score *= 1.2
            else:
                # Possible trend shift - reduce score
                score *= 0.8
                
        # Final adjustment based on z-score (distance from VWAP)
        # Weaker signals when price is near VWAP
        if abs(z_score) < 0.5:
            score *= abs(z_score) * 2  # Scale down signals near VWAP
            
        return float(np.clip(score, -1.0, 1.0))
        
    def __str__(self) -> str:
        return "VWAP Band Agent" 