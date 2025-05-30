"""
Volume Breakout Agent
~~~~~~~~~~~~~~~~~~~~
Detects significant volume spikes accompanied by price movement that can signal
a potential breakout from consolidation or continuation of an existing trend.
Analyzes both absolute and relative volume increases along with price patterns
to identify true breakouts from false signals.

Logic:
1. Monitor volume relative to historical averages (daily, weekly)
2. Detect significant volume spikes (>150% of recent average)
3. Generate signals when:
   - High volume accompanied by price breakout above resistance
   - High volume with price breakdown below support
   - High volume with price closing strongly in the direction of the move
4. Scale signals based on:
   - Volume increase magnitude
   - Price movement strength
   - Prior consolidation/tight range
   - Follow-through confirmation

Input: OHLCV DataFrame. Output ∈ [-1, +1].
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class VolumeBreakoutAgent:
    def __init__(
        self,
        volume_threshold: float = 1.2,  # Reduced from 1.5 to 1.2 (120% increase)
        lookback_short: int = 10,        # Short-term average window
        lookback_medium: int = 30,       # Medium-term average window
        consolidation_threshold: float = 0.3,  # Max normalized range for consolidation
        price_change_threshold: float = 0.5,   # Min price change percentile for breakout
        confirmation_bars: int = 3             # Bars to look for follow-through
    ):
        self.vol_threshold = volume_threshold
        self.lb_short = lookback_short
        self.lb_medium = lookback_medium
        self.consol_threshold = consolidation_threshold
        self.price_threshold = price_change_threshold
        self.confirm_bars = confirmation_bars
        
    def _detect_volume_spike(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Check if current bar has a significant volume spike"""
        if len(df) < self.lb_medium + 1:
            return False, 0.0
            
        # Get current volume and historical averages
        current_vol = df['volume'].iloc[-1]
        short_avg = df['volume'].iloc[-self.lb_short-1:-1].mean()
        medium_avg = df['volume'].iloc[-self.lb_medium-1:-1].mean()
        
        # Calculate relative volume increase
        short_increase = current_vol / short_avg if short_avg > 0 else 0
        medium_increase = current_vol / medium_avg if medium_avg > 0 else 0
        
        # Average the two increases (more weight to short-term)
        vol_increase = (short_increase * 0.7) + (medium_increase * 0.3)
        
        # Modified to return some signal even for smaller volume increases
        is_spike = vol_increase >= self.vol_threshold
        
        # Scale the increase to provide signal even below threshold
        scaled_increase = min(1.0, vol_increase / self.vol_threshold)
        
        return is_spike, scaled_increase
        
    def _check_consolidation(self, df: pd.DataFrame, window: int = 10) -> float:
        """Check if price has been consolidating before the current bar
        Returns consolidation strength [0-1]"""
        if len(df) < window + 1:
            return 0.0
            
        # Get price range over the window before current bar
        hist_window = df.iloc[-window-1:-1]
        price_high = hist_window['high'].max()
        price_low = hist_window['low'].min()
        avg_price = hist_window['close'].mean()
        
        # Calculate normalized price range
        if avg_price == 0:
            return 0.0
            
        norm_range = (price_high - price_low) / avg_price
        
        # Calculate ATR for comparison
        tr_sum = 0.0
        prev_close = hist_window['close'].iloc[0]
        
        for i in range(1, len(hist_window)):
            row = hist_window.iloc[i]
            tr = max(
                row['high'] - row['low'],
                abs(row['high'] - prev_close),
                abs(row['low'] - prev_close)
            )
            tr_sum += tr
            prev_close = row['close']
            
        avg_tr = tr_sum / (len(hist_window) - 1) if len(hist_window) > 1 else 0
        
        # Normalize ATR as percentage of price
        norm_atr = avg_tr / avg_price if avg_price > 0 else 0
        
        # Compare range to ATR * window to see if it's consolidating
        expected_range = norm_atr * window * 0.5  # Expected range if random walk
        
        if norm_range < expected_range * self.consol_threshold:
            # Strong consolidation
            consolidation_strength = 1.0
        elif norm_range < expected_range:
            # Partial consolidation
            consolidation_strength = 1.0 - (norm_range / (expected_range * self.consol_threshold))
        else:
            # Not consolidating
            consolidation_strength = 0.0
            
        return consolidation_strength
        
    def _check_price_breakout(self, df: pd.DataFrame) -> Tuple[bool, bool, float]:
        """Detect if current bar represents a price breakout/breakdown
        Returns (bullish_break, bearish_break, strength)"""
        if len(df) < self.lb_short + 1:
            return False, False, 0.0
            
        # Current bar
        current = df.iloc[-1]
        
        # Previous N bars
        hist_window = df.iloc[-self.lb_short-1:-1]
        
        # Calculate average true range for normalization
        atr = 0.0
        prev_close = hist_window['close'].iloc[0]
        
        for i in range(1, len(hist_window)):
            row = hist_window.iloc[i]
            tr = max(
                row['high'] - row['low'],
                abs(row['high'] - prev_close),
                abs(row['low'] - prev_close)
            )
            atr += tr
            prev_close = row['close']
            
        atr = atr / (len(hist_window) - 1) if len(hist_window) > 1 else 0
        
        # Debug ATR calculation
        print(f"ATR: {atr}")
        
        # Check for breakout above recent high or breakdown below recent low
        recent_high = hist_window['high'].max()
        recent_low = hist_window['low'].min()
        recent_close_avg = hist_window['close'].mean()
        
        # Get recent price action
        last_3_closes = hist_window['close'].iloc[-3:]
        price_momentum = (last_3_closes.iloc[-1] - last_3_closes.iloc[0]) / last_3_closes.iloc[0]
        
        print(f"Recent high: {recent_high}")
        print(f"Recent low: {recent_low}")
        print(f"Current close: {current['close']}")
        print(f"Recent close avg: {recent_close_avg}")
        print(f"Price momentum (3-bar): {price_momentum * 100:.2f}%")
        
        # Calculate price changes
        breakout_amount = current['close'] - recent_high
        breakdown_amount = recent_low - current['close']
        
        # Normalize by ATR and make more sensitive (reduced from 0.5 to 0.25)
        norm_breakout = breakout_amount / (atr * 0.25) if atr > 0 else 0
        norm_breakdown = breakdown_amount / (atr * 0.25) if atr > 0 else 0
        
        print(f"Normalized breakout: {norm_breakout}")
        print(f"Normalized breakdown: {norm_breakdown}")
        
        # Multiple conditions for breakout/breakdown
        bullish_break = (
            norm_breakout > 0 or  # Price above recent high
            (current['close'] > recent_close_avg * 1.002) or  # Reduced to 0.2%
            (price_momentum > 0.001)  # Added momentum condition
        )
        
        bearish_break = (
            norm_breakdown > 0 or  # Price below recent low
            (current['close'] < recent_close_avg * 0.998) or  # Reduced to 0.2%
            (price_momentum < -0.001)  # Added momentum condition
        )
        
        # Calculate strength based on close relative to range
        bar_range = current['high'] - current['low']
        if bar_range == 0:
            bar_strength = 0.5
        else:
            # Close position within bar (0-1, higher = stronger close)
            bar_strength = (current['close'] - current['low']) / bar_range
            
        # Calculate final strength
        if bullish_break and not bearish_break:
            # Base strength on the strongest signal
            strength = max(
                abs(norm_breakout),
                abs((current['close'] / recent_close_avg) - 1) * 10,
                abs(price_momentum) * 10,
                0.1  # Minimum strength
            ) * (0.5 + bar_strength / 2)
        elif bearish_break and not bullish_break:
            # Base strength on the strongest signal
            strength = max(
                abs(norm_breakdown),
                abs((current['close'] / recent_close_avg) - 1) * 10,
                abs(price_momentum) * 10,
                0.1  # Minimum strength
            ) * (0.5 + (1 - bar_strength) / 2)
        else:
            strength = 0.0
            
        return bullish_break, bearish_break, strength
        
    def _check_prior_trend(self, df: pd.DataFrame) -> float:
        """Detect prior trend direction and strength
        Returns: trend score [-1 to +1]"""
        if len(df) < self.lb_medium + 1:
            return 0.0
            
        # Calculate price changes over different windows
        short_change = df['close'].iloc[-1] / df['close'].iloc[-self.lb_short] - 1
        medium_change = df['close'].iloc[-1] / df['close'].iloc[-self.lb_medium] - 1
        
        # Calculate moving averages for trend detection
        short_ma = df['close'].iloc[-self.lb_short:].mean()
        medium_ma = df['close'].iloc[-self.lb_medium:].mean()
        
        # Combine signals for trend strength and direction
        trend_direction = np.sign(short_change + medium_change)
        trend_strength = min(1.0, (abs(short_change) * 50) + (abs(medium_change) * 30))
        
        # Additional weight if moving averages confirm trend
        if (short_ma > medium_ma and trend_direction > 0) or (short_ma < medium_ma and trend_direction < 0):
            trend_strength *= 1.2
            
        return trend_direction * min(1.0, trend_strength)
        
    def _check_followthrough(self, df: pd.DataFrame, bullish: bool) -> float:
        """Check for follow-through confirmation after breakout
        Returns: confirmation strength [0-1]"""
        if len(df) < self.confirm_bars + 1:
            return 0.0
            
        # Get relevant bars
        confirm_window = df.iloc[-self.confirm_bars-1:]
        
        if bullish:
            # For bullish breakout, check if price continues higher
            initial_close = confirm_window['close'].iloc[0]
            subsequent_closes = confirm_window['close'].iloc[1:]
            
            # Count number of higher closes
            higher_closes = sum(close > initial_close for close in subsequent_closes)
            
            # Calculate average volume compared to breakout bar
            initial_volume = confirm_window['volume'].iloc[0]
            avg_subsequent_volume = confirm_window['volume'].iloc[1:].mean()
            
            # Stronger confirmation with higher closes and sustained volume
            confirm_score = (higher_closes / len(subsequent_closes)) * 0.7
            
            # Add bonus for continued higher volume
            if avg_subsequent_volume > initial_volume * 0.7:
                confirm_score += 0.3
                
            return confirm_score
            
        else:
            # For bearish breakdown, check if price continues lower
            initial_close = confirm_window['close'].iloc[0]
            subsequent_closes = confirm_window['close'].iloc[1:]
            
            # Count number of lower closes
            lower_closes = sum(close < initial_close for close in subsequent_closes)
            
            # Calculate average volume compared to breakdown bar
            initial_volume = confirm_window['volume'].iloc[0]
            avg_subsequent_volume = confirm_window['volume'].iloc[1:].mean()
            
            # Stronger confirmation with lower closes and sustained volume
            confirm_score = (lower_closes / len(subsequent_closes)) * 0.7
            
            # Add bonus for continued higher volume
            if avg_subsequent_volume > initial_volume * 0.7:
                confirm_score += 0.3
                
            return confirm_score
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        # No training needed for this rule-based agent
        pass
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """Generate signal based on volume breakout conditions"""
        print(f"\nVolumeBreakoutAgent Debug:")
        print(f"Data length: {len(historical_df)}")
        print(f"Required length: {self.lb_medium + self.confirm_bars + 1}")
        
        if len(historical_df) < self.lb_medium + self.confirm_bars + 1:
            print("Not enough data")
            return 0.0  # Not enough data
            
        # Check for volume spike
        volume_spike, vol_increase = self._detect_volume_spike(historical_df)
        print(f"Volume spike detected: {volume_spike}")
        print(f"Volume increase: {vol_increase}")
        
        # Modified to still generate signals with lower volume
        if vol_increase < 0.3:  # Minimum 30% of threshold
            print("Volume increase too low")
            return 0.0
            
        # Check price action for breakout
        bullish_break, bearish_break, price_strength = self._check_price_breakout(historical_df)
        print(f"Bullish break: {bullish_break}")
        print(f"Bearish break: {bearish_break}")
        print(f"Price strength: {price_strength}")
        
        # Allow weaker breakouts to still generate signals
        if not (bullish_break or bearish_break):
            print("No breakout detected")
            return 0.0
            
        # Check prior consolidation
        consolidation = self._check_consolidation(historical_df)
        
        # Check prior trend
        prior_trend = self._check_prior_trend(historical_df)
        
        # Initialize score
        score = 0.0
        
        # Generate breakout signal
        if bullish_break:
            # Bullish breakout
            # Base score on price and volume strength
            base_score = price_strength * vol_increase  # Removed threshold division
            
            # Stronger if breakout from consolidation
            if consolidation > 0.3:
                base_score *= (1.0 + consolidation * 0.5)
                
            # Weaker if against prior trend, stronger if with trend
            if prior_trend < 0:
                # Potential reversal - more confirmation needed
                base_score *= 0.7
            elif prior_trend > 0:
                # Trend continuation - more reliable
                base_score *= 1.2
                
            # Check for follow-through if we have enough data
            if len(historical_df) > self.lb_medium + self.confirm_bars + 1:
                followthrough = self._check_followthrough(historical_df, True)
                
                # Adjust score based on follow-through
                if followthrough > 0.5:
                    # Strong follow-through confirms breakout
                    base_score *= 1.3
                elif followthrough < 0.2:
                    # Poor follow-through suggests false breakout
                    base_score *= 0.7  # Changed from 0.5 to be less punitive
                    
            score = base_score
            
        elif bearish_break:
            # Bearish breakdown
            # Base score on price and volume strength
            base_score = -price_strength * vol_increase  # Removed threshold division
            
            # Stronger if breakdown from consolidation
            if consolidation > 0.3:
                base_score *= (1.0 + consolidation * 0.5)
                
            # Weaker if against prior trend, stronger if with trend
            if prior_trend > 0:
                # Potential reversal - more confirmation needed
                base_score *= 0.7
            elif prior_trend < 0:
                # Trend continuation - more reliable
                base_score *= 1.2
                
            # Check for follow-through if we have enough data
            if len(historical_df) > self.lb_medium + self.confirm_bars + 1:
                followthrough = self._check_followthrough(historical_df, False)
                
                # Adjust score based on follow-through
                if followthrough > 0.5:
                    # Strong follow-through confirms breakdown
                    base_score *= 1.3
                elif followthrough < 0.2:
                    # Poor follow-through suggests false breakdown
                    base_score *= 0.7  # Changed from 0.5 to be less punitive
                    
            score = base_score
            
        return float(np.clip(score, -1.0, 1.0))
        
    def __str__(self) -> str:
        return "Volume Breakout Agent" 