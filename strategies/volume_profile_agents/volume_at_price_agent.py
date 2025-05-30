"""
Volume At Price (VAP) Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~
Identifies significant volume nodes (price levels with high trading activity)
and generates signals based on price action relative to these volume nodes.
Key volume levels serve as support/resistance and can indicate potential
reversal or continuation points.

Logic:
1. Create volume profile by binning historical volume at price levels
2. Identify key volume nodes (high volume areas) and low volume nodes (areas of low liquidity)
3. Generate signals when:
   - Price tests and bounces from high volume node (potential support/resistance)
   - Price breaks through high volume node with strong momentum
   - Price moves quickly through low volume node (potential vacuum)
4. Scale signals based on:
   - Volume node significance (relative volume at the level)
   - Price momentum during the test/breakout
   - Historical reliability of the level

Input: OHLCV DataFrame. Output ∈ [-1, +1].
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple


class VolumeAtPriceAgent:
    def __init__(
        self,
        lookback_periods: int = 100,
        num_bins: int = 50,
        high_vol_threshold: float = 0.7,  # Top percentile to consider high volume
        low_vol_threshold: float = 0.3,   # Bottom percentile to consider low volume
        window_size: int = 5,             # Recent window to check for level tests
        distance_threshold: float = 0.02   # Distance threshold for level tests (2%)
    ):
        self.lookback = lookback_periods
        self.bins = num_bins
        self.high_threshold = high_vol_threshold
        self.low_threshold = low_vol_threshold
        self.window = window_size
        self.distance_threshold = distance_threshold
        
        # Will be set in fit()
        self.volume_profile = None
        self.high_vol_nodes = []
        self.low_vol_nodes = []
        self.price_range = (0.0, 0.0)
        
    def _build_volume_profile(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """Build volume profile by binning price and accumulating volume"""
        print("\nBuilding Volume Profile:")
        print(f"Data points: {len(df)}")
        
        # Extract price range with some padding
        price_min = df["low"].min() * 0.995  # Add 0.5% padding
        price_max = df["high"].max() * 1.005  # Add 0.5% padding
        price_range = (price_min, price_max)
        
        print(f"Price range: {price_min:.2f} - {price_max:.2f}")
        
        # Create price bins
        price_bins = np.linspace(price_min, price_max, self.bins + 1)
        bin_width = (price_max - price_min) / self.bins
        
        print(f"Number of bins: {self.bins}")
        print(f"Bin width: {bin_width:.2f}")
        
        # Initialize volume profile
        volume_profile = np.zeros(self.bins)
        
        # Debug counters
        total_volume = 0
        processed_candles = 0
        
        # Aggregate volume at each price level with weighted distribution
        for _, row in df.iterrows():
            # Determine which bins this candle spans
            low_bin = max(0, int((row["low"] - price_min) / bin_width))
            high_bin = min(self.bins - 1, int((row["high"] - price_min) / bin_width))
            
            # Calculate volume per price point (VWAP-style weighting)
            if row["high"] > row["low"]:
                price_points = np.linspace(row["low"], row["high"], high_bin - low_bin + 1)
                weights = np.exp(-0.5 * ((price_points - row["close"]) / (row["high"] - row["low"])) ** 2)
                weights = weights / weights.sum()  # Normalize weights
                vol_distribution = row["volume"] * weights
            else:
                # If high == low, put all volume in one bin
                vol_distribution = np.array([row["volume"]])
                
            # Distribute volume
            for i, vol in zip(range(low_bin, high_bin + 1), vol_distribution):
                volume_profile[i] += vol
                
            total_volume += row["volume"]
            processed_candles += 1
            
        print(f"Processed candles: {processed_candles}")
        print(f"Total volume: {total_volume:,.0f}")
        
        # Calculate bin centers (price levels)
        bin_centers = price_bins[:-1] + bin_width / 2
        
        # Debug volume distribution
        non_zero_bins = np.count_nonzero(volume_profile)
        print(f"Bins with volume: {non_zero_bins} ({non_zero_bins/self.bins*100:.1f}%)")
        print(f"Max volume in a bin: {np.max(volume_profile):,.0f}")
        print(f"Mean volume in non-zero bins: {np.mean(volume_profile[volume_profile > 0]):,.0f}")
        
        # Smooth the profile
        volume_profile = np.convolve(volume_profile, [0.25, 0.5, 0.25], mode='same')
        
        return volume_profile, bin_centers, price_range
        
    def _identify_volume_nodes(self, volume_profile: np.ndarray, bin_centers: np.ndarray) -> Tuple[List[float], List[float]]:
        """Identify high and low volume nodes"""
        print("\nIdentifying Volume Nodes:")
        
        # Calculate moving average for trend
        window = 3
        vol_ma = np.convolve(volume_profile, np.ones(window)/window, mode='same')
        
        # Find local maxima and minima
        peaks = []
        troughs = []
        
        for i in range(1, len(volume_profile)-1):
            if volume_profile[i] > volume_profile[i-1] and volume_profile[i] > volume_profile[i+1]:
                peaks.append((bin_centers[i], volume_profile[i]))
            elif volume_profile[i] < volume_profile[i-1] and volume_profile[i] < volume_profile[i+1]:
                troughs.append((bin_centers[i], volume_profile[i]))
                
        print(f"Found {len(peaks)} peaks and {len(troughs)} troughs")
        
        # Sort by volume
        peaks.sort(key=lambda x: x[1], reverse=True)
        troughs.sort(key=lambda x: x[1])
        
        # Take top percentile nodes, ensuring we have nodes above and below current price
        high_vol_prices = []
        low_vol_prices = []
        
        # Get current price from bin centers median
        current_price = np.median(bin_centers)
        
        # Split peaks into above and below current price
        peaks_below = [(p, v) for p, v in peaks if p < current_price]
        peaks_above = [(p, v) for p, v in peaks if p > current_price]
        
        # Take top nodes from each side
        num_each_side = max(2, int(len(peaks) * self.high_threshold / 2))
        high_vol_prices.extend([price for price, _ in peaks_below[:num_each_side]])
        high_vol_prices.extend([price for price, _ in peaks_above[:num_each_side]])
        
        # Same for troughs
        troughs_below = [(p, v) for p, v in troughs if p < current_price]
        troughs_above = [(p, v) for p, v in troughs if p > current_price]
        
        num_each_side = max(2, int(len(troughs) * self.low_threshold / 2))
        low_vol_prices.extend([price for price, _ in troughs_below[:num_each_side]])
        low_vol_prices.extend([price for price, _ in troughs_above[:num_each_side]])
        
        # Sort by price level
        high_vol_prices.sort()
        low_vol_prices.sort()
        
        print("\nHigh volume nodes:")
        for price in high_vol_prices:
            print(f"  {price:.2f}")
            
        print("\nLow volume nodes:")
        for price in low_vol_prices:
            print(f"  {price:.2f}")
        
        return high_vol_prices, low_vol_prices
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Build volume profile from historical data"""
        print("\nFitting VolumeAtPriceAgent:")
        print(f"Historical data points: {len(historical_df)}")
        
        if len(historical_df) < self.lookback:
            print("Not enough data")
            return  # Not enough data
            
        # First get recent data
        recent_data = historical_df.tail(self.lookback).copy()
        current_price = recent_data.iloc[-1]['close']
        print(f"Current price: {current_price:.2f}")
        
        # Then filter by price range around current price
        price_range = current_price * 0.02  # 2% range
        mask = (
            (recent_data['high'] >= current_price - price_range) & 
            (recent_data['low'] <= current_price + price_range)
        )
        filtered_data = recent_data[mask]
        
        if len(filtered_data) < 10:  # Need at least 10 bars
            print(f"Not enough data within price range (found {len(filtered_data)} bars)")
            filtered_data = recent_data  # Fall back to all recent data
            
        print(f"Using {len(filtered_data)} recent bars near current price {current_price:.2f}")
        print(f"Date range: {filtered_data.index[0]} to {filtered_data.index[-1]}")
        
        # Build volume profile
        self.volume_profile, bin_centers, self.price_range = self._build_volume_profile(filtered_data)
        
        # Identify volume nodes
        self.high_vol_nodes, self.low_vol_nodes = self._identify_volume_nodes(self.volume_profile, bin_centers)
        
    def _find_closest_nodes(self, price: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Find closest high and low volume nodes above and below current price"""
        if not self.high_vol_nodes or not self.low_vol_nodes:
            return (None, None), (None, None)
            
        # Find closest high volume nodes
        nodes_below = [node for node in self.high_vol_nodes if node < price]
        nodes_above = [node for node in self.high_vol_nodes if node > price]
        
        high_below = max(nodes_below) if nodes_below else None
        high_above = min(nodes_above) if nodes_above else None
        
        # Find closest low volume nodes
        nodes_below = [node for node in self.low_vol_nodes if node < price]
        nodes_above = [node for node in self.low_vol_nodes if node > price]
        
        low_below = max(nodes_below) if nodes_below else None
        low_above = min(nodes_above) if nodes_above else None
        
        # If we don't have nodes on both sides, try to use next closest nodes
        if high_below is None and len(self.high_vol_nodes) >= 2:
            # Take the two lowest nodes above as our levels
            sorted_nodes = sorted(self.high_vol_nodes)
            high_below = sorted_nodes[0]
            high_above = sorted_nodes[1]
        elif high_above is None and len(self.high_vol_nodes) >= 2:
            # Take the two highest nodes below as our levels
            sorted_nodes = sorted(self.high_vol_nodes, reverse=True)
            high_above = sorted_nodes[0]
            high_below = sorted_nodes[1]
            
        # Same for low volume nodes
        if low_below is None and len(self.low_vol_nodes) >= 2:
            sorted_nodes = sorted(self.low_vol_nodes)
            low_below = sorted_nodes[0]
            low_above = sorted_nodes[1]
        elif low_above is None and len(self.low_vol_nodes) >= 2:
            sorted_nodes = sorted(self.low_vol_nodes, reverse=True)
            low_above = sorted_nodes[0]
            low_below = sorted_nodes[1]
            
        return (high_below, high_above), (low_below, low_above)
        
    def _check_level_test(self, df: pd.DataFrame, level: float, direction: str) -> float:
        """
        Check if price has recently tested a volume level
        direction: 'support' or 'resistance'
        Returns strength of test [0, 1]
        """
        if len(df) < self.window:
            return 0.0
            
        recent_bars = df.iloc[-self.window:]
        
        # For support test (price approached from above)
        if direction == 'support':
            # Check if price got close to level
            min_low = recent_bars['low'].min()
            test_strength = 0.0
            
            # How close price got to level (as % of recent range)
            if min_low > level:
                recent_range = recent_bars['high'].max() - recent_bars['low'].min()
                if recent_range > 0:
                    distance = (min_low - level) / recent_range
                    if distance < 0.05:  # Within 5% of recent range
                        test_strength = 1.0 - (distance / 0.05)
                        
            # Actual test or break of level
            elif min_low <= level and recent_bars.iloc[-1]['close'] > level:
                # Bounced off support
                test_strength = 1.0
                
            return test_strength
                
        # For resistance test (price approached from below)
        elif direction == 'resistance':
            # Check if price got close to level
            max_high = recent_bars['high'].max()
            test_strength = 0.0
            
            # How close price got to level (as % of recent range)
            if max_high < level:
                recent_range = recent_bars['high'].max() - recent_bars['low'].min()
                if recent_range > 0:
                    distance = (level - max_high) / recent_range
                    if distance < 0.05:  # Within 5% of recent range
                        test_strength = 1.0 - (distance / 0.05)
                        
            # Actual test or break of level
            elif max_high >= level and recent_bars.iloc[-1]['close'] < level:
                # Rejected at resistance
                test_strength = 1.0
                
            return test_strength
            
        return 0.0
        
    def _check_level_break(self, df: pd.DataFrame, level: float, direction: str) -> float:
        """
        Check if price has recently broken through a volume level
        direction: 'breakout' (bullish) or 'breakdown' (bearish)
        Returns strength of break [0, 1]
        """
        if len(df) < self.window + 1:
            return 0.0
            
        # Check if we've broken the level recently
        if direction == 'breakout':  # Bullish break of resistance
            # Price closed below level before window, now closed above
            if df.iloc[-(self.window+1)]['close'] < level and df.iloc[-1]['close'] > level:
                # Calculate momentum of break
                momentum = (df.iloc[-1]['close'] - level) / level
                volume_increase = df.iloc[-1]['volume'] / df.iloc[-5:-1]['volume'].mean() - 1
                
                # Strong break = price momentum + volume confirmation
                strength = min(1.0, (momentum * 100) + max(0, volume_increase * 0.5))
                return strength
                
        elif direction == 'breakdown':  # Bearish break of support
            # Price closed above level before window, now closed below
            if df.iloc[-(self.window+1)]['close'] > level and df.iloc[-1]['close'] < level:
                # Calculate momentum of break
                momentum = (level - df.iloc[-1]['close']) / level
                volume_increase = df.iloc[-1]['volume'] / df.iloc[-5:-1]['volume'].mean() - 1
                
                # Strong break = price momentum + volume confirmation
                strength = min(1.0, (momentum * 100) + max(0, volume_increase * 0.5))
                return strength
                
        return 0.0
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """Generate signal based on price relative to volume profile"""
        print("\nVolumeAtPriceAgent Debug:")
        print(f"Data length: {len(historical_df)}")
        print(f"Current price: {current_price}")
        
        # Build profile if we don't have one
        if self.volume_profile is None or len(historical_df) > self.lookback:
            self.fit(historical_df)
            
        if not self.high_vol_nodes or len(historical_df) < self.window + 1:
            print("Not enough data or no volume nodes found")
            return 0.0  # Can't generate signal
            
        # Get current price
        price = float(current_price)
        
        # Find closest volume nodes
        (high_below, high_above), (low_below, low_above) = self._find_closest_nodes(price)
        
        print(f"\nVolume Nodes:")
        print(f"High volume nodes below: {high_below}")
        print(f"High volume nodes above: {high_above}")
        print(f"Low volume nodes below: {low_below}")
        print(f"Low volume nodes above: {low_above}")
        
        # Initialize score
        score = 0.0
        
        # Check for significant levels
        recent_df = historical_df.iloc[-self.window-1:]
        
        # Check high volume support test/bounce
        if high_below is not None:
            # Distance to level as percent
            distance = (price - high_below) / price
            print(f"\nSupport Test:")
            print(f"Distance to support: {distance * 100:.2f}%")
            
            if distance < self.distance_threshold:  # Within distance threshold of high volume support
                # Check for support test
                test_strength = self._check_level_test(recent_df, high_below, 'support')
                print(f"Support test strength: {test_strength}")
                if test_strength > 0:
                    # Bullish signal - stronger the closer we bounced to the exact level
                    signal = test_strength * (1.0 - distance / self.distance_threshold)
                    print(f"Support signal: +{signal:.4f}")
                    score += signal
                    
        # Check high volume resistance test/reject
        if high_above is not None:
            # Distance to level as percent
            distance = (high_above - price) / price
            print(f"\nResistance Test:")
            print(f"Distance to resistance: {distance * 100:.2f}%")
            
            if distance < self.distance_threshold:  # Within distance threshold of high volume resistance
                # Check for resistance test
                test_strength = self._check_level_test(recent_df, high_above, 'resistance')
                print(f"Resistance test strength: {test_strength}")
                if test_strength > 0:
                    # Bearish signal - stronger the closer we rejected at the exact level
                    signal = -test_strength * (1.0 - distance / self.distance_threshold)
                    print(f"Resistance signal: {signal:.4f}")
                    score += signal
                    
        # Check for breakouts/breakdowns
        if high_above is not None:
            break_strength = self._check_level_break(recent_df, high_above, 'breakout')
            print(f"\nBreakout Test:")
            print(f"Breakout strength: {break_strength}")
            if break_strength > 0:
                # Bullish breakout signal
                signal = break_strength * 1.2  # Increased weight for breakouts
                print(f"Breakout signal: +{signal:.4f}")
                score += signal
                
        if high_below is not None:
            break_strength = self._check_level_break(recent_df, high_below, 'breakdown')
            print(f"\nBreakdown Test:")
            print(f"Breakdown strength: {break_strength}")
            if break_strength > 0:
                # Bearish breakdown signal
                signal = -break_strength * 1.2  # Increased weight for breakdowns
                print(f"Breakdown signal: {signal:.4f}")
                score += signal
                
        # Check if price is in low volume node (vacuum)
        if low_below is not None and low_above is not None:
            # Calculate relative position between low volume nodes
            if low_below < price < low_above:
                node_range = low_above - low_below
                if node_range > 0:
                    position = (price - low_below) / node_range
                    print(f"\nVacuum Zone:")
                    print(f"Position in vacuum: {position * 100:.2f}%")
                    
                    # Recent momentum
                    recent_momentum = (recent_df.iloc[-1]['close'] - recent_df.iloc[-3]['close']) / recent_df.iloc[-3]['close']
                    print(f"Recent momentum: {recent_momentum * 100:.2f}%")
                    
                    # Stronger signal in direction of momentum
                    if abs(recent_momentum) > 0.001:  # 0.1% move
                        momentum_direction = np.sign(recent_momentum)
                        # Scale based on position in vacuum zone and momentum
                        vacuum_score = momentum_direction * 0.5 * min(abs(recent_momentum) * 300, 1.0)
                        # Stronger effect near edges of vacuum
                        edge_effect = 1.0 - abs(position - 0.5) * 2  # 1.0 at edges, 0.0 in middle
                        signal = vacuum_score * (1.0 + edge_effect)
                        print(f"Vacuum signal: {signal:.4f}")
                        score += signal
                
        # Add small bias based on closest high volume node
        if high_below is not None and high_above is not None:
            # Calculate relative position between high volume nodes
            node_range = high_above - high_below
            if node_range > 0:
                position = (price - high_below) / node_range
                print(f"\nPosition between nodes: {position * 100:.2f}%")
                # Bias towards closest node
                if position < 0.5:
                    score += 0.1  # Small upward bias when closer to support
                    print("Added upward bias: +0.1")
                else:
                    score -= 0.1  # Small downward bias when closer to resistance
                    print("Added downward bias: -0.1")
                
        print(f"\nFinal score: {score}")
        return float(np.clip(score, -1.0, 1.0))
        
    def __str__(self) -> str:
        return "Volume At Price (VAP) Agent" 