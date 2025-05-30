"""
Large Order Agent
~~~~~~~~~~~~~
Identifies unusually large orders/trades and assesses their potential impact on future price movement.
Large orders often represent institutional activity and can signal significant price levels.

Logic:
1. Identify trades with volume significantly above the average
2. Track the price levels where large orders execute
3. Monitor for absorption (large orders being filled without price impact)
4. Evaluate context of large orders to determine their significance

Input: OHLCV DataFrame with tick/trade data. Output ∈ [-1, +1] where:
* Positive values: Large buying orders detected, bullish
* Negative values: Large selling orders detected, bearish
* Values near zero: No significant large orders detected
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class LargeOrderAgent:
    def __init__(
        self,
        lookback_period: int = 100,       # Lookback period for volume analysis
        volume_threshold: float = 2.0,    # Multiple of average volume to identify large orders
        large_order_memory: int = 20,     # Bars to remember large orders
        absorption_threshold: float = 0.3, # Price movement threshold for absorption
        signal_decay: float = 0.8,        # Signal decay factor per bar
        min_large_orders: int = 3         # Minimum number of large orders needed for a signal
    ):
        self.lookback_period = lookback_period
        self.volume_threshold = volume_threshold
        self.large_order_memory = large_order_memory
        self.absorption_threshold = absorption_threshold
        self.signal_decay = signal_decay
        self.min_large_orders = min_large_orders
        self.latest_signal = 0.0
        self.recent_large_orders = []  # List of (bar_index, price, volume, direction)
        self.large_order_data = {}
        
    def _identify_large_orders(self, df: pd.DataFrame) -> Dict:
        """Identify large orders/trades in the historical data"""
        result = {}
        df_copy = df.copy()
        
        # Calculate average volume
        avg_volume = df_copy['volume'].mean()
        # Use standard deviation to identify outliers
        vol_std = df_copy['volume'].std()
        
        # Calculate volume threshold
        volume_cutoff = avg_volume + (vol_std * self.volume_threshold)
        
        # Mark large volume bars
        df_copy['is_large'] = df_copy['volume'] > volume_cutoff
        
        # Count large volume bars
        large_bars = df_copy[df_copy['is_large']]
        result['large_bar_count'] = len(large_bars)
        
        # If no large bars found, return early
        if result['large_bar_count'] == 0:
            return result
            
        # Store the large order indices for later reference
        result['large_bar_indices'] = large_bars.index.tolist()
        
        # Determine direction of large bars (based on price movement)
        large_bars_with_dir = []
        for idx in result['large_bar_indices']:
            # Get position in the dataframe
            pos = df_copy.index.get_loc(idx)
            
            # Skip if first bar
            if pos == 0:
                continue
                
            # Get current and previous bar
            curr_bar = df_copy.iloc[pos]
            prev_bar = df_copy.iloc[pos-1]
            
            # Determine direction based on close vs previous close
            direction = 1 if curr_bar['close'] > prev_bar['close'] else -1
            
            # Check for potential absorption (large volume with small price movement)
            price_change_pct = abs((curr_bar['close'] - prev_bar['close']) / prev_bar['close'])
            is_absorption = price_change_pct < self.absorption_threshold
            
            # Only count recent bars within memory window
            if pos >= len(df_copy) - self.large_order_memory:
                # Store (relative position, price, volume, direction, is_absorption)
                large_bars_with_dir.append(
                    (pos - len(df_copy), curr_bar['close'], curr_bar['volume'], direction, is_absorption)
                )
        
        # Store recent large orders
        self.recent_large_orders = large_bars_with_dir
        
        # Calculate buy/sell pressure from large orders
        buy_volume = sum(vol for _, _, vol, dir, _ in large_bars_with_dir if dir > 0)
        sell_volume = sum(vol for _, _, vol, dir, _ in large_bars_with_dir if dir < 0)
        
        # Calculate absorption volume (large orders with minimal price impact)
        absorption_buy = sum(vol for _, _, vol, dir, abs in large_bars_with_dir if dir > 0 and abs)
        absorption_sell = sum(vol for _, _, vol, dir, abs in large_bars_with_dir if dir < 0 and abs)
        
        # Store volumes
        result['large_buy_volume'] = buy_volume
        result['large_sell_volume'] = sell_volume
        result['absorption_buy'] = absorption_buy
        result['absorption_sell'] = absorption_sell
        
        # Calculate volume ratios if there's sufficient data
        total_large_volume = buy_volume + sell_volume
        if total_large_volume > 0:
            result['buy_ratio'] = buy_volume / total_large_volume
            
            # Calculate large order net volume ratio [-1, 1]
            result['large_order_net_ratio'] = (buy_volume - sell_volume) / total_large_volume
        else:
            result['buy_ratio'] = 0.5
            result['large_order_net_ratio'] = 0.0
            
        # Calculate price impact of large orders
        if len(large_bars_with_dir) > 1:
            # Get average price movement after large orders
            price_impacts = []
            
            for i, (pos, price, _, direction, _) in enumerate(large_bars_with_dir[:-1]):
                # Skip if this is the last order (no future data to measure impact)
                abs_pos = len(df_copy) + pos  # Convert relative to absolute position
                
                # Measure price impact over next 3 bars if available
                future_bars = min(3, len(df_copy) - abs_pos - 1)
                if future_bars <= 0:
                    continue
                    
                future_price = df_copy.iloc[abs_pos + future_bars]['close']
                price_change = (future_price - price) / price
                
                # Add to impacts with associated direction
                price_impacts.append((direction, price_change))
            
            # Calculate average directional impact
            if price_impacts:
                # Average impact in the direction of the large order
                same_dir_impacts = [impact for dir, impact in price_impacts if np.sign(impact) == dir]
                if same_dir_impacts:
                    result['avg_directional_impact'] = np.mean(same_dir_impacts)
                else:
                    result['avg_directional_impact'] = 0.0
                    
                # Average price impact regardless of direction
                result['avg_price_impact'] = np.mean([abs(impact) for _, impact in price_impacts])
                
        return result
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to identify large orders and generate signals
        """
        # Need enough bars for calculation
        if len(historical_df) < max(20, self.lookback_period):
            self.latest_signal = 0.0
            return
            
        # Use lookback window if specified
        if self.lookback_period and len(historical_df) > self.lookback_period:
            df_subset = historical_df.iloc[-self.lookback_period:]
        else:
            df_subset = historical_df
            
        # Identify large orders
        self.large_order_data = self._identify_large_orders(df_subset)
        
        # Generate signal based on recent large orders
        if not self.recent_large_orders:
            # No recent large orders
            self.latest_signal = 0.0
            return
            
        # Check if we have enough large orders to generate a meaningful signal
        if len(self.recent_large_orders) < self.min_large_orders:
            # Not enough large orders
            self.latest_signal = 0.0
            return
            
        # Calculate signal from large order net ratio
        base_signal = self.large_order_data.get('large_order_net_ratio', 0.0)
        
        # Adjust signal based on absorption
        # If absorption is high, the signal should be stronger in the opposite direction
        absorption_buy = self.large_order_data.get('absorption_buy', 0)
        absorption_sell = self.large_order_data.get('absorption_sell', 0)
        
        # Calculate total large volume
        total_large_volume = (self.large_order_data.get('large_buy_volume', 0) + 
                             self.large_order_data.get('large_sell_volume', 0))
        
        if total_large_volume > 0:
            # If buy absorption is high, it's bullish (selling is being absorbed)
            buy_absorption_ratio = absorption_buy / total_large_volume
            # If sell absorption is high, it's bearish (buying is being absorbed)
            sell_absorption_ratio = absorption_sell / total_large_volume
            
            # Adjust signal based on absorption
            absorption_adjustment = (sell_absorption_ratio - buy_absorption_ratio) * 0.3
            adjusted_signal = base_signal + absorption_adjustment
        else:
            adjusted_signal = base_signal
            
        # Check for recency effect - more recent large orders have more impact
        if self.recent_large_orders:
            # Calculate weighted signal based on recency
            weighted_signals = []
            total_weight = 0
            
            for i, (pos, _, vol, direction, _) in enumerate(self.recent_large_orders):
                # More recent = higher weight
                # Closer to 0 = more recent (pos is negative)
                recency_weight = self.signal_decay ** abs(pos)
                # Larger volume = higher weight
                volume_weight = vol / sum(v for _, _, v, _, _ in self.recent_large_orders)
                
                # Combined weight
                weight = recency_weight * volume_weight
                
                # Signal contribution
                signal = direction * weight
                
                weighted_signals.append(signal)
                total_weight += weight
            
            # Calculate weighted average signal
            if total_weight > 0:
                recency_signal = sum(weighted_signals) / total_weight
            else:
                recency_signal = 0.0
                
            # Combine base signal with recency signal
            final_signal = (adjusted_signal * 0.7) + (recency_signal * 0.3)
        else:
            final_signal = adjusted_signal
            
        # Ensure signal is in [-1, 1] range
        self.latest_signal = max(-1.0, min(1.0, final_signal))
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict large order signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate significant buying orders detected
          * Negative values indicate significant selling orders detected
          * Values near zero indicate no significant large orders
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Large Order Agent" 