"""
Volume Delta Agent
~~~~~~~~~~~~~~~
Analyzes the difference between buying and selling volume at each price level.
Buying volume is measured by trades executed at the ask price, while selling volume
is measured by trades executed at the bid price.

Logic:
1. Classify trades as buys or sells based on whether they were executed at bid or ask
2. Calculate volume delta (buy volume - sell volume) at each price level
3. Analyze the volume delta distribution to identify price levels with significant imbalances
4. Generate signals when volume delta indicates strong buying or selling pressure

Input: OHLCV DataFrame with bid/ask data. Output ∈ [-1, +1] where:
* Positive values: Buying pressure exceeds selling pressure
* Negative values: Selling pressure exceeds buying pressure
* Values near zero: Balanced buying and selling
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class VolumeDeltaAgent:
    def __init__(
        self,
        lookback_period: int = 50,        # Period for volume delta analysis
        volume_threshold: float = 0.6,    # Volume threshold for significant imbalance
        signal_smoothing: int = 3,        # Reduced from 5 to 3 to be more responsive
        use_vwap: bool = True,            # Use VWAP as a reference price
        delta_normalization: str = "total" # Method to normalize delta: "total" or "max"
    ):
        self.lookback_period = lookback_period
        self.volume_threshold = volume_threshold
        self.signal_smoothing = signal_smoothing
        self.use_vwap = use_vwap
        self.delta_normalization = delta_normalization
        self.latest_signal = 0.0
        self.signal_history = []
        self.volume_delta_data = {}
        
    def _calculate_volume_delta(self, df: pd.DataFrame) -> Dict:
        """Calculate volume delta from trade data"""
        # Check if we have the necessary bid/ask data
        # If not, we'll estimate using price movement
        has_bid_ask = all(col in df.columns for col in ['bid', 'ask'])
        
        result = {}
        df_copy = df.copy()
        
        if has_bid_ask:
            # Classify trades based on bid/ask
            df_copy['trade_type'] = np.where(df_copy['close'] >= df_copy['ask'], 'buy', 
                                      np.where(df_copy['close'] <= df_copy['bid'], 'sell', 'neutral'))
            
            # Calculate volume delta
            buy_volume = df_copy[df_copy['trade_type'] == 'buy']['volume'].sum()
            sell_volume = df_copy[df_copy['trade_type'] == 'sell']['volume'].sum()
            neutral_volume = df_copy[df_copy['trade_type'] == 'neutral']['volume'].sum()
            
            total_volume = buy_volume + sell_volume + neutral_volume
            
            # Store results
            result['buy_volume'] = buy_volume
            result['sell_volume'] = sell_volume
            result['neutral_volume'] = neutral_volume
            result['total_volume'] = total_volume
            
            # Calculate delta and ratio
            if total_volume > 0:
                result['volume_delta'] = buy_volume - sell_volume
                
                # Normalize delta based on configuration
                if self.delta_normalization == "total":
                    result['normalized_delta'] = result['volume_delta'] / total_volume
                else:  # max
                    max_vol = max(buy_volume, sell_volume)
                    result['normalized_delta'] = result['volume_delta'] / max_vol if max_vol > 0 else 0
            else:
                result['volume_delta'] = 0
                result['normalized_delta'] = 0
                
        else:
            # Estimate using price movement as a proxy
            df_copy['price_change'] = df_copy['close'].diff()
            
            # Positive price change suggests buying, negative suggests selling
            df_copy['est_trade_type'] = np.where(df_copy['price_change'] > 0, 'buy',
                                          np.where(df_copy['price_change'] < 0, 'sell', 'neutral'))
            
            # Weight volume by price change magnitude
            df_copy['weighted_volume'] = df_copy['volume'] * abs(df_copy['price_change'])
            
            # Calculate estimated volume delta
            buy_volume = df_copy[df_copy['est_trade_type'] == 'buy']['weighted_volume'].sum()
            sell_volume = df_copy[df_copy['est_trade_type'] == 'sell']['weighted_volume'].sum()
            neutral_volume = df_copy[df_copy['est_trade_type'] == 'neutral']['volume'].sum()
            
            # Normalize to account for price scale
            total_weighted = buy_volume + sell_volume
            
            # Store results
            result['buy_volume'] = buy_volume
            result['sell_volume'] = sell_volume
            result['neutral_volume'] = neutral_volume
            result['total_volume'] = buy_volume + sell_volume + neutral_volume
            
            # Calculate normalized delta
            if total_weighted > 0:
                result['volume_delta'] = buy_volume - sell_volume
                result['normalized_delta'] = result['volume_delta'] / total_weighted
            else:
                result['volume_delta'] = 0
                result['normalized_delta'] = 0
        
        # Calculate VWAP and related metrics
        if self.use_vwap:
            df_copy['vwap'] = (df_copy['close'] * df_copy['volume']).cumsum() / df_copy['volume'].cumsum()
            result['vwap'] = df_copy['vwap'].iloc[-1]
            result['close_vs_vwap'] = (df_copy['close'].iloc[-1] / result['vwap']) - 1
            
            # Calculate VWAP-based signal
            vwap_distance = result['close_vs_vwap']
            # Scale to [-1, 1] range with sigmoid-like function
            result['vwap_signal'] = np.tanh(vwap_distance * 5)  # *5 to make it more sensitive
        
        # Calculate volume profile signal
        price_levels = {}
        for _, row in df_copy.iterrows():
            price = round(row['close'], 2)  # Round to 2 decimal places for binning
            vol = row['volume']
            
            if price not in price_levels:
                price_levels[price] = {'buy': 0, 'sell': 0, 'neutral': 0}
                
            if has_bid_ask:
                trade_type = row['trade_type']
            else:
                trade_type = row['est_trade_type']
                
            price_levels[price][trade_type] += vol
        
        # Calculate delta at each price level and volume profile signal
        price_deltas = {}
        weighted_profile_signal = 0.0
        total_volume_at_levels = 0.0
        
        for price, data in price_levels.items():
            delta = data['buy'] - data['sell']
            total = data['buy'] + data['sell'] + data['neutral']
            if total > 0:
                price_deltas[price] = delta / total
                # Weight the signal by volume at this level
                weighted_profile_signal += (delta / total) * total
                total_volume_at_levels += total
                
        result['price_deltas'] = price_deltas
        
        # Calculate final volume profile signal
        if total_volume_at_levels > 0:
            result['volume_profile_signal'] = weighted_profile_signal / total_volume_at_levels
        else:
            result['volume_profile_signal'] = 0.0
            
        # Calculate momentum signal based on recent price changes
        if len(df_copy) >= 2:
            recent_returns = df_copy['close'].pct_change().fillna(0)
            # Use exponentially weighted momentum
            momentum = pd.Series(recent_returns).ewm(span=10).mean().iloc[-1]
            # Scale to [-1, 1] range
            result['momentum_signal'] = np.tanh(momentum * 100)  # *100 to make it more sensitive
        else:
            result['momentum_signal'] = 0.0
        
        return result
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to calculate volume delta and generate signals
        """
        # Need enough bars for calculation
        if len(historical_df) < self.lookback_period:
            self.latest_signal = 0.0
            return
            
        # Use lookback window if specified
        if self.lookback_period and len(historical_df) > self.lookback_period:
            df_subset = historical_df.iloc[-self.lookback_period:]
        else:
            df_subset = historical_df
            
        # Calculate volume delta metrics
        self.volume_delta_data = self._calculate_volume_delta(df_subset)
        
        # Generate signal based on volume delta metrics
        if not self.volume_delta_data:
            self.latest_signal = 0.0
            return
            
        signal_components = []
        weights = []
        
        # 1. Normalized delta as base signal (40% weight)
        if 'normalized_delta' in self.volume_delta_data:
            # Scale normalized delta more aggressively
            norm_delta = self.volume_delta_data['normalized_delta']
            scaled_delta = np.tanh(norm_delta * 4.0)  # Increased sensitivity
            signal_components.append(scaled_delta)
            weights.append(0.40)
            
        # 2. VWAP-based component (25% weight)
        if self.use_vwap and 'vwap_signal' in self.volume_delta_data:
            vwap_signal = self.volume_delta_data['vwap_signal']
            # Increase VWAP signal impact
            scaled_vwap = np.tanh(vwap_signal * 3.0)
            signal_components.append(scaled_vwap)
            weights.append(0.25)
            
        # 3. Volume profile component (20% weight)
        if 'volume_profile_signal' in self.volume_delta_data:
            profile_signal = self.volume_delta_data['volume_profile_signal']
            # Scale profile signal more aggressively
            scaled_profile = np.tanh(profile_signal * 3.5)
            signal_components.append(scaled_profile)
            weights.append(0.20)
            
        # 4. Momentum component (15% weight)
        if 'momentum_signal' in self.volume_delta_data:
            momentum_signal = self.volume_delta_data['momentum_signal']
            # Increase momentum sensitivity
            scaled_momentum = np.tanh(momentum_signal * 2.5)
            signal_components.append(scaled_momentum)
            weights.append(0.15)
            
        # Combine signals with weights
        if signal_components:
            # Normalize weights
            weights = [w / sum(weights) for w in weights]
            raw_signal = sum(s * w for s, w in zip(signal_components, weights))
            
            # Apply smoothing with reduced window
            self.signal_history.append(raw_signal)
            if len(self.signal_history) > self.signal_smoothing:
                self.signal_history.pop(0)
                
            # Use exponential smoothing with increased decay
            weights = np.exp(np.linspace(-2., 0., len(self.signal_history)))
            weights /= weights.sum()
            
            # Calculate final signal with increased sensitivity
            smoothed_signal = np.sum(np.array(self.signal_history) * weights)
            self.latest_signal = np.tanh(smoothed_signal * 2.0)  # Increased final scaling
        else:
            self.latest_signal = 0.0
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict volume delta signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate net buying pressure
          * Negative values indicate net selling pressure
          * Values near zero indicate balanced volume
        """
        # First update our state with the latest data
        self.fit(historical_df)
        
        # Return the latest smoothed signal
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation with latest metrics"""
        metrics = []
        if self.volume_delta_data:
            if 'normalized_delta' in self.volume_delta_data:
                metrics.append(f"Delta: {self.volume_delta_data['normalized_delta']:.2f}")
            if 'close_vs_vwap' in self.volume_delta_data:
                metrics.append(f"VWAP Diff: {self.volume_delta_data['close_vs_vwap']:.2%}")
            metrics.append(f"Signal: {self.latest_signal:.2f}")
        return " | ".join(metrics) if metrics else "No data" 