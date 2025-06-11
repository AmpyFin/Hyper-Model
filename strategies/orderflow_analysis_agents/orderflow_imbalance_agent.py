"""
OrderFlow Imbalance Agent
~~~~~~~~~~~~~~~~~~~~~
Detects imbalances in the order book by comparing buying and selling limit orders
at different price levels. Large imbalances often precede price movements.

Logic:
1. Analyze the order book to identify price levels with significant imbalances
2. Calculate the ratio of buy orders to sell orders at key price levels
3. Weight imbalances by their proximity to the current price
4. Generate signals when order book imbalances suggest potential price movement

Input: OHLCV DataFrame with order book data. Output ∈ [-1, +1] where:
* Positive values: More buying interest than selling interest
* Negative values: More selling interest than buying interest
* Values near zero: Balanced order book
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class OrderFlowImbalanceAgent:
    def __init__(
        self,
        price_levels: int = 5,           # Number of price levels to analyze
        imbalance_threshold: float = 2.0, # Threshold for significant imbalance (ratio)
        distance_decay: float = 0.8,     # Weight decay for farther price levels
        signal_smoothing: int = 2,       # Reduced from 3 to 2 for more responsiveness
        use_order_flow_proxy: bool = True # Use proxy if order book data unavailable
    ):
        self.price_levels = price_levels
        self.imbalance_threshold = imbalance_threshold
        self.distance_decay = distance_decay
        self.signal_smoothing = signal_smoothing
        self.use_order_flow_proxy = use_order_flow_proxy
        self.latest_signal = 0.0
        self.signal_history = []
        self.imbalance_data = {}
        
    def _calculate_orderbook_imbalance(self, df: pd.DataFrame) -> Dict:
        """Calculate order book imbalance from order book data"""
        result = {}
        
        # Check if we have order book data
        has_orderbook = all(col in df.columns for col in ['bid_size', 'ask_size'])
        
        if has_orderbook:
            # Use actual order book data
            latest_row = df.iloc[-1]
            
            # Basic imbalance ratio at best bid/ask with non-linear scaling
            bid_size = latest_row['bid_size']
            ask_size = latest_row['ask_size']
            
            if ask_size > 0:
                result['bid_ask_ratio'] = np.tanh(np.log(bid_size / ask_size))
            else:
                result['bid_ask_ratio'] = 0.0  # Default to balanced
                
            # Calculate imbalance at each price level with exponential decay
            level_ratios = []
            level_weights = []
            
            for i in range(1, self.price_levels + 1):
                bid_col = f'bid_size_{i}'
                ask_col = f'ask_size_{i}'
                
                if bid_col in df.columns and ask_col in df.columns:
                    bid_size_level = latest_row[bid_col]
                    ask_size_level = latest_row[ask_col]
                    
                    if ask_size_level > 0:
                        # Non-linear transformation of ratio
                        level_ratio = np.tanh(np.log(bid_size_level / ask_size_level))
                    else:
                        level_ratio = 0.0
                        
                    # Apply distance decay with exponential weighting
                    weight = np.exp(-self.distance_decay * (i - 1))
                    
                    level_ratios.append(level_ratio)
                    level_weights.append(weight)
            
            # Calculate weighted average imbalance
            if level_ratios:
                sum_weights = sum(level_weights)
                norm_weights = [w / sum_weights for w in level_weights]
                weighted_imbalance = sum(ratio * weight for ratio, weight in zip(level_ratios, norm_weights))
                result['weighted_imbalance'] = weighted_imbalance
            else:
                # Just use the best bid/ask ratio
                result['weighted_imbalance'] = result['bid_ask_ratio']
                
        elif self.use_order_flow_proxy:
            # Enhanced proxy calculation using multiple factors
            df_copy = df.copy()
            
            # 1. Price momentum (30% weight)
            df_copy['returns'] = df_copy['close'].pct_change()
            momentum = df_copy['returns'].rolling(10).mean().iloc[-1]
            momentum_signal = np.tanh(momentum * 100)  # Scale for sensitivity
            
            # 2. Volume momentum (30% weight)
            df_copy['volume_change'] = df_copy['volume'].pct_change()
            vol_momentum = df_copy['volume_change'].rolling(10).mean().iloc[-1]
            volume_signal = np.tanh(vol_momentum * 5)  # Scale for sensitivity
            
            # 3. Price-volume correlation (20% weight)
            recent_df = df_copy.iloc[-20:]
            price_vol_corr = recent_df['returns'].corr(recent_df['volume_change'])
            correlation_signal = np.tanh(price_vol_corr * 2)
            
            # 4. Trend strength (20% weight)
            df_copy['ma_fast'] = df_copy['close'].rolling(5).mean()
            df_copy['ma_slow'] = df_copy['close'].rolling(20).mean()
            trend_strength = (df_copy['ma_fast'].iloc[-1] / df_copy['ma_slow'].iloc[-1] - 1)
            trend_signal = np.tanh(trend_strength * 10)
            
            # Combine signals with weights
            proxy_signal = (
                0.3 * momentum_signal +
                0.3 * volume_signal +
                0.2 * correlation_signal +
                0.2 * trend_signal
            )
            
            result['proxy_signal'] = proxy_signal
            result['normalized_imbalance'] = proxy_signal
            
        else:
            # No order book data and proxy disabled
            result['normalized_imbalance'] = 0.0
            
        return result
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to calculate order book imbalance and generate signals
        """
        # Need enough bars for calculation
        if len(historical_df) < 20:  # Minimum required for proper proxy calculation
            self.latest_signal = 0.0
            return
            
        # Calculate order book imbalance
        self.imbalance_data = self._calculate_orderbook_imbalance(historical_df)
        
        # Get raw signal
        raw_signal = self.imbalance_data.get('normalized_imbalance', 0.0)
        
        # Apply adaptive smoothing
        self.signal_history.append(raw_signal)
        if len(self.signal_history) > self.signal_smoothing:
            self.signal_history.pop(0)
            
        # Use exponential weights for smoothing
        exp_weights = [np.exp(i) for i in range(len(self.signal_history))]
        sum_weights = sum(exp_weights)
        exp_weights = [w / sum_weights for w in exp_weights]
        
        self.latest_signal = sum(s * w for s, w in zip(self.signal_history, exp_weights))
        
        # Apply soft clipping for final signal
        self.latest_signal = np.tanh(self.latest_signal)
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict order book imbalance signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate more buying interest
          * Negative values indicate more selling interest
          * Values near zero indicate balanced order book
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "OrderFlow Imbalance Agent" 