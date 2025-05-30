"""
Price Ladder Agent
~~~~~~~~~~~~~~
Analyzes the limit order book (price ladder) to identify key price levels and evaluate
buying and selling pressure at different levels.

Logic:
1. Analyze the distribution of orders across price levels
2. Identify price levels with significant order clustering
3. Monitor for order book events like spoofing, iceberg orders, and large limit orders
4. Generate signals based on changes in order book structure and imbalances

Input: OHLCV DataFrame with order book data. Output ∈ [-1, +1] where:
* Positive values: Buy limit orders dominate the order book
* Negative values: Sell limit orders dominate the order book
* Values near zero: Balanced order book or insufficient data
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class PriceLadderAgent:
    def __init__(
        self,
        num_levels: int = 5,              # Number of price levels to analyze
        level_weights: Optional[List[float]] = None,  # Weights for each level
        imbalance_threshold: float = 2.0,  # Threshold for significant imbalance
        spoofing_detection: bool = True,   # Whether to detect potential spoofing
        iceberg_detection: bool = True,    # Whether to detect potential iceberg orders
        signal_smoothing: int = 2          # Reduced from 3 to 2 for more responsiveness
    ):
        self.num_levels = num_levels
        # Default weights give higher importance to closer levels with exponential decay
        self.level_weights = level_weights or [np.exp(-0.5 * i) for i in range(num_levels)]
        # Normalize weights
        sum_weights = sum(self.level_weights)
        self.level_weights = [w / sum_weights for w in self.level_weights]
        
        self.imbalance_threshold = imbalance_threshold
        self.spoofing_detection = spoofing_detection
        self.iceberg_detection = iceberg_detection
        self.signal_smoothing = signal_smoothing
        self.latest_signal = 0.0
        self.signal_history = []
        self.ladder_data = {}
        
    def _analyze_price_ladder(self, df: pd.DataFrame) -> Dict:
        """Analyze price ladder (order book) data"""
        result = {}
        
        # Check if we have order book data (at least some levels)
        has_orderbook = False
        for i in range(1, self.num_levels + 1):
            if all(col in df.columns for col in [f'bid_{i}', f'ask_{i}', f'bid_size_{i}', f'ask_size_{i}']):
                has_orderbook = True
                break
                
        if not has_orderbook:
            # Try alternative column naming
            for i in range(1, self.num_levels + 1):
                if all(col in df.columns for col in [f'bid_price_{i}', f'ask_price_{i}', f'bid_size_{i}', f'ask_size_{i}']):
                    has_orderbook = True
                    break
                    
        if not has_orderbook:
            # Check for basic bid/ask columns
            has_orderbook = all(col in df.columns for col in ['bid', 'ask', 'bid_size', 'ask_size'])
        
        if not has_orderbook:
            # No order book data - use price action as proxy
            df_copy = df.copy()
            df_copy['returns'] = df_copy['close'].pct_change()
            df_copy['volume_change'] = df_copy['volume'].pct_change()
            
            # Calculate momentum signals
            momentum = df_copy['returns'].rolling(10).mean().iloc[-1]
            vol_momentum = df_copy['volume_change'].rolling(10).mean().iloc[-1]
            
            # Price-volume correlation
            recent_df = df_copy.iloc[-20:]
            price_vol_corr = recent_df['returns'].corr(recent_df['volume_change'])
            
            # Trend strength
            df_copy['ma_fast'] = df_copy['close'].rolling(5).mean()
            df_copy['ma_slow'] = df_copy['close'].rolling(20).mean()
            trend_strength = (df_copy['ma_fast'].iloc[-1] / df_copy['ma_slow'].iloc[-1] - 1)
            
            # Combine signals
            proxy_signal = (
                0.4 * np.tanh(momentum * 100) +
                0.3 * np.tanh(vol_momentum * 5) +
                0.2 * np.tanh(price_vol_corr * 2) +
                0.1 * np.tanh(trend_strength * 10)
            )
            
            result['proxy_signal'] = proxy_signal
            return result
            
        # Get the latest row for analysis
        latest_row = df.iloc[-1]
        
        # Prepare arrays to store order book data
        bid_prices = []
        ask_prices = []
        bid_sizes = []
        ask_sizes = []
        
        # Extract order book data from the latest row
        # Handle different column naming conventions
        for i in range(1, self.num_levels + 1):
            # Try standard naming
            bid_col = f'bid_{i}' if f'bid_{i}' in df.columns else f'bid_price_{i}'
            ask_col = f'ask_{i}' if f'ask_{i}' in df.columns else f'ask_price_{i}'
            bid_size_col = f'bid_size_{i}'
            ask_size_col = f'ask_size_{i}'
            
            # Check if we have all needed columns for this level
            if all(col in df.columns for col in [bid_col, ask_col, bid_size_col, ask_size_col]):
                bid_prices.append(latest_row[bid_col])
                ask_prices.append(latest_row[ask_col])
                bid_sizes.append(latest_row[bid_size_col])
                ask_sizes.append(latest_row[ask_size_col])
                
        # If we don't have any data from the levels, try basic bid/ask
        if not bid_prices and 'bid' in df.columns and 'ask' in df.columns:
            bid_prices.append(latest_row['bid'])
            ask_prices.append(latest_row['ask'])
            bid_sizes.append(latest_row['bid_size'])
            ask_sizes.append(latest_row['ask_size'])
        
        # If we still don't have any data, return proxy result
        if not bid_prices or not ask_prices:
            return result
            
        # Store basic order book data
        result['spread'] = ask_prices[0] - bid_prices[0]
        result['midpoint'] = (ask_prices[0] + bid_prices[0]) / 2
        
        # Calculate total volume on each side of the book
        total_bid_size = sum(bid_sizes)
        total_ask_size = sum(ask_sizes)
        
        result['total_bid_size'] = total_bid_size
        result['total_ask_size'] = total_ask_size
        
        # Calculate weighted imbalance
        if len(bid_sizes) > 1 and len(ask_sizes) > 1:
            # Use weights for each level
            weights = self.level_weights[:min(len(bid_sizes), len(ask_sizes))]
            
            # Weighted sizes
            weighted_bid_size = sum(size * weight for size, weight in zip(bid_sizes, weights))
            weighted_ask_size = sum(size * weight for size, weight in zip(ask_sizes, weights))
            
            result['weighted_bid_size'] = weighted_bid_size
            result['weighted_ask_size'] = weighted_ask_size
            
            # Calculate weighted imbalance ratio
            if weighted_ask_size > 0:
                result['weighted_imbalance'] = weighted_bid_size / weighted_ask_size
            else:
                result['weighted_imbalance'] = 1.0
        else:
            # Only have one level
            if total_ask_size > 0:
                result['weighted_imbalance'] = total_bid_size / total_ask_size
            else:
                result['weighted_imbalance'] = 1.0
        
        # Detect potential spoofing (large orders that disappear quickly)
        if self.spoofing_detection and len(df) > 1:
            # Get previous row for comparison
            prev_row = df.iloc[-2]
            
            # Check for large changes in order sizes
            spoofing_evidence = 0
            
            for i in range(min(len(bid_sizes), len(ask_sizes))):
                # Column names might vary
                bid_size_col = f'bid_size_{i+1}'
                ask_size_col = f'ask_size_{i+1}'
                
                if bid_size_col in df.columns and ask_size_col in df.columns:
                    # Check for large decrease in bid size (possible sell spoof)
                    if prev_row[bid_size_col] > latest_row[bid_size_col] * 3:  # 300% decrease
                        spoofing_evidence -= 1
                        
                    # Check for large decrease in ask size (possible buy spoof)
                    if prev_row[ask_size_col] > latest_row[ask_size_col] * 3:  # 300% decrease
                        spoofing_evidence += 1
            
            result['spoofing_evidence'] = spoofing_evidence
            
        # Detect potential iceberg orders (orders that refill after being partially filled)
        if self.iceberg_detection and len(df) > 5:
            iceberg_evidence_buy = 0
            iceberg_evidence_sell = 0
            
            # Check if bid size keeps replenishing at the same price level
            recent_bids = df[['bid_1', 'bid_size_1']].iloc[-5:] if 'bid_1' in df.columns else df[['bid', 'bid_size']].iloc[-5:]
            
            # Count instances where price stayed same but size increased after decreasing
            for i in range(1, len(recent_bids) - 1):
                bid_col = 'bid_1' if 'bid_1' in df.columns else 'bid'
                bid_size_col = 'bid_size_1' if 'bid_size_1' in df.columns else 'bid_size'
                
                if (recent_bids[bid_col].iloc[i] == recent_bids[bid_col].iloc[i-1] and
                    recent_bids[bid_size_col].iloc[i] < recent_bids[bid_size_col].iloc[i-1] and
                    recent_bids[bid_size_col].iloc[i+1] > recent_bids[bid_size_col].iloc[i]):
                    iceberg_evidence_buy += 1
                    
            # Same check for asks
            recent_asks = df[['ask_1', 'ask_size_1']].iloc[-5:] if 'ask_1' in df.columns else df[['ask', 'ask_size']].iloc[-5:]
            
            for i in range(1, len(recent_asks) - 1):
                ask_col = 'ask_1' if 'ask_1' in df.columns else 'ask'
                ask_size_col = 'ask_size_1' if 'ask_size_1' in df.columns else 'ask_size'
                
                if (recent_asks[ask_col].iloc[i] == recent_asks[ask_col].iloc[i-1] and
                    recent_asks[ask_size_col].iloc[i] < recent_asks[ask_size_col].iloc[i-1] and
                    recent_asks[ask_size_col].iloc[i+1] > recent_asks[ask_size_col].iloc[i]):
                    iceberg_evidence_sell += 1
                    
            result['iceberg_evidence_buy'] = iceberg_evidence_buy
            result['iceberg_evidence_sell'] = iceberg_evidence_sell
            
        return result
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to calculate price ladder metrics and generate signals
        """
        # Calculate price ladder metrics
        self.ladder_data = self._analyze_price_ladder(historical_df)
        
        # Generate signal based on order book imbalance
        if not self.ladder_data:
            self.latest_signal = 0.0
            return
            
        signal_components = []
        weights = []
        
        # 1. Weighted imbalance (35% weight)
        if 'weighted_imbalance' in self.ladder_data:
            imbalance = self.ladder_data['weighted_imbalance']
            
            # Non-linear transformation with increased sensitivity
            if imbalance > 1:
                imb_signal = np.tanh((imbalance - 1) * 2.5)  # Increased from 1.5
            else:
                imb_signal = -np.tanh((1 - imbalance) * 2.5)  # Increased from 1.5
                
            signal_components.append(imb_signal)
            weights.append(0.35)
            
        # 2. Volume pressure (25% weight)
        if 'total_bid_size' in self.ladder_data and 'total_ask_size' in self.ladder_data:
            total_bid = self.ladder_data['total_bid_size']
            total_ask = self.ladder_data['total_ask_size']
            total_vol = total_bid + total_ask
            
            if total_vol > 0:
                vol_imbalance = (total_bid - total_ask) / total_vol
                # Apply more aggressive sigmoid transformation
                vol_signal = 2 / (1 + np.exp(-4 * vol_imbalance)) - 1  # Increased from -3
                signal_components.append(vol_signal)
                weights.append(0.25)
                
        # 3. Spoofing evidence (20% weight)
        if self.spoofing_detection and 'spoofing_evidence' in self.ladder_data:
            spoof_factor = self.ladder_data['spoofing_evidence'] * 0.6  # Increased from 0.4
            spoof_signal = -np.tanh(spoof_factor * 1.5)  # Added multiplier
            signal_components.append(spoof_signal)
            weights.append(0.20)
            
        # 4. Iceberg orders (20% weight)
        if self.iceberg_detection:
            buy_icebergs = self.ladder_data.get('iceberg_evidence_buy', 0)
            sell_icebergs = self.ladder_data.get('iceberg_evidence_sell', 0)
            
            # More aggressive scaling for iceberg impact
            iceberg_factor = (buy_icebergs - sell_icebergs) * 0.5  # Increased from 0.3
            ice_signal = np.tanh(iceberg_factor * 2.0)  # Added multiplier
            signal_components.append(ice_signal)
            weights.append(0.20)
            
        # 5. Proxy signal if available (use full weight if no other components)
        if 'proxy_signal' in self.ladder_data:
            # Scale proxy signal more aggressively
            proxy_signal = np.tanh(self.ladder_data['proxy_signal'] * 2.0)
            signal_components.append(proxy_signal)
            weights.append(1.0 if not weights else (1.0 - sum(weights)))
            
        # Combine signals with weights
        if signal_components:
            # Normalize weights
            weights = [w / sum(weights) for w in weights]
            raw_signal = sum(s * w for s, w in zip(signal_components, weights))
            
            # Apply adaptive smoothing based on signal strength
            self.signal_history.append(raw_signal)
            if len(self.signal_history) > self.signal_smoothing:
                self.signal_history.pop(0)
                
            # Use exponential weights with steeper decay
            exp_weights = [np.exp(i * 1.5) for i in range(len(self.signal_history))]  # Increased from i
            sum_weights = sum(exp_weights)
            exp_weights = [w / sum_weights for w in exp_weights]
            
            # Calculate final signal with increased sensitivity
            smoothed_signal = sum(s * w for s, w in zip(self.signal_history, exp_weights))
            self.latest_signal = np.tanh(smoothed_signal * 2.0)  # Added final scaling
        else:
            self.latest_signal = 0.0
            
        # Ensure signal stays in [-1, 1] range with soft clipping
        self.latest_signal = np.tanh(self.latest_signal)
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict price ladder signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate buy orders dominate the book
          * Negative values indicate sell orders dominate the book
          * Values near zero indicate a balanced book
        """
        # First update our state with the latest data
        self.fit(historical_df)
        
        # Return the latest smoothed signal
        return self.latest_signal
        
    def __str__(self) -> str:
        """String representation with latest metrics"""
        metrics = []
        if self.ladder_data:
            if 'weighted_imbalance' in self.ladder_data:
                metrics.append(f"Imbalance: {self.ladder_data['weighted_imbalance']:.2f}")
            if 'spoofing_evidence' in self.ladder_data:
                metrics.append(f"Spoof: {self.ladder_data['spoofing_evidence']:+d}")
            if 'iceberg_evidence_buy' in self.ladder_data:
                metrics.append(f"Iceberg B/S: {self.ladder_data['iceberg_evidence_buy']}/{self.ladder_data['iceberg_evidence_sell']}")
            metrics.append(f"Signal: {self.latest_signal:.2f}")
        return " | ".join(metrics) if metrics else "No data" 