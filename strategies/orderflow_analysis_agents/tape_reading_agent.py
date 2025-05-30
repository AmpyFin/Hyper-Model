"""
Tape Reading Agent
~~~~~~~~~~~~~~
Analyzes time and sales data (the "tape") to identify buying and selling patterns
and detect changes in trader behavior through transaction analysis.

Logic:
1. Monitor trade executions for changes in frequency, size, and aggression
2. Track transaction speed and clustering patterns
3. Detect sweeps, blocks, and unusual transaction activity
4. Compare trade flow with price movement to identify possible divergences

Input: OHLCV DataFrame with transaction data. Output ∈ [-1, +1] where:
* Positive values: Increasing buy-side aggression and transaction speed
* Negative values: Increasing sell-side aggression and transaction speed
* Values near zero: Balanced or inconclusive transaction patterns
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from collections import defaultdict


class TapeReadingAgent:
    def __init__(
        self,
        lookback_period: int = 50,       # Period for tape analysis
        transaction_window: int = 10,    # Window for transaction rate calculation
        block_size_threshold: float = 3.0, # Multiple of average size for block trades
        sweep_detection: bool = True,    # Whether to detect sweep patterns
        transaction_rate_weight: float = 0.4, # Weight of transaction rate in signal
        signal_smoothing: int = 3        # Periods for signal smoothing
    ):
        self.lookback_period = lookback_period
        self.transaction_window = transaction_window
        self.block_size_threshold = block_size_threshold
        self.sweep_detection = sweep_detection
        self.transaction_rate_weight = transaction_rate_weight
        self.signal_smoothing = signal_smoothing
        self.latest_signal = 0.0
        self.signal_history = []
        self.tape_data = {}
        
    def _analyze_tape(self, df: pd.DataFrame) -> Dict:
        """Analyze time and sales data (tape)"""
        result = {}
        
        # Check if we have transaction-level data
        has_transactions = 'trade_id' in df.columns or 'transaction_id' in df.columns
        
        if has_transactions:
            # We have actual transaction data
            # Create copy to avoid modifying original
            df_copy = df.copy()
            
            # Get transaction ID column
            tx_id_col = 'trade_id' if 'trade_id' in df.columns else 'transaction_id'
            
            # Ensure sorted by transaction ID or timestamp
            if 'timestamp' in df.columns:
                df_copy = df_copy.sort_values(by=['timestamp', tx_id_col])
            else:
                df_copy = df_copy.sort_values(by=tx_id_col)
                
            # Calculate basic transaction metrics
            result['transaction_count'] = len(df_copy)
            result['avg_transaction_size'] = df_copy['volume'].mean()
            
            # Identify transaction direction if available
            if 'direction' in df_copy.columns:
                # Use explicit direction
                buy_txs = df_copy[df_copy['direction'] == 'buy']
                sell_txs = df_copy[df_copy['direction'] == 'sell']
            elif all(col in df_copy.columns for col in ['price', 'bid', 'ask']):
                # Infer direction from price relative to bid/ask
                buy_txs = df_copy[df_copy['price'] >= df_copy['ask']]
                sell_txs = df_copy[df_copy['price'] <= df_copy['bid']]
            else:
                # Cannot determine direction
                buy_txs = pd.DataFrame()
                sell_txs = pd.DataFrame()
                
            # Calculate buy/sell metrics if direction is available
            if not buy_txs.empty or not sell_txs.empty:
                result['buy_count'] = len(buy_txs)
                result['sell_count'] = len(sell_txs)
                
                result['buy_volume'] = buy_txs['volume'].sum()
                result['sell_volume'] = sell_txs['volume'].sum()
                
                # Calculate average sizes
                if not buy_txs.empty:
                    result['avg_buy_size'] = buy_txs['volume'].mean()
                if not sell_txs.empty:
                    result['avg_sell_size'] = sell_txs['volume'].mean()
                
                # Calculate buy/sell ratio
                total_volume = result.get('buy_volume', 0) + result.get('sell_volume', 0)
                if total_volume > 0:
                    result['buy_ratio'] = result.get('buy_volume', 0) / total_volume
                    
                # Normalized buy/sell imbalance (-1 to +1)
                buy_vol = result.get('buy_volume', 0)
                sell_vol = result.get('sell_volume', 0)
                if buy_vol + sell_vol > 0:
                    result['volume_imbalance'] = (buy_vol - sell_vol) / (buy_vol + sell_vol)
                    
            # Detect large block trades
            block_threshold = result['avg_transaction_size'] * self.block_size_threshold
            block_trades = df_copy[df_copy['volume'] > block_threshold]
            
            result['block_trade_count'] = len(block_trades)
            result['block_trade_volume'] = block_trades['volume'].sum()
            
            # Calculate block trade direction if possible
            if 'direction' in block_trades.columns:
                block_buys = block_trades[block_trades['direction'] == 'buy']
                block_sells = block_trades[block_trades['direction'] == 'sell']
                
                result['block_buy_count'] = len(block_buys)
                result['block_sell_count'] = len(block_sells)
                result['block_buy_volume'] = block_buys['volume'].sum()
                result['block_sell_volume'] = block_sells['volume'].sum()
                
            # Calculate transaction rate
            if 'timestamp' in df_copy.columns:
                # Convert timestamp to seconds if needed
                if pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
                    # Get timestamp differences in seconds
                    time_diffs = df_copy['timestamp'].diff().dt.total_seconds()
                else:
                    # Assume timestamp is already numeric
                    time_diffs = df_copy['timestamp'].diff()
                    
                # Calculate average time between transactions
                avg_time_diff = time_diffs.mean()
                if avg_time_diff > 0:
                    result['transaction_rate'] = 1 / avg_time_diff  # Transactions per second
                    
                # Calculate recent rate (last window)
                recent_diffs = time_diffs.iloc[-min(self.transaction_window, len(time_diffs)):]
                recent_avg = recent_diffs.mean()
                if recent_avg > 0:
                    result['recent_rate'] = 1 / recent_avg
                    
                    # Calculate rate change
                    if 'transaction_rate' in result:
                        result['rate_change'] = (result['recent_rate'] / result['transaction_rate']) - 1
            
            # Detect sweeps (sequential transactions clearing multiple price levels)
            if self.sweep_detection and len(df_copy) > 5:
                sweeps = self._detect_sweeps(df_copy)
                result.update(sweeps)
                
        else:
            # No transaction data, use bar data to estimate
            if len(df) < 5:
                return result
                
            # Use changes in volume and price to infer transaction patterns
            df_copy = df.copy()
            
            # Calculate volume changes
            df_copy['volume_change'] = df_copy['volume'].pct_change()
            
            # Calculate average volume
            avg_volume = df_copy['volume'].mean()
            
            # Identify bars with above-average volume
            high_volume_bars = df_copy[df_copy['volume'] > avg_volume * 1.5]
            result['high_volume_bar_count'] = len(high_volume_bars)
            
            # Categorize high volume bars by price direction
            if not high_volume_bars.empty:
                up_bars = high_volume_bars[high_volume_bars['close'] > high_volume_bars['open']]
                down_bars = high_volume_bars[high_volume_bars['close'] < high_volume_bars['open']]
                
                result['up_volume'] = up_bars['volume'].sum()
                result['down_volume'] = down_bars['volume'].sum()
                
                # Calculate directional volume ratio
                total_high_volume = result.get('up_volume', 0) + result.get('down_volume', 0)
                if total_high_volume > 0:
                    result['directional_volume_ratio'] = (result.get('up_volume', 0) - 
                                                        result.get('down_volume', 0)) / total_high_volume
            
            # Calculate volume acceleration
            recent_volumes = df_copy['volume'].iloc[-5:]
            if len(recent_volumes) == 5:
                # Calculate slope of recent volume
                x = np.arange(5)
                slope, _ = np.polyfit(x, recent_volumes.values, 1)
                result['volume_acceleration'] = slope / avg_volume
            
        return result
    
    def _detect_sweeps(self, df: pd.DataFrame) -> Dict:
        """Detect sweep patterns in the tape"""
        result = {}
        
        # Check if we have the necessary columns
        if not all(col in df.columns for col in ['price', 'volume']):
            return result
            
        # Check if we have timestamps
        has_timestamps = 'timestamp' in df.columns
        
        # Prepare for sweep detection
        sweep_window = 10  # Maximum transactions to consider for a sweep
        sweep_time_threshold = 2.0  # Maximum seconds between transactions in a sweep
        
        # Initialize counters
        buy_sweeps = 0
        sell_sweeps = 0
        buy_sweep_volume = 0
        sell_sweep_volume = 0
        
        # Process the transactions to identify sweeps
        i = 0
        while i < len(df) - 1:
            # Start of potential sweep
            start_price = df['price'].iloc[i]
            current_direction = None
            sweep_transactions = 1
            sweep_volume = df['volume'].iloc[i]
            last_idx = i
            
            # Check subsequent transactions for sweep pattern
            for j in range(i + 1, min(i + sweep_window, len(df))):
                current_price = df['price'].iloc[j]
                
                # Check time difference if timestamps available
                if has_timestamps:
                    time_diff = df['timestamp'].iloc[j] - df['timestamp'].iloc[last_idx]
                    
                    # Convert to seconds if datetime
                    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        time_diff = time_diff.total_seconds()
                        
                    # Skip if too much time has passed
                    if time_diff > sweep_time_threshold:
                        break
                
                # Determine or continue direction
                if current_direction is None:
                    # First price difference, establish direction
                    if current_price > start_price:
                        current_direction = 'buy'  # Prices moving up = buy sweep
                    elif current_price < start_price:
                        current_direction = 'sell'  # Prices moving down = sell sweep
                    else:
                        # Same price, continue to next transaction
                        sweep_transactions += 1
                        sweep_volume += df['volume'].iloc[j]
                        last_idx = j
                        continue
                else:
                    # Check if continuing in the same direction
                    if (current_direction == 'buy' and current_price < df['price'].iloc[last_idx]) or \
                       (current_direction == 'sell' and current_price > df['price'].iloc[last_idx]):
                        # Direction changed, end sweep
                        break
                
                # Add to sweep
                sweep_transactions += 1
                sweep_volume += df['volume'].iloc[j]
                last_idx = j
            
            # Check if we've identified a sweep (at least 3 transactions)
            if sweep_transactions >= 3 and current_direction:
                if current_direction == 'buy':
                    buy_sweeps += 1
                    buy_sweep_volume += sweep_volume
                else:
                    sell_sweeps += 1
                    sell_sweep_volume += sweep_volume
                    
                # Skip past this sweep
                i = last_idx + 1
            else:
                # Not a sweep, move to next transaction
                i += 1
        
        # Store sweep results
        result['buy_sweeps'] = buy_sweeps
        result['sell_sweeps'] = sell_sweeps
        result['buy_sweep_volume'] = buy_sweep_volume
        result['sell_sweep_volume'] = sell_sweep_volume
        
        # Calculate sweep imbalance if any sweeps detected
        total_sweep_volume = buy_sweep_volume + sell_sweep_volume
        if total_sweep_volume > 0:
            result['sweep_imbalance'] = (buy_sweep_volume - sell_sweep_volume) / total_sweep_volume
            
        return result
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to interpret tape reading signals
        """
        # Need enough bars for calculation
        if len(historical_df) < 5:  # Minimum required data
            self.latest_signal = 0.0
            return
            
        # Use lookback window if specified
        if self.lookback_period and len(historical_df) > self.lookback_period:
            df_subset = historical_df.iloc[-self.lookback_period:]
        else:
            df_subset = historical_df
            
        # Analyze the tape
        self.tape_data = self._analyze_tape(df_subset)
        
        # Generate signal components
        signal_components = []
        weights = []
        
        # Volume imbalance component
        if 'volume_imbalance' in self.tape_data:
            signal_components.append(self.tape_data['volume_imbalance'])
            weights.append(1.0)  # Base weight
            
        # Transaction rate change component
        if 'rate_change' in self.tape_data:
            # Scale rate change to [-1, 1]
            rate_signal = np.clip(self.tape_data['rate_change'], -1.0, 1.0)
            
            # For transaction rate, we need to combine with direction
            # If no direction info, use as a signal magnitude modifier
            if 'volume_imbalance' in self.tape_data:
                # Increase signal strength if rate is increasing
                vol_imbalance = self.tape_data['volume_imbalance']
                rate_direction = np.sign(vol_imbalance) if vol_imbalance != 0 else 0
                rate_signal = rate_signal * rate_direction
                
            signal_components.append(rate_signal)
            weights.append(self.transaction_rate_weight)
            
        # Sweep imbalance component
        if 'sweep_imbalance' in self.tape_data:
            signal_components.append(self.tape_data['sweep_imbalance'])
            weights.append(0.7)  # Sweeps are important signals
            
        # Block trade imbalance component
        if 'block_buy_volume' in self.tape_data and 'block_sell_volume' in self.tape_data:
            block_buy = self.tape_data['block_buy_volume']
            block_sell = self.tape_data['block_sell_volume']
            total_block = block_buy + block_sell
            
            if total_block > 0:
                block_imbalance = (block_buy - block_sell) / total_block
                signal_components.append(block_imbalance)
                weights.append(0.8)  # Block trades are significant
                
        # Directional volume ratio for bar data
        if 'directional_volume_ratio' in self.tape_data:
            signal_components.append(self.tape_data['directional_volume_ratio'])
            weights.append(0.6)  # Lower weight for bar-based estimate
            
        # Calculate weighted signal
        if signal_components and weights:
            raw_signal = sum(s * w for s, w in zip(signal_components, weights)) / sum(weights)
            
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
        Predict tape reading signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate increasing buy-side aggression
          * Negative values indicate increasing sell-side aggression
          * Values near zero indicate balanced or inconclusive activity
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Tape Reading Agent" 