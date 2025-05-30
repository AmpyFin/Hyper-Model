"""
Stocks Above Moving Average Agent
~~~~~~~~~~~~~~~~~~~~~
Tracks the percentage of stocks trading above key moving averages to gauge market
health and identify overbought/oversold conditions and trend changes.

Logic:
1. Monitor the percentage of stocks above 50-day and 200-day moving averages
2. Identify extreme readings that often mark market turning points
3. Track changes in these percentages to detect shifts in market breadth
4. Generate signals based on bullish and bearish breadth thresholds

Input: DataFrame containing market breadth data. Output ∈ [-1, +1] where:
* Positive values: High percentage of stocks above key moving averages (bullish)
* Negative values: Low percentage of stocks above key moving averages (bearish)
* Values near zero: Average percentage or mixed readings
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class StocksAboveMaAgent:
    def __init__(
        self,
        lookback_period: int = 60,         # Period for breadth analysis
        short_ma_period: int = 50,         # Short-term moving average period
        long_ma_period: int = 200,         # Long-term moving average period
        overbought_level: float = 0.8,     # Threshold for overbought (>80% above MA)
        oversold_level: float = 0.2,       # Threshold for oversold (<20% above MA)
        signal_smoothing: int = 3,         # Periods for signal smoothing
        short_ma_weight: float = 0.6,      # Weight given to short-term MA vs long-term
        fallback_mode: bool = True         # Use single asset MA data as fallback
    ):
        self.lookback_period = lookback_period
        self.short_ma_period = short_ma_period
        self.long_ma_period = long_ma_period
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
        self.signal_smoothing = signal_smoothing
        self.short_ma_weight = short_ma_weight
        self.long_ma_weight = 1.0 - short_ma_weight
        self.fallback_mode = fallback_mode
        self.latest_signal = 0.0
        self.signal_history = []
        self.ma_data = {}
        
    def _calculate_stocks_above_ma(self, df: pd.DataFrame) -> Dict:
        """Calculate percentage of stocks above moving averages"""
        result = {}
        
        # Check if we have percentage data directly
        has_pct_data = False
        
        # Check for short-term MA percentage data
        short_ma_col = f'pct_above_{self.short_ma_period}ma'
        alt_short_ma_col = f'pct_above_{self.short_ma_period}d_ma'
        
        if short_ma_col in df.columns:
            has_pct_data = True
            short_col = short_ma_col
        elif alt_short_ma_col in df.columns:
            has_pct_data = True
            short_col = alt_short_ma_col
        elif f'pct_above_50ma' in df.columns:
            # Use 50-day as fallback if exact period not available
            has_pct_data = True
            short_col = 'pct_above_50ma'
            
        # Check for long-term MA percentage data
        long_ma_col = f'pct_above_{self.long_ma_period}ma'
        alt_long_ma_col = f'pct_above_{self.long_ma_period}d_ma'
        
        if long_ma_col in df.columns:
            has_pct_data = True
            long_col = long_ma_col
        elif alt_long_ma_col in df.columns:
            has_pct_data = True
            long_col = alt_long_ma_col
        elif f'pct_above_200ma' in df.columns:
            # Use 200-day as fallback if exact period not available
            has_pct_data = True
            long_col = 'pct_above_200ma'
            
        if has_pct_data:
            # We have direct percentage data
            df_copy = df.copy()
            
            # Store latest values
            if short_col in df_copy.columns:
                result['pct_above_short_ma'] = df_copy[short_col].iloc[-1]
                
                # Calculate short-term momentum
                if len(df_copy) >= 5:
                    result['short_ma_momentum'] = df_copy[short_col].iloc[-1] - df_copy[short_col].iloc[-5]
                    
            if long_col in df_copy.columns:
                result['pct_above_long_ma'] = df_copy[long_col].iloc[-1]
                
                # Calculate long-term momentum
                if len(df_copy) >= 10:
                    result['long_ma_momentum'] = df_copy[long_col].iloc[-1] - df_copy[long_col].iloc[-10]
                    
            # Check for extreme readings
            if 'pct_above_short_ma' in result:
                result['short_ma_overbought'] = result['pct_above_short_ma'] > self.overbought_level
                result['short_ma_oversold'] = result['pct_above_short_ma'] < self.oversold_level
                
            if 'pct_above_long_ma' in result:
                result['long_ma_overbought'] = result['pct_above_long_ma'] > self.overbought_level
                result['long_ma_oversold'] = result['pct_above_long_ma'] < self.oversold_level
                
            # Check for crossovers between percentages
            if 'pct_above_short_ma' in result and 'pct_above_long_ma' in result:
                if len(df_copy) >= 2:
                    prev_short = df_copy[short_col].iloc[-2]
                    curr_short = df_copy[short_col].iloc[-1]
                    prev_long = df_copy[long_col].iloc[-2]
                    curr_long = df_copy[long_col].iloc[-1]
                    
                    # Check for crossovers
                    if prev_short < prev_long and curr_short > curr_long:
                        result['bullish_crossover'] = True
                    elif prev_short > prev_long and curr_short < curr_long:
                        result['bearish_crossover'] = True
                        
                # Calculate spread between short and long
                result['ma_spread'] = result['pct_above_short_ma'] - result['pct_above_long_ma']
                
            # Check for divergences
            if 'close' in df_copy.columns and len(df_copy) >= 20:
                # Calculate price change
                price_change = (df_copy['close'].iloc[-1] / df_copy['close'].iloc[-20]) - 1
                
                # Breadth divergences
                if 'pct_above_short_ma' in result:
                    breadth_change = result['pct_above_short_ma'] - df_copy[short_col].iloc[-20]
                    
                    # Bearish divergence: price up, breadth down
                    if price_change > 0.03 and breadth_change < -0.1:
                        result['bearish_divergence'] = True
                        
                    # Bullish divergence: price down, breadth up
                    if price_change < -0.03 and breadth_change > 0.1:
                        result['bullish_divergence'] = True
                        
        elif self.fallback_mode and 'close' in df.columns:
            # No percentage data, use single asset as proxy
            df_copy = df.copy()
            
            # Calculate moving averages
            if len(df_copy) >= self.long_ma_period:
                df_copy[f'ma_{self.short_ma_period}'] = df_copy['close'].rolling(window=self.short_ma_period).mean()
                df_copy[f'ma_{self.long_ma_period}'] = df_copy['close'].rolling(window=self.long_ma_period).mean()
                
                # Create binary indicators for above/below MA
                above_short = df_copy['close'] > df_copy[f'ma_{self.short_ma_period}']
                above_long = df_copy['close'] > df_copy[f'ma_{self.long_ma_period}']
                
                # Store latest values
                result['above_short_ma'] = above_short.iloc[-1]
                result['above_long_ma'] = above_long.iloc[-1]
                
                # Calculate "percentage" using a rolling window
                window_size = min(20, len(df_copy))
                result['pct_days_above_short_ma'] = above_short.rolling(window=window_size).mean().iloc[-1]
                result['pct_days_above_long_ma'] = above_long.rolling(window=window_size).mean().iloc[-1]
                
                # Calculate MA spread (distance from price to MA)
                short_ma_spread = (df_copy['close'].iloc[-1] / df_copy[f'ma_{self.short_ma_period}'].iloc[-1]) - 1
                long_ma_spread = (df_copy['close'].iloc[-1] / df_copy[f'ma_{self.long_ma_period}'].iloc[-1]) - 1
                
                result['short_ma_spread'] = short_ma_spread
                result['long_ma_spread'] = long_ma_spread
                
                # Normalize spreads to a signal
                result['short_ma_signal'] = np.clip(short_ma_spread * 20, -1.0, 1.0)
                result['long_ma_signal'] = np.clip(long_ma_spread * 10, -1.0, 1.0)
                
                # Check for recent crossovers
                if len(df_copy) >= 5:
                    recent_cross_up = not above_short.iloc[-5] and above_short.iloc[-1]
                    recent_cross_down = above_short.iloc[-5] and not above_short.iloc[-1]
                    
                    result['recent_cross_up'] = recent_cross_up
                    result['recent_cross_down'] = recent_cross_down
        
        return result
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to calculate stocks above MA metrics and generate signals
        """
        # Need enough bars for calculation
        if len(historical_df) < self.long_ma_period:
            self.latest_signal = 0.0
            return
            
        # Use lookback window if specified, but ensure we have enough data for MA calculation
        if self.lookback_period and len(historical_df) > self.lookback_period:
            # For fallback mode, we need at least long_ma_period rows
            min_required = max(self.lookback_period, self.long_ma_period) if self.fallback_mode else self.lookback_period
            df_subset = historical_df.iloc[-min_required:]
        else:
            df_subset = historical_df
            
        # Calculate moving average metrics
        self.ma_data = self._calculate_stocks_above_ma(df_subset)
        
        # Generate signal components
        signal_components = []
        
        # Primary component: Percentage above MAs
        if 'pct_above_short_ma' in self.ma_data:
            # Convert from 0-1 to -1 to +1
            short_ma_signal = (self.ma_data['pct_above_short_ma'] - 0.5) * 2
            signal_components.append(short_ma_signal * self.short_ma_weight)
            
        if 'pct_above_long_ma' in self.ma_data:
            # Convert from 0-1 to -1 to +1
            long_ma_signal = (self.ma_data['pct_above_long_ma'] - 0.5) * 2
            signal_components.append(long_ma_signal * self.long_ma_weight)
            
        # Fallback proxy signals
        if 'short_ma_signal' in self.ma_data:
            signal_components.append(self.ma_data['short_ma_signal'] * self.short_ma_weight * 0.8)  # Lower weight for proxy
            
        if 'long_ma_signal' in self.ma_data:
            signal_components.append(self.ma_data['long_ma_signal'] * self.long_ma_weight * 0.8)  # Lower weight for proxy
            
        # Momentum signals
        if 'short_ma_momentum' in self.ma_data:
            # Scale momentum to reasonable range
            momentum_signal = np.clip(self.ma_data['short_ma_momentum'] * 5, -1.0, 1.0)
            signal_components.append(momentum_signal * 0.7)  # Momentum has lower weight
            
        if 'long_ma_momentum' in self.ma_data:
            # Scale momentum to reasonable range
            momentum_signal = np.clip(self.ma_data['long_ma_momentum'] * 3, -1.0, 1.0)
            signal_components.append(momentum_signal * 0.5)  # Long-term momentum has even lower weight
            
        # Extreme readings
        if self.ma_data.get('short_ma_overbought', False):
            # Overbought can be both bullish (strength) or a warning sign
            # Look at momentum to determine
            if self.ma_data.get('short_ma_momentum', 0) > 0:
                # Still rising, positive
                signal_components.append(0.7)
            else:
                # Overbought and falling, potential warning
                signal_components.append(0.2)
                
        if self.ma_data.get('short_ma_oversold', False):
            # Oversold can be both bearish (weakness) or a bottoming sign
            # Look at momentum to determine
            if self.ma_data.get('short_ma_momentum', 0) < 0:
                # Still falling, negative
                signal_components.append(-0.7)
            else:
                # Oversold but rising, potential bottom
                signal_components.append(-0.2)
                
        # Crossover signals
        if self.ma_data.get('bullish_crossover', False):
            signal_components.append(0.8)  # Strong bullish
            
        if self.ma_data.get('bearish_crossover', False):
            signal_components.append(-0.8)  # Strong bearish
            
        # Recent crossovers from fallback mode
        if self.ma_data.get('recent_cross_up', False):
            signal_components.append(0.6)  # Moderate bullish
            
        if self.ma_data.get('recent_cross_down', False):
            signal_components.append(-0.6)  # Moderate bearish
            
        # Divergence signals
        if self.ma_data.get('bullish_divergence', False):
            signal_components.append(0.7)  # Strong bullish
            
        if self.ma_data.get('bearish_divergence', False):
            signal_components.append(-0.7)  # Strong bearish
            
        # Calculate final signal
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
        Predict stocks above MA signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate high percentage above MAs (bullish)
          * Negative values indicate low percentage above MAs (bearish)
          * Values near zero indicate average or mixed readings
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Stocks Above MA Agent" 