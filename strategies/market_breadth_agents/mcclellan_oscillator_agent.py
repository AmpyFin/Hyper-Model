"""
McClellan Oscillator Agent
~~~~~~~~~~~~~~~~~~~
Implements the McClellan Oscillator and Summation Index, popular breadth indicators
that use exponential moving averages of advances and declines.

Logic:
1. Calculate the McClellan Oscillator (19-day EMA minus 39-day EMA of net advances)
2. Track the McClellan Summation Index (running sum of the Oscillator)
3. Identify bullish and bearish divergences in the indicators
4. Generate signals based on extreme readings, zero-line crossovers, and divergences

Input: DataFrame containing market breadth data. Output ∈ [-1, +1] where:
* Positive values: Strong positive breadth momentum (bullish)
* Negative values: Strong negative breadth momentum (bearish)
* Values near zero: Neutral breadth momentum
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class McclellanOscillatorAgent:
    def __init__(
        self,
        lookback_period: int = 60,         # Period for oscillator analysis
        short_ema_period: int = 19,        # Short-term EMA period
        long_ema_period: int = 39,         # Long-term EMA period
        oversold_threshold: float = -70,   # Threshold for oversold conditions
        overbought_threshold: float = 70,  # Threshold for overbought conditions
        signal_smoothing: int = 3,         # Periods for signal smoothing
        use_summation_index: bool = True   # Whether to incorporate Summation Index
    ):
        self.lookback_period = lookback_period
        self.short_ema_period = short_ema_period
        self.long_ema_period = long_ema_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.signal_smoothing = signal_smoothing
        self.use_summation_index = use_summation_index
        self.latest_signal = 0.0
        self.signal_history = []
        self.mcclellan_data = {}
        
    def _calculate_mcclellan(self, df: pd.DataFrame) -> Dict:
        """Calculate McClellan Oscillator and Summation Index from market data"""
        result = {}
        
        # Check if we have advance-decline data
        has_ad_data = False
        
        # Check different column naming conventions
        if all(col in df.columns for col in ['advances', 'declines']):
            has_ad_data = True
            advances_col = 'advances'
            declines_col = 'declines'
        elif all(col in df.columns for col in ['advancing_issues', 'declining_issues']):
            has_ad_data = True
            advances_col = 'advancing_issues'
            declines_col = 'declining_issues'
            
        if has_ad_data:
            # Calculate McClellan metrics
            df_copy = df.copy()
            
            # Calculate net advances
            df_copy['net_advances'] = df_copy[advances_col] - df_copy[declines_col]
            
            # Calculate advance-decline ratio
            total_issues = df_copy[advances_col] + df_copy[declines_col]
            if 'unchanged' in df_copy.columns:
                total_issues += df_copy['unchanged']
            elif 'unchanged_issues' in df_copy.columns:
                total_issues += df_copy['unchanged_issues']
                
            df_copy['ad_ratio'] = df_copy[advances_col] / total_issues
            
            # Calculate EMAs for the oscillator
            if len(df_copy) >= max(self.short_ema_period, self.long_ema_period):
                df_copy['short_ema'] = df_copy['net_advances'].ewm(span=self.short_ema_period, adjust=False).mean()
                df_copy['long_ema'] = df_copy['net_advances'].ewm(span=self.long_ema_period, adjust=False).mean()
                
                # Calculate McClellan Oscillator
                df_copy['mcclellan_oscillator'] = df_copy['short_ema'] - df_copy['long_ema']
                
                # Calculate Summation Index (running sum of oscillator)
                if self.use_summation_index:
                    df_copy['summation_index'] = df_copy['mcclellan_oscillator'].cumsum()
                    
                # Store latest values
                result['latest_oscillator'] = df_copy['mcclellan_oscillator'].iloc[-1]
                if self.use_summation_index:
                    result['latest_summation'] = df_copy['summation_index'].iloc[-1]
                    
                # Calculate normalized oscillator (typical range is -100 to +100)
                normalized_osc = df_copy['mcclellan_oscillator'].iloc[-1] / 100
                result['normalized_oscillator'] = max(-1.0, min(1.0, normalized_osc))
                
                # Check for zero-line crossovers
                if len(df_copy) >= 2:
                    prev_osc = df_copy['mcclellan_oscillator'].iloc[-2]
                    curr_osc = df_copy['mcclellan_oscillator'].iloc[-1]
                    
                    if prev_osc < 0 and curr_osc > 0:
                        result['bullish_crossover'] = True
                    elif prev_osc > 0 and curr_osc < 0:
                        result['bearish_crossover'] = True
                        
                # Check for oscillator extremes
                result['is_overbought'] = curr_osc > self.overbought_threshold
                result['is_oversold'] = curr_osc < self.oversold_threshold
                
                # Calculate oscillator momentum
                if len(df_copy) >= 5:
                    osc_momentum = df_copy['mcclellan_oscillator'].iloc[-1] - df_copy['mcclellan_oscillator'].iloc[-5]
                    result['oscillator_momentum'] = osc_momentum
                    
                # Check for divergences
                if 'close' in df_copy.columns and len(df_copy) >= 20:
                    # Get price change
                    price_change = (df_copy['close'].iloc[-1] / df_copy['close'].iloc[-10]) - 1
                    
                    # Get oscillator change
                    osc_change = df_copy['mcclellan_oscillator'].iloc[-1] - df_copy['mcclellan_oscillator'].iloc[-10]
                    
                    # Check for divergences
                    if price_change > 0.02 and osc_change < -20:
                        result['bearish_divergence'] = True
                    elif price_change < -0.02 and osc_change > 20:
                        result['bullish_divergence'] = True
                        
                # Analyze Summation Index if enabled
                if self.use_summation_index and 'summation_index' in df_copy.columns:
                    # Calculate summation index rate of change
                    if len(df_copy) >= 5:
                        summ_roc = (df_copy['summation_index'].iloc[-1] - df_copy['summation_index'].iloc[-5])
                        result['summation_momentum'] = summ_roc
                        
                    # Check for trend confirmations/divergences between oscillator and summation
                    if 'oscillator_momentum' in result:
                        osc_mom = result['oscillator_momentum']
                        summ_mom = result.get('summation_momentum', 0)
                        
                        # If both moving in same direction, it's a stronger signal
                        if (osc_mom > 0 and summ_mom > 0) or (osc_mom < 0 and summ_mom < 0):
                            result['trend_confirmation'] = True
                        # If moving in opposite directions, it suggests a potential change
                        elif abs(osc_mom) > 10 and abs(summ_mom) > 100:
                            if osc_mom > 0 and summ_mom < 0:
                                result['momentum_divergence'] = 'bullish'
                            elif osc_mom < 0 and summ_mom > 0:
                                result['momentum_divergence'] = 'bearish'
        else:
            # No advance-decline data, try to create a proxy
            # This is less accurate but can give some signal when real data unavailable
            if 'close' in df.columns and len(df) >= max(self.short_ema_period, self.long_ema_period):
                df_copy = df.copy()
                
                # Use price changes as proxy for net advances
                df_copy['price_change'] = df_copy['close'].diff()
                df_copy['proxy_net_advances'] = np.where(df_copy['price_change'] > 0, 1, -1)
                
                # Calculate EMAs for the proxy oscillator
                df_copy['short_ema'] = df_copy['proxy_net_advances'].ewm(span=self.short_ema_period, adjust=False).mean()
                df_copy['long_ema'] = df_copy['proxy_net_advances'].ewm(span=self.long_ema_period, adjust=False).mean()
                
                # Calculate proxy oscillator
                df_copy['proxy_oscillator'] = df_copy['short_ema'] - df_copy['long_ema']
                
                # Scale proxy to typical oscillator range
                df_copy['scaled_proxy'] = df_copy['proxy_oscillator'] * 50
                
                # Store results
                result['proxy_oscillator'] = df_copy['scaled_proxy'].iloc[-1]
                
                # Calculate normalized proxy
                normalized_proxy = df_copy['proxy_oscillator'].iloc[-1] * 2  # Scale to [-1, +1] range
                result['normalized_proxy'] = max(-1.0, min(1.0, normalized_proxy))
                
                # Check for zero-line crossovers
                if len(df_copy) >= 2:
                    prev_proxy = df_copy['proxy_oscillator'].iloc[-2]
                    curr_proxy = df_copy['proxy_oscillator'].iloc[-1]
                    
                    if prev_proxy < 0 and curr_proxy > 0:
                        result['proxy_bullish_crossover'] = True
                    elif prev_proxy > 0 and curr_proxy < 0:
                        result['proxy_bearish_crossover'] = True
        
        return result
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to calculate McClellan indicators and generate signals
        """
        # Need enough bars for calculation
        if len(historical_df) < max(20, self.long_ema_period):
            self.latest_signal = 0.0
            return
            
        # Use lookback window if specified
        if self.lookback_period and len(historical_df) > self.lookback_period:
            df_subset = historical_df.iloc[-self.lookback_period:]
        else:
            df_subset = historical_df
            
        # Calculate McClellan metrics
        self.mcclellan_data = self._calculate_mcclellan(df_subset)
        
        # Generate signal components
        signal_components = []
        
        # Primary component: Normalized oscillator
        if 'normalized_oscillator' in self.mcclellan_data:
            signal_components.append(self.mcclellan_data['normalized_oscillator'])
        elif 'normalized_proxy' in self.mcclellan_data:
            signal_components.append(self.mcclellan_data['normalized_proxy'] * 0.7)  # Lower weight for proxy
            
        # Crossover signals
        if self.mcclellan_data.get('bullish_crossover', False):
            signal_components.append(0.7)  # Strong bullish
        elif self.mcclellan_data.get('proxy_bullish_crossover', False):
            signal_components.append(0.5)  # Moderate bullish (proxy)
            
        if self.mcclellan_data.get('bearish_crossover', False):
            signal_components.append(-0.7)  # Strong bearish
        elif self.mcclellan_data.get('proxy_bearish_crossover', False):
            signal_components.append(-0.5)  # Moderate bearish (proxy)
            
        # Overbought/oversold conditions
        if self.mcclellan_data.get('is_overbought', False):
            # Overbought can be a warning sign but depends on momentum
            if self.mcclellan_data.get('oscillator_momentum', 0) > 0:
                # Still rising, maintain moderate bullish
                signal_components.append(0.3)
            else:
                # Overbought and falling, potential bearish
                signal_components.append(-0.2)
                
        if self.mcclellan_data.get('is_oversold', False):
            # Oversold can be a potential bottom but depends on momentum
            if self.mcclellan_data.get('oscillator_momentum', 0) < 0:
                # Still falling, maintain moderate bearish
                signal_components.append(-0.3)
            else:
                # Oversold and rising, potential bullish
                signal_components.append(0.2)
                
        # Divergence signals
        if self.mcclellan_data.get('bullish_divergence', False):
            signal_components.append(0.6)  # Strong bullish
            
        if self.mcclellan_data.get('bearish_divergence', False):
            signal_components.append(-0.6)  # Strong bearish
            
        # Summation Index signals
        if self.mcclellan_data.get('trend_confirmation', False):
            # Strengthen the current signal
            if 'normalized_oscillator' in self.mcclellan_data:
                signal_components.append(self.mcclellan_data['normalized_oscillator'] * 0.5)
            elif 'normalized_proxy' in self.mcclellan_data:
                signal_components.append(self.mcclellan_data['normalized_proxy'] * 0.3)
                
        if self.mcclellan_data.get('momentum_divergence') == 'bullish':
            signal_components.append(0.4)  # Moderate bullish
            
        if self.mcclellan_data.get('momentum_divergence') == 'bearish':
            signal_components.append(-0.4)  # Moderate bearish
            
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
        Predict McClellan Oscillator signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate positive breadth momentum (bullish)
          * Negative values indicate negative breadth momentum (bearish)
          * Values near zero indicate neutral momentum
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "McClellan Oscillator Agent" 