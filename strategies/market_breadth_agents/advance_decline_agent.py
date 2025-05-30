"""
Advance-Decline Agent
~~~~~~~~~~~~~~~~~
Tracks the ratio and net difference between advancing and declining stocks in a market
or index to assess overall market breadth and identify potential trend reversals.

Logic:
1. Calculate the number and percentage of advancing vs declining stocks
2. Track the advance-decline line (cumulative sum of net advances)
3. Generate signals based on changes in breadth momentum
4. Detect divergences between breadth indicators and price movement

Input: DataFrame containing market breadth data. Output ∈ [-1, +1] where:
* Positive values: Breadth expanding positively (bullish)
* Negative values: Breadth deteriorating (bearish)
* Values near zero: Neutral or mixed breadth conditions
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class AdvanceDeclineAgent:
    def __init__(
        self,
        lookback_period: int = 50,        # Period for breadth analysis
        short_ma_period: int = 5,         # Short-term moving average period
        long_ma_period: int = 20,         # Long-term moving average period
        signal_threshold: float = 0.3,    # Threshold for signal generation
        signal_smoothing: int = 3,        # Periods for signal smoothing
        breadth_fallback: bool = True     # Use single-asset fallback when index data unavailable
    ):
        self.lookback_period = lookback_period
        self.short_ma_period = short_ma_period
        self.long_ma_period = long_ma_period
        self.signal_threshold = signal_threshold
        self.signal_smoothing = signal_smoothing
        self.breadth_fallback = breadth_fallback
        self.latest_signal = 0.0
        self.signal_history = []
        self.breadth_data = {}
        
    def _calculate_breadth_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate advance-decline metrics from market data"""
        result = {}
        
        # Check if we have advance-decline data columns
        has_breadth_data = all(col in df.columns for col in ['advances', 'declines'])
        
        if has_breadth_data:
            # Calculate advance-decline metrics
            df_copy = df.copy()
            
            # Calculate basic metrics
            df_copy['net_advances'] = df_copy['advances'] - df_copy['declines']
            df_copy['advance_decline_ratio'] = df_copy['advances'] / df_copy['declines']
            df_copy['ad_line'] = df_copy['net_advances'].cumsum()
            
            # Total issues if available
            if 'total_issues' in df_copy.columns:
                df_copy['advance_pct'] = df_copy['advances'] / df_copy['total_issues']
                df_copy['decline_pct'] = df_copy['declines'] / df_copy['total_issues']
            else:
                total_issues = df_copy['advances'] + df_copy['declines']
                df_copy['advance_pct'] = df_copy['advances'] / total_issues
                df_copy['decline_pct'] = df_copy['declines'] / total_issues
                
            # Calculate moving averages of advance percentage
            df_copy[f'advance_pct_{self.short_ma_period}ma'] = df_copy['advance_pct'].rolling(
                window=self.short_ma_period).mean()
            df_copy[f'advance_pct_{self.long_ma_period}ma'] = df_copy['advance_pct'].rolling(
                window=self.long_ma_period).mean()
                
            # Calculate advance-decline oscillator
            df_copy['ad_oscillator'] = df_copy[f'advance_pct_{self.short_ma_period}ma'] - \
                                      df_copy[f'advance_pct_{self.long_ma_period}ma']
            
            # Store latest values
            result['latest_net_advances'] = df_copy['net_advances'].iloc[-1]
            result['latest_ad_ratio'] = df_copy['advance_decline_ratio'].iloc[-1]
            result['latest_ad_line'] = df_copy['ad_line'].iloc[-1]
            result['latest_ad_oscillator'] = df_copy['ad_oscillator'].iloc[-1]
            
            # Calculate rate of change for AD Line
            if len(df_copy) >= 5:
                ad_line_5d_change = df_copy['ad_line'].iloc[-1] - df_copy['ad_line'].iloc[-5]
                result['ad_line_5d_change'] = ad_line_5d_change
                
            # Calculate normalized metrics for signal generation
            # Normalize the oscillator to a reasonable range
            recent_oscillator = df_copy['ad_oscillator'].dropna()
            if len(recent_oscillator) > 0:
                oscillator_std = recent_oscillator.std()
                if oscillator_std > 0:
                    result['normalized_oscillator'] = df_copy['ad_oscillator'].iloc[-1] / (oscillator_std * 2)
                    result['normalized_oscillator'] = max(-1.0, min(1.0, result['normalized_oscillator']))
                
            # Calculate breadth momentum
            if len(df_copy) >= 10:
                breadth_momentum = (df_copy['advance_pct'].iloc[-5:].mean() - 
                                   df_copy['advance_pct'].iloc[-10:-5].mean())
                result['breadth_momentum'] = breadth_momentum
                
                # Normalize momentum
                result['normalized_momentum'] = max(-1.0, min(1.0, breadth_momentum * 5))
                
            # Detect breadth divergences
            if 'close' in df_copy.columns and len(df_copy) >= 20:
                # Price momentum
                price_change = (df_copy['close'].iloc[-1] / df_copy['close'].iloc[-20]) - 1
                
                # AD line momentum
                ad_change = df_copy['ad_line'].iloc[-1] - df_copy['ad_line'].iloc[-20]
                
                # Check for divergence (price up, breadth down or vice versa)
                if price_change > 0 and ad_change < 0:
                    result['breadth_divergence'] = -1  # Bearish divergence
                elif price_change < 0 and ad_change > 0:
                    result['breadth_divergence'] = 1   # Bullish divergence
                else:
                    result['breadth_divergence'] = 0   # No divergence
            
        elif self.breadth_fallback:
            # Use individual asset data as a fallback
            # This is a crude approximation using a single asset's data
            df_copy = df.copy()
            
            # Create a breadth proxy based on price movement relative to moving averages
            if 'close' in df_copy.columns:
                # Calculate moving averages
                df_copy['ma_short'] = df_copy['close'].rolling(window=self.short_ma_period).mean()
                df_copy['ma_long'] = df_copy['close'].rolling(window=self.long_ma_period).mean()
                
                # Use relationship to MAs as breadth proxy
                df_copy['above_ma_short'] = df_copy['close'] > df_copy['ma_short']
                df_copy['above_ma_long'] = df_copy['close'] > df_copy['ma_long']
                
                # Calculate proxy oscillator
                df_copy['ma_diff'] = (df_copy['ma_short'] / df_copy['ma_long']) - 1
                
                # Store results
                result['proxy_ma_diff'] = df_copy['ma_diff'].iloc[-1]
                
                # Normalize to signal range
                result['normalized_proxy'] = max(-1.0, min(1.0, result['proxy_ma_diff'] * 10))
                
                # Calculate price momentum as proxy for breadth momentum
                if len(df_copy) >= 10:
                    momentum = (df_copy['close'].iloc[-1] / df_copy['close'].iloc[-10]) - 1
                    result['price_momentum'] = momentum
                    
                    # Normalize to signal range
                    result['normalized_momentum'] = max(-1.0, min(1.0, momentum * 5))
        
        return result
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to calculate advance-decline metrics and generate signals
        """
        # Need enough bars for calculation
        if len(historical_df) < max(10, self.long_ma_period):
            self.latest_signal = 0.0
            return
            
        # Use lookback window if specified
        if self.lookback_period and len(historical_df) > self.lookback_period:
            df_subset = historical_df.iloc[-self.lookback_period:]
        else:
            df_subset = historical_df
            
        # Calculate breadth metrics
        self.breadth_data = self._calculate_breadth_metrics(df_subset)
        
        # Generate signal components
        signal_components = []
        
        # Primary component: AD Oscillator
        if 'normalized_oscillator' in self.breadth_data:
            signal_components.append(self.breadth_data['normalized_oscillator'])
            
        # Secondary component: Breadth Momentum
        if 'normalized_momentum' in self.breadth_data:
            signal_components.append(self.breadth_data['normalized_momentum'])
            
        # Divergence component (if detected)
        if 'breadth_divergence' in self.breadth_data and self.breadth_data['breadth_divergence'] != 0:
            signal_components.append(self.breadth_data['breadth_divergence'] * 0.7)
            
        # Fallback proxy components
        if 'normalized_proxy' in self.breadth_data:
            signal_components.append(self.breadth_data['normalized_proxy'])
            
        # Calculate combined signal
        if signal_components:
            raw_signal = sum(signal_components) / len(signal_components)
            
            # Apply thresholding for noise reduction
            if abs(raw_signal) < self.signal_threshold:
                raw_signal *= (abs(raw_signal) / self.signal_threshold)
                
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
        Predict advance-decline signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate positive breadth (bullish)
          * Negative values indicate negative breadth (bearish)
          * Values near zero indicate neutral breadth conditions
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Advance-Decline Agent" 