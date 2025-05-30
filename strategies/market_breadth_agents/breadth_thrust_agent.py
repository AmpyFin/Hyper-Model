"""
Breadth Thrust Agent
~~~~~~~~~~~~~~~
Monitors for rapid expansions in market breadth that often occur at the beginning
of significant market rallies, based on Martin Zweig's Breadth Thrust indicator.

Logic:
1. Calculate the 10-day ratio of advancing issues to total issues
2. Identify breadth thrust signals (rapid expansion from oversold to overbought)
3. Monitor the acceleration of breadth metrics
4. Generate signals based on breadth thrust conditions and follow-through

Input: DataFrame containing market breadth data. Output ∈ [-1, +1] where:
* Positive values: Strong breadth expansion detected (bullish)
* Negative values: Breadth contraction or deterioration (bearish)
* Values near zero: No significant breadth thrust detected
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class BreadthThrustAgent:
    def __init__(
        self,
        lookback_period: int = 50,         # Period for breadth analysis
        thrust_period: int = 10,           # Breadth thrust calculation period
        oversold_threshold: float = 0.4,   # Threshold for oversold conditions
        thrust_threshold: float = 0.615,   # Threshold for breadth thrust signal
        signal_decay: float = 0.9,         # Decay factor for signal over time
        normalize_output: bool = True      # Whether to normalize output to [-1, +1]
    ):
        self.lookback_period = lookback_period
        self.thrust_period = thrust_period
        self.oversold_threshold = oversold_threshold
        self.thrust_threshold = thrust_threshold
        self.signal_decay = signal_decay
        self.normalize_output = normalize_output
        self.latest_signal = 0.0
        self.thrust_detected = False
        self.days_since_thrust = 0
        self.thrust_data = {}
        
    def _calculate_thrust_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate breadth thrust metrics from market data"""
        result = {}
        
        # Check if we have the appropriate data columns
        has_breadth_data = False
        
        # Check for advances/declines data first
        if all(col in df.columns for col in ['advances', 'declines']):
            # Standard advances/declines columns
            has_breadth_data = True
            advances_col = 'advances'
            
            # Calculate total issues
            if 'total_issues' in df.columns:
                total_issues_col = 'total_issues'
            else:
                # Create total issues
                df = df.copy()
                df['total_issues'] = df['advances'] + df['declines']
                if 'unchanged' in df.columns:
                    df['total_issues'] += df['unchanged']
                total_issues_col = 'total_issues'
        
        # Alternative column naming
        elif all(col in df.columns for col in ['advancing_issues', 'declining_issues']):
            has_breadth_data = True
            advances_col = 'advancing_issues'
            
            if 'total_issues' in df.columns:
                total_issues_col = 'total_issues'
            else:
                df = df.copy()
                df['total_issues'] = df['advancing_issues'] + df['declining_issues']
                if 'unchanged_issues' in df.columns:
                    df['total_issues'] += df['unchanged_issues']
                total_issues_col = 'total_issues'
        
        if has_breadth_data:
            # Calculate breadth thrust metrics
            df_copy = df.copy()
            
            # Calculate advance ratio
            df_copy['advance_ratio'] = df_copy[advances_col] / df_copy[total_issues_col]
            
            # Calculate thrust indicator (10-day moving average of advance ratio)
            df_copy['thrust_indicator'] = df_copy['advance_ratio'].rolling(window=self.thrust_period).mean()
            
            # Calculate 5-day rate of change to measure acceleration
            if len(df_copy) >= self.thrust_period + 5:
                df_copy['thrust_5d_change'] = df_copy['thrust_indicator'].diff(5)
                
            # Store latest values
            result['latest_advance_ratio'] = df_copy['advance_ratio'].iloc[-1]
            if not pd.isna(df_copy['thrust_indicator'].iloc[-1]):
                result['latest_thrust_indicator'] = df_copy['thrust_indicator'].iloc[-1]
            
            if 'thrust_5d_change' in df_copy.columns and not pd.isna(df_copy['thrust_5d_change'].iloc[-1]):
                result['thrust_acceleration'] = df_copy['thrust_5d_change'].iloc[-1]
                
            # Check for breadth thrust conditions
            # Condition 1: Indicator was below oversold threshold recently
            was_oversold = False
            if len(df_copy) >= 20:  # Check the last 20 days
                recent_indicators = df_copy['thrust_indicator'].iloc[-20:].dropna()
                min_indicator = recent_indicators.min() if not recent_indicators.empty else 1.0
                was_oversold = min_indicator < self.oversold_threshold
                
            result['was_oversold'] = was_oversold
            
            # Condition 2: Current reading is above thrust threshold
            is_thrust = False
            if 'latest_thrust_indicator' in result:
                is_thrust = result['latest_thrust_indicator'] > self.thrust_threshold
                
            result['is_thrust'] = is_thrust
            
            # Official breadth thrust signal
            result['thrust_signal'] = was_oversold and is_thrust
            
            # In case we don't have a thrust signal, still provide a normalized indicator
            if 'latest_thrust_indicator' in result:
                # Normalize around the typical range (0.3 to 0.7)
                normalized = (result['latest_thrust_indicator'] - 0.5) * 5  # Scale to [-1, +1]
                result['normalized_indicator'] = max(-1.0, min(1.0, normalized))
                
            # Provide additional context about recent breadth performance
            if len(df_copy) >= 5:
                # Calculate 5-day sum of advances vs declines
                recent_advance_sum = df_copy[advances_col].iloc[-5:].sum()
                recent_decline_sum = df_copy['declines'].iloc[-5:].sum() if 'declines' in df_copy.columns else \
                                    df_copy['declining_issues'].iloc[-5:].sum()
                
                # Calculate 5-day breadth momentum
                total_issues_5d = recent_advance_sum + recent_decline_sum
                if total_issues_5d > 0:
                    result['recent_breadth_momentum'] = (recent_advance_sum - recent_decline_sum) / total_issues_5d
                    
        else:
            # No breadth data, try to use OHLCV data as a proxy
            df_copy = df.copy()
            
            if all(col in df_copy.columns for col in ['open', 'high', 'low', 'close']):
                # Use price momentum as proxy for breadth
                # Calculate percentage of days closing up in the last N days
                if len(df_copy) >= self.thrust_period:
                    df_copy['day_change'] = df_copy['close'].diff()
                    up_days = (df_copy['day_change'] > 0).rolling(window=self.thrust_period).sum()
                    
                    # Calculate proxy thrust indicator (percentage of up days)
                    df_copy['proxy_thrust'] = up_days / self.thrust_period
                    
                    # Store results
                    result['proxy_thrust'] = df_copy['proxy_thrust'].iloc[-1]
                    
                    # Check for thrust-like conditions in the proxy
                    # Was the market mostly down recently?
                    if len(df_copy) >= 20:
                        min_up_ratio = df_copy['proxy_thrust'].iloc[-20:].min()
                        was_low = min_up_ratio < 0.3  # Less than 30% up days
                        
                        # Is the market mostly up now?
                        is_high = df_copy['proxy_thrust'].iloc[-1] > 0.7  # More than 70% up days
                        
                        result['proxy_thrust_signal'] = was_low and is_high
                        
                    # Normalize proxy to [-1, +1]
                    proxy_normalized = (df_copy['proxy_thrust'].iloc[-1] - 0.5) * 2
                    result['normalized_proxy'] = proxy_normalized
        
        return result
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to calculate breadth thrust metrics and generate signals
        """
        # Need enough bars for calculation
        if len(historical_df) < max(20, self.thrust_period):
            self.latest_signal = 0.0
            return
            
        # Use lookback window if specified
        if self.lookback_period and len(historical_df) > self.lookback_period:
            df_subset = historical_df.iloc[-self.lookback_period:]
        else:
            df_subset = historical_df
            
        # Calculate thrust metrics
        self.thrust_data = self._calculate_thrust_metrics(df_subset)
        
        # Generate signal based on thrust data
        signal = 0.0
        
        # Check if a thrust signal was triggered
        if self.thrust_data.get('thrust_signal', False):
            # New thrust signal
            self.thrust_detected = True
            self.days_since_thrust = 0
            signal = 1.0
        elif self.thrust_data.get('proxy_thrust_signal', False):
            # Proxy signal (less strong)
            self.thrust_detected = True
            self.days_since_thrust = 0
            signal = 0.8
        elif self.thrust_detected:
            # Existing thrust signal, apply decay factor
            self.days_since_thrust += 1
            
            # Thrust signals typically remain relevant for about 30 days
            if self.days_since_thrust < 30:
                signal = 1.0 * (self.signal_decay ** self.days_since_thrust)
            else:
                # Reset after 30 days
                self.thrust_detected = False
                self.days_since_thrust = 0
                signal = 0.0
        else:
            # No thrust signal, use normalized indicator
            if 'normalized_indicator' in self.thrust_data:
                signal = self.thrust_data['normalized_indicator']
            elif 'normalized_proxy' in self.thrust_data:
                signal = self.thrust_data['normalized_proxy']
                
            # Adjust based on recent breadth momentum
            if 'recent_breadth_momentum' in self.thrust_data:
                momentum = self.thrust_data['recent_breadth_momentum']
                
                # Strong momentum strengthens signal
                if abs(momentum) > 0.1:
                    # If momentum and indicator agree, strengthen
                    if (momentum > 0 and signal > 0) or (momentum < 0 and signal < 0):
                        signal = signal * 1.2
                        
                    # If they disagree, signal may be changing direction
                    else:
                        signal = signal * 0.5 + momentum * 0.3
                        
        # Ensure signal is in [-1, +1] range
        self.latest_signal = max(-1.0, min(1.0, signal))
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict breadth thrust signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate a breadth thrust or strong breadth (bullish)
          * Negative values indicate deteriorating breadth (bearish)
          * Values near zero indicate no significant breadth conditions
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Breadth Thrust Agent" 