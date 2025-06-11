"""
New Highs-Lows Agent
~~~~~~~~~~~~~~~~
Analyzes the number of stocks making new highs versus new lows to assess market strength
and identify potential market reversals or confirmations.

Logic:
1. Track the number and ratio of new highs to new lows in the market
2. Calculate the High-Low Index (new highs - new lows) / total issues
3. Identify extremes in new highs or lows that often mark exhaustion points
4. Detect divergences between price action and new high/low data

Input: DataFrame containing market breadth data. Output ∈ [-1, +1] where:
* Positive values: More new highs than lows (bullish)
* Negative values: More new lows than highs (bearish)
* Values near zero: Balanced new highs and lows
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class NewHighsLowsAgent:
    def __init__(
        self,
        lookback_period: int = 50,           # Period for breadth analysis
        hl_smoothing_period: int = 10,       # Smoothing period for high-low index
        extreme_threshold: float = 0.1,      # Threshold for extreme readings (% of total issues)
        signal_smoothing: int = 3,           # Periods for signal smoothing
        use_52week: bool = True,             # Whether to use 52-week highs/lows (vs daily)
        reversal_detection: bool = True      # Whether to detect potential reversals
    ):
        self.lookback_period = lookback_period
        self.hl_smoothing_period = hl_smoothing_period
        self.extreme_threshold = extreme_threshold
        self.signal_smoothing = signal_smoothing
        self.use_52week = use_52week
        self.reversal_detection = reversal_detection
        self.latest_signal = 0.0
        self.signal_history = []
        self.hl_data = {}
        
    def _calculate_hl_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate new highs/lows metrics from market data"""
        result = {}
        
        # Check if we have the appropriate data columns
        has_hl_data = False
        
        # Check for 52-week high/low data first if that's our preference
        if self.use_52week:
            if all(col in df.columns for col in ['new_52wk_highs', 'new_52wk_lows']):
                high_col = 'new_52wk_highs'
                low_col = 'new_52wk_lows'
                has_hl_data = True
            elif all(col in df.columns for col in ['new_highs_52w', 'new_lows_52w']):
                high_col = 'new_highs_52w'
                low_col = 'new_lows_52w'
                has_hl_data = True
                
        # Fallback to daily highs/lows if needed
        if not has_hl_data:
            if all(col in df.columns for col in ['new_highs', 'new_lows']):
                high_col = 'new_highs'
                low_col = 'new_lows'
                has_hl_data = True
                
        if has_hl_data:
            # Calculate high-low metrics
            df_copy = df.copy()
            
            # Basic metrics
            df_copy['hl_diff'] = df_copy[high_col] - df_copy[low_col]
            df_copy['hl_ratio'] = df_copy[high_col] / df_copy[low_col].replace(0, 0.5)  # Avoid division by zero
            
            # Calculate as percentage of total issues if available
            if 'total_issues' in df_copy.columns:
                total_issues = df_copy['total_issues']
            else:
                # Approximate by summing highs and lows and other issues if available
                if 'unchanged' in df_copy.columns:
                    total_issues = df_copy[high_col] + df_copy[low_col] + df_copy['unchanged']
                else:
                    # Just use highs + lows as denominator (less accurate)
                    total_issues = df_copy[high_col] + df_copy[low_col]
                    
            df_copy['hl_index'] = df_copy['hl_diff'] / total_issues
            
            # Calculate smoothed high-low index
            df_copy['hl_index_smooth'] = df_copy['hl_index'].rolling(window=self.hl_smoothing_period).mean()
            
            # Store latest values
            result['latest_hl_diff'] = df_copy['hl_diff'].iloc[-1]
            result['latest_hl_ratio'] = df_copy['hl_ratio'].iloc[-1]
            result['latest_hl_index'] = df_copy['hl_index'].iloc[-1]
            
            if not pd.isna(df_copy['hl_index_smooth'].iloc[-1]):
                result['latest_hl_index_smooth'] = df_copy['hl_index_smooth'].iloc[-1]
            
            # Calculate percentage of total issues making new highs/lows
            result['pct_new_highs'] = df_copy[high_col].iloc[-1] / total_issues.iloc[-1]
            result['pct_new_lows'] = df_copy[low_col].iloc[-1] / total_issues.iloc[-1]
            
            # Check for extreme readings
            result['extreme_highs'] = result['pct_new_highs'] > self.extreme_threshold
            result['extreme_lows'] = result['pct_new_lows'] > self.extreme_threshold
            
            # Calculate rate of change
            if len(df_copy) >= 5:
                if not pd.isna(df_copy['hl_index_smooth'].iloc[-5]):
                    result['hl_index_5d_change'] = (df_copy['hl_index_smooth'].iloc[-1] - 
                                                  df_copy['hl_index_smooth'].iloc[-5])
                                                  
            # Detect potential reversals if enabled
            if self.reversal_detection and len(df_copy) >= 10:
                # Look for a change in direction of the smoothed high-low index
                recent_hl = df_copy['hl_index_smooth'].dropna().iloc[-10:]
                
                if len(recent_hl) >= 5:
                    # Calculate recent slope
                    recent_hl_vals = recent_hl.values
                    recent_x = np.arange(len(recent_hl_vals))
                    recent_slope, _ = np.polyfit(recent_x, recent_hl_vals, 1)
                    
                    # Calculate previous slope
                    prev_hl = df_copy['hl_index_smooth'].dropna().iloc[-15:-5]
                    if len(prev_hl) >= 5:
                        prev_hl_vals = prev_hl.values
                        prev_x = np.arange(len(prev_hl_vals))
                        prev_slope, _ = np.polyfit(prev_x, prev_hl_vals, 1)
                        
                        # Check for slope change
                        if (prev_slope < 0 and recent_slope > 0):
                            # Changing from down to up = bullish reversal
                            result['reversal_signal'] = 1
                        elif (prev_slope > 0 and recent_slope < 0):
                            # Changing from up to down = bearish reversal
                            result['reversal_signal'] = -1
                            
            # Detect divergences
            if 'close' in df_copy.columns and len(df_copy) >= 20:
                # Price momentum
                price_change = (df_copy['close'].iloc[-1] / df_copy['close'].iloc[-20]) - 1
                
                # High-Low index momentum
                if 'hl_index_smooth' in df_copy.columns:
                    hl_values = df_copy['hl_index_smooth'].dropna()
                    if len(hl_values) >= 20:
                        hl_change = hl_values.iloc[-1] - hl_values.iloc[-20]
                        
                        # Check for divergence
                        if price_change > 0 and hl_change < 0:
                            result['hl_divergence'] = -1  # Bearish divergence
                        elif price_change < 0 and hl_change > 0:
                            result['hl_divergence'] = 1   # Bullish divergence
                        else:
                            result['hl_divergence'] = 0   # No divergence
                            
        else:
            # No high-low data, try to use price data as a proxy
            df_copy = df.copy()
            
            if 'close' in df_copy.columns and len(df_copy) > 20:
                # Create a proxy using distance from high/low
                # Calculate 20-day high and low
                df_copy['rolling_high'] = df_copy['high'].rolling(window=20).max()
                df_copy['rolling_low'] = df_copy['low'].rolling(window=20).min()
                
                # Calculate distance from high/low as percentage
                df_copy['pct_from_high'] = (df_copy['rolling_high'] - df_copy['close']) / df_copy['rolling_high']
                df_copy['pct_from_low'] = (df_copy['close'] - df_copy['rolling_low']) / df_copy['close']
                
                # Store values
                result['pct_from_high'] = df_copy['pct_from_high'].iloc[-1]
                result['pct_from_low'] = df_copy['pct_from_low'].iloc[-1]
                
                # Calculate simple proxy
                # Close to high = bullish, close to low = bearish
                proxy_value = df_copy['pct_from_low'].iloc[-1] - df_copy['pct_from_high'].iloc[-1]
                result['hl_proxy'] = proxy_value
                
                # Normalize to a reasonable range
                result['normalized_proxy'] = max(-1.0, min(1.0, proxy_value * 5))
        
        return result
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to calculate high-low metrics and generate signals
        """
        # Need enough bars for calculation
        if len(historical_df) < max(10, self.hl_smoothing_period):
            self.latest_signal = 0.0
            return
            
        # Use lookback window if specified
        if self.lookback_period and len(historical_df) > self.lookback_period:
            df_subset = historical_df.iloc[-self.lookback_period:]
        else:
            df_subset = historical_df
            
        # Calculate high-low metrics
        self.hl_data = self._calculate_hl_metrics(df_subset)
        
        # Generate signal components
        signal_components = []
        
        # Primary component: High-Low Index
        if 'latest_hl_index_smooth' in self.hl_data:
            # Ensure it's in a reasonable range (typically -0.3 to 0.3)
            hl_index_signal = max(-1.0, min(1.0, self.hl_data['latest_hl_index_smooth'] * 3))
            signal_components.append(hl_index_signal)
            
        elif 'latest_hl_index' in self.hl_data:
            # Use non-smoothed value if smoothed isn't available
            hl_index_signal = max(-1.0, min(1.0, self.hl_data['latest_hl_index'] * 3))
            signal_components.append(hl_index_signal)
            
        # Add reversal signal if detected
        if 'reversal_signal' in self.hl_data:
            signal_components.append(self.hl_data['reversal_signal'])
            
        # Add divergence signal if detected
        if 'hl_divergence' in self.hl_data and self.hl_data['hl_divergence'] != 0:
            signal_components.append(self.hl_data['hl_divergence'] * 0.7)
            
        # Add extreme reading adjustments
        if 'extreme_highs' in self.hl_data and self.hl_data['extreme_highs']:
            # Extreme highs can signal either strength or exhaustion
            # Adjust based on whether the trend is accelerating or decelerating
            if 'hl_index_5d_change' in self.hl_data:
                if self.hl_data['hl_index_5d_change'] > 0:
                    # Accelerating highs = strong bullish
                    signal_components.append(0.9)
                else:
                    # Decelerating highs = potential exhaustion
                    signal_components.append(0.3)
            else:
                signal_components.append(0.7)  # Default without trend info
                
        if 'extreme_lows' in self.hl_data and self.hl_data['extreme_lows']:
            # Extreme lows can signal either weakness or potential bottoming
            if 'hl_index_5d_change' in self.hl_data:
                if self.hl_data['hl_index_5d_change'] < 0:
                    # Accelerating lows = strong bearish
                    signal_components.append(-0.9)
                else:
                    # Decelerating lows = potential bottoming
                    signal_components.append(-0.3)
            else:
                signal_components.append(-0.7)  # Default without trend info
                
        # Use proxy if needed
        if 'normalized_proxy' in self.hl_data and not signal_components:
            signal_components.append(self.hl_data['normalized_proxy'])
            
        # Calculate combined signal
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
        Predict new highs-lows signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate more new highs than lows (bullish)
          * Negative values indicate more new lows than highs (bearish)
          * Values near zero indicate balanced high-low readings
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "New Highs-Lows Agent" 