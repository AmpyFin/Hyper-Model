"""
Up-Down Volume Agent
~~~~~~~~~~~~~~~
Analyzes the volume flowing into advancing versus declining stocks to measure
the conviction behind market moves and identify potential reversals.

Logic:
1. Calculate the ratio of up volume to down volume
2. Track up-down volume divergences from price movement
3. Identify volume surges that can signal capitulation or buying climaxes
4. Generate signals based on volume flow patterns and extremes

Input: DataFrame containing market breadth data. Output ∈ [-1, +1] where:
* Positive values: Strong up volume dominance (bullish)
* Negative values: Strong down volume dominance (bearish)
* Values near zero: Balanced up/down volume
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class UpDownVolumeAgent:
    def __init__(
        self,
        lookback_period: int = 50,         # Period for volume analysis
        volume_ma_period: int = 10,        # Moving average period for volume
        surge_threshold: float = 3.0,      # Threshold for volume surge detection
        signal_smoothing: int = 3,         # Periods for signal smoothing
        use_price_volume_proxy: bool = True # Use price-volume as proxy when needed
    ):
        self.lookback_period = lookback_period
        self.volume_ma_period = volume_ma_period
        self.surge_threshold = surge_threshold
        self.signal_smoothing = signal_smoothing
        self.use_price_volume_proxy = use_price_volume_proxy
        self.latest_signal = 0.0
        self.signal_history = []
        self.volume_data = {}
        
    def _calculate_volume_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate up-down volume metrics from market data"""
        result = {}
        
        # Check if we have the appropriate data columns
        has_updown_volume = False
        
        # Check for up/down volume data
        if all(col in df.columns for col in ['up_volume', 'down_volume']):
            has_updown_volume = True
            up_vol_col = 'up_volume'
            down_vol_col = 'down_volume'
        elif all(col in df.columns for col in ['advancing_volume', 'declining_volume']):
            has_updown_volume = True
            up_vol_col = 'advancing_volume'
            down_vol_col = 'declining_volume'
            
        if has_updown_volume:
            # Calculate up-down volume metrics
            df_copy = df.copy()
            
            # Basic metrics
            df_copy['up_down_ratio'] = df_copy[up_vol_col] / df_copy[down_vol_col].replace(0, 0.1)  # Avoid div by 0
            df_copy['net_volume'] = df_copy[up_vol_col] - df_copy[down_vol_col]
            
            # Calculate percentage of total volume
            df_copy['total_volume'] = df_copy[up_vol_col] + df_copy[down_vol_col]
            df_copy['up_volume_pct'] = df_copy[up_vol_col] / df_copy['total_volume']
            
            # Calculate moving averages
            df_copy['up_vol_ma'] = df_copy[up_vol_col].rolling(window=self.volume_ma_period).mean()
            df_copy['down_vol_ma'] = df_copy[down_vol_col].rolling(window=self.volume_ma_period).mean()
            df_copy['up_down_ratio_ma'] = df_copy['up_down_ratio'].rolling(window=self.volume_ma_period).mean()
            
            # Store latest values
            result['latest_up_down_ratio'] = df_copy['up_down_ratio'].iloc[-1]
            result['latest_up_volume_pct'] = df_copy['up_volume_pct'].iloc[-1]
            
            if not pd.isna(df_copy['up_down_ratio_ma'].iloc[-1]):
                result['latest_ratio_ma'] = df_copy['up_down_ratio_ma'].iloc[-1]
                
            # Check for volume surges
            # Calculate average daily volume
            avg_volume = df_copy['total_volume'].mean()
            
            # Check for surge in up volume
            latest_up_vol = df_copy[up_vol_col].iloc[-1]
            avg_up_vol = df_copy[up_vol_col].rolling(window=20).mean().iloc[-1]
            
            if not pd.isna(avg_up_vol) and avg_up_vol > 0:
                up_surge = latest_up_vol / avg_up_vol
                result['up_volume_surge'] = up_surge > self.surge_threshold
                
            # Check for surge in down volume
            latest_down_vol = df_copy[down_vol_col].iloc[-1]
            avg_down_vol = df_copy[down_vol_col].rolling(window=20).mean().iloc[-1]
            
            if not pd.isna(avg_down_vol) and avg_down_vol > 0:
                down_surge = latest_down_vol / avg_down_vol
                result['down_volume_surge'] = down_surge > self.surge_threshold
                
            # Calculate volume momentum
            if len(df_copy) >= 5:
                # 5-day momentum
                up_vol_5d_sum = df_copy[up_vol_col].iloc[-5:].sum()
                down_vol_5d_sum = df_copy[down_vol_col].iloc[-5:].sum()
                
                if up_vol_5d_sum + down_vol_5d_sum > 0:
                    result['volume_momentum_5d'] = (up_vol_5d_sum - down_vol_5d_sum) / (up_vol_5d_sum + down_vol_5d_sum)
                    
            # Check for potential capitulation or climax
            if result.get('down_volume_surge', False):
                # Potential selling capitulation
                # Check if this is coming after sustained selling
                if len(df_copy) >= 10:
                    prior_up_down = df_copy['up_down_ratio'].iloc[-10:-1].mean()
                    if prior_up_down < 0.8:  # Sustained selling before surge
                        result['potential_capitulation'] = True
                        
            if result.get('up_volume_surge', False):
                # Potential buying climax
                # Check if this is coming after sustained buying
                if len(df_copy) >= 10:
                    prior_up_down = df_copy['up_down_ratio'].iloc[-10:-1].mean()
                    if prior_up_down > 1.2:  # Sustained buying before surge
                        result['potential_climax'] = True
                        
            # Detect divergences
            if 'close' in df_copy.columns and len(df_copy) >= 20:
                # Price momentum
                price_change = (df_copy['close'].iloc[-1] / df_copy['close'].iloc[-20]) - 1
                
                # Volume momentum
                vol_change = None
                if not pd.isna(df_copy['up_down_ratio_ma'].iloc[-20]):
                    vol_change = df_copy['up_down_ratio_ma'].iloc[-1] - df_copy['up_down_ratio_ma'].iloc[-20]
                    
                    # Check for divergence
                    if price_change > 0.01 and vol_change < -0.2:  # Price up, volume down
                        result['volume_divergence'] = -1  # Bearish divergence
                    elif price_change < -0.01 and vol_change > 0.2:  # Price down, volume up
                        result['volume_divergence'] = 1   # Bullish divergence
                    else:
                        result['volume_divergence'] = 0
                        
        elif self.use_price_volume_proxy and 'volume' in df.columns and 'close' in df.columns:
            # No explicit up/down volume data, use price-volume as proxy
            df_copy = df.copy()
            
            # Calculate whether each day is up or down
            df_copy['price_change'] = df_copy['close'].diff()
            df_copy['is_up_day'] = df_copy['price_change'] > 0
            
            # Assign volume to up/down based on price direction
            df_copy['proxy_up_volume'] = np.where(df_copy['is_up_day'], df_copy['volume'], 0)
            df_copy['proxy_down_volume'] = np.where(~df_copy['is_up_day'], df_copy['volume'], 0)
            
            # Calculate up/down volume metrics using proxies
            if len(df_copy) >= self.volume_ma_period:
                # Calculate moving averages
                df_copy['up_vol_ma'] = df_copy['proxy_up_volume'].rolling(window=self.volume_ma_period).mean()
                df_copy['down_vol_ma'] = df_copy['proxy_down_volume'].rolling(window=self.volume_ma_period).mean()
                
                # Calculate up/down ratio
                df_copy['ud_ratio'] = df_copy['up_vol_ma'] / df_copy['down_vol_ma'].replace(0, 0.1)
                
                # Store results
                if not pd.isna(df_copy['ud_ratio'].iloc[-1]):
                    result['proxy_up_down_ratio'] = df_copy['ud_ratio'].iloc[-1]
                
                # Calculate volume momentum
                if len(df_copy) >= 5:
                    up_vol_5d = df_copy['proxy_up_volume'].iloc[-5:].sum()
                    down_vol_5d = df_copy['proxy_down_volume'].iloc[-5:].sum()
                    
                    if up_vol_5d + down_vol_5d > 0:
                        result['proxy_volume_momentum'] = (up_vol_5d - down_vol_5d) / (up_vol_5d + down_vol_5d)
                        
                # Check for volume surges
                if len(df_copy) >= 20:
                    avg_volume = df_copy['volume'].iloc[-20:-1].mean()
                    current_volume = df_copy['volume'].iloc[-1]
                    
                    if avg_volume > 0:
                        volume_surge = current_volume / avg_volume
                        result['volume_surge'] = volume_surge > self.surge_threshold
                        
                        # Determine if surge is on up or down day
                        if result['volume_surge']:
                            result['surge_on_up_day'] = df_copy['is_up_day'].iloc[-1]
        
        return result
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to calculate up-down volume metrics and generate signals
        """
        # Need enough bars for calculation
        if len(historical_df) < max(20, self.volume_ma_period):
            self.latest_signal = 0.0
            return
            
        # Use lookback window if specified
        if self.lookback_period and len(historical_df) > self.lookback_period:
            df_subset = historical_df.iloc[-self.lookback_period:]
        else:
            df_subset = historical_df
            
        # Calculate volume metrics
        self.volume_data = self._calculate_volume_metrics(df_subset)
        
        # Generate signal components
        signal_components = []
        
        # Primary component: Up-Down Volume Ratio
        if 'latest_up_down_ratio' in self.volume_data:
            # Log transform ratio for better scaling
            log_ratio = np.log(self.volume_data['latest_up_down_ratio'])
            
            # Scale to [-1, +1] range (log(2) ≈ 0.693, represents 2:1 ratio)
            normalized_ratio = log_ratio / np.log(3)  # Scale so log(3) maps to 1.0
            normalized_ratio = max(-1.0, min(1.0, normalized_ratio))
            
            signal_components.append(normalized_ratio)
            
        elif 'latest_ratio_ma' in self.volume_data:
            # Use MA if available
            log_ratio_ma = np.log(self.volume_data['latest_ratio_ma'])
            normalized_ratio_ma = log_ratio_ma / np.log(3)
            normalized_ratio_ma = max(-1.0, min(1.0, normalized_ratio_ma))
            
            signal_components.append(normalized_ratio_ma)
            
        elif 'proxy_up_down_ratio' in self.volume_data:
            # Use proxy ratio
            log_proxy_ratio = np.log(self.volume_data['proxy_up_down_ratio'])
            normalized_proxy = log_proxy_ratio / np.log(3)
            normalized_proxy = max(-1.0, min(1.0, normalized_proxy))
            
            signal_components.append(normalized_proxy * 0.8)  # Lower weight for proxy
            
        # Volume momentum component
        if 'volume_momentum_5d' in self.volume_data:
            signal_components.append(self.volume_data['volume_momentum_5d'])
        elif 'proxy_volume_momentum' in self.volume_data:
            signal_components.append(self.volume_data['proxy_volume_momentum'] * 0.8)
            
        # Climax and capitulation signals
        if self.volume_data.get('potential_capitulation', False):
            # Capitulation can be a bullish signal
            signal_components.append(0.5)  # Moderate bullish
            
        if self.volume_data.get('potential_climax', False):
            # Buying climax can be a bearish signal
            signal_components.append(-0.5)  # Moderate bearish
            
        # Volume surge signals
        if self.volume_data.get('up_volume_surge', False):
            signal_components.append(0.7)  # Strong bullish
            
        if self.volume_data.get('down_volume_surge', False):
            signal_components.append(-0.7)  # Strong bearish
            
        # Simple surge on proxy data
        if self.volume_data.get('volume_surge', False):
            if self.volume_data.get('surge_on_up_day', False):
                signal_components.append(0.6)  # Bullish surge
            else:
                signal_components.append(-0.6)  # Bearish surge
                
        # Volume divergence component
        if 'volume_divergence' in self.volume_data and self.volume_data['volume_divergence'] != 0:
            signal_components.append(self.volume_data['volume_divergence'] * 0.4)
            
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
        Predict up-down volume signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate stronger up volume (bullish)
          * Negative values indicate stronger down volume (bearish)
          * Values near zero indicate balanced volume flow
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Up-Down Volume Agent" 