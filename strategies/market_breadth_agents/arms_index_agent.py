"""
Arms Index (TRIN) Agent
~~~~~~~~~~~~~~~~~
Implements the Arms Index (also known as TRIN or TRading INdex), which measures
the relationship between advancing/declining issues and their associated volume.

Logic:
1. Calculate the Arms Index (TRIN): (advancing issues/declining issues) / (advancing volume/declining volume)
2. Monitor short and long-term TRIN readings to identify overbought/oversold conditions
3. Identify divergences between TRIN and price movement
4. Generate signals based on extreme readings and trend reversals

Input: DataFrame containing market breadth data. Output ∈ [-1, +1] where:
* Positive values: Bullish TRIN readings (typically low TRIN values)
* Negative values: Bearish TRIN readings (typically high TRIN values)
* Values near zero: Neutral TRIN readings
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class ArmsIndexAgent:
    def __init__(
        self,
        lookback_period: int = 50,        # Period for TRIN analysis
        short_ma_period: int = 5,         # Short-term moving average period
        long_ma_period: int = 20,         # Long-term moving average period
        overbought_level: float = 0.65,   # Threshold for overbought (low TRIN)
        oversold_level: float = 1.5,      # Threshold for oversold (high TRIN)
        signal_smoothing: int = 3,        # Periods for signal smoothing
        use_inverse_trin: bool = False    # Whether to use 1/TRIN for easier interpretation
    ):
        self.lookback_period = lookback_period
        self.short_ma_period = short_ma_period
        self.long_ma_period = long_ma_period
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
        self.signal_smoothing = signal_smoothing
        self.use_inverse_trin = use_inverse_trin
        self.latest_signal = 0.0
        self.signal_history = []
        self.trin_data = {}
        
    def _calculate_trin(self, df: pd.DataFrame) -> Dict:
        """Calculate Arms Index (TRIN) from market data"""
        result = {}
        
        # Check if we have the necessary data columns
        has_trin_data = False
        
        # Check for direct TRIN column
        if 'trin' in df.columns or 'arms_index' in df.columns:
            has_trin_data = True
            trin_col = 'trin' if 'trin' in df.columns else 'arms_index'
            df_copy = df.copy()
            
        # Check for components to calculate TRIN
        elif all(item in df.columns for item in ['advances', 'declines', 'up_volume', 'down_volume']):
            has_trin_data = True
            df_copy = df.copy()
            
            # Calculate TRIN from components
            advance_ratio = df_copy['advances'] / df_copy['declines'].replace(0, 0.001)  # Avoid div by zero
            volume_ratio = df_copy['up_volume'] / df_copy['down_volume'].replace(0, 0.001)
            df_copy['trin'] = advance_ratio / volume_ratio
            trin_col = 'trin'
            
        # Alternative column names
        elif all(item in df.columns for item in ['advancing_issues', 'declining_issues', 
                                               'advancing_volume', 'declining_volume']):
            has_trin_data = True
            df_copy = df.copy()
            
            # Calculate TRIN from components
            advance_ratio = df_copy['advancing_issues'] / df_copy['declining_issues'].replace(0, 0.001)
            volume_ratio = df_copy['advancing_volume'] / df_copy['declining_volume'].replace(0, 0.001)
            df_copy['trin'] = advance_ratio / volume_ratio
            trin_col = 'trin'
            
        if has_trin_data:
            # Calculate inverse TRIN if requested
            if self.use_inverse_trin:
                df_copy['inv_trin'] = 1 / df_copy[trin_col].replace(0, 0.001)
                working_col = 'inv_trin'
            else:
                working_col = trin_col
                
            # TRIN sometimes has extreme values, so cap them
            df_copy['capped_trin'] = df_copy[working_col].clip(lower=0.1, upper=10.0)
            working_col = 'capped_trin'
            
            # Calculate TRIN moving averages
            df_copy[f'trin_{self.short_ma_period}ma'] = df_copy[working_col].rolling(window=self.short_ma_period).mean()
            df_copy[f'trin_{self.long_ma_period}ma'] = df_copy[working_col].rolling(window=self.long_ma_period).mean()
            
            # Store latest values
            result['latest_trin'] = df_copy[trin_col].iloc[-1]
            
            if not pd.isna(df_copy[f'trin_{self.short_ma_period}ma'].iloc[-1]):
                result[f'trin_{self.short_ma_period}ma'] = df_copy[f'trin_{self.short_ma_period}ma'].iloc[-1]
                
            if not pd.isna(df_copy[f'trin_{self.long_ma_period}ma'].iloc[-1]):
                result[f'trin_{self.long_ma_period}ma'] = df_copy[f'trin_{self.long_ma_period}ma'].iloc[-1]
                
            # Check for extreme TRIN readings
            # Note: For standard TRIN, low = bullish, high = bearish
            # For inverse TRIN, high = bullish, low = bearish
            if self.use_inverse_trin:
                result['is_overbought'] = df_copy[working_col].iloc[-1] > (1 / self.overbought_level)
                result['is_oversold'] = df_copy[working_col].iloc[-1] < (1 / self.oversold_level)
            else:
                result['is_overbought'] = df_copy[working_col].iloc[-1] < self.overbought_level
                result['is_oversold'] = df_copy[working_col].iloc[-1] > self.oversold_level
                
            # Calculate TRIN momentum
            if len(df_copy) >= 5:
                trin_momentum = df_copy[working_col].iloc[-1] - df_copy[working_col].iloc[-5]
                result['trin_momentum'] = trin_momentum
                
                # For standard TRIN, decreasing = bullish, increasing = bearish
                # For inverse TRIN, increasing = bullish, decreasing = bearish
                if not self.use_inverse_trin:
                    trin_momentum = -trin_momentum  # Invert for standard TRIN
                    
                # Normalize momentum
                result['normalized_momentum'] = np.clip(trin_momentum * 2, -1, 1)
                
            # Check for TRIN moving average crossovers
            if (f'trin_{self.short_ma_period}ma' in df_copy.columns and 
                f'trin_{self.long_ma_period}ma' in df_copy.columns and 
                len(df_copy) >= 2):
                
                curr_short_ma = df_copy[f'trin_{self.short_ma_period}ma'].iloc[-1]
                prev_short_ma = df_copy[f'trin_{self.short_ma_period}ma'].iloc[-2]
                curr_long_ma = df_copy[f'trin_{self.long_ma_period}ma'].iloc[-1]
                prev_long_ma = df_copy[f'trin_{self.long_ma_period}ma'].iloc[-2]
                
                if not pd.isna(curr_short_ma) and not pd.isna(prev_short_ma) and not pd.isna(curr_long_ma) and not pd.isna(prev_long_ma):
                    # Detect crossovers
                    if self.use_inverse_trin:
                        # For inverse TRIN, short MA crossing above long MA = bullish
                        if prev_short_ma < prev_long_ma and curr_short_ma > curr_long_ma:
                            result['bullish_crossover'] = True
                        # Short MA crossing below long MA = bearish
                        elif prev_short_ma > prev_long_ma and curr_short_ma < curr_long_ma:
                            result['bearish_crossover'] = True
                    else:
                        # For standard TRIN, short MA crossing below long MA = bullish
                        if prev_short_ma > prev_long_ma and curr_short_ma < curr_long_ma:
                            result['bullish_crossover'] = True
                        # Short MA crossing above long MA = bearish
                        elif prev_short_ma < prev_long_ma and curr_short_ma > curr_long_ma:
                            result['bearish_crossover'] = True
                            
            # Check for divergences against price if price data available
            if 'close' in df_copy.columns and len(df_copy) >= 20:
                # Calculate price movement
                price_change = (df_copy['close'].iloc[-1] / df_copy['close'].iloc[-10]) - 1
                
                # Calculate TRIN movement (invert if using standard TRIN)
                trin_change = df_copy[working_col].iloc[-1] - df_copy[working_col].iloc[-10]
                if not self.use_inverse_trin:
                    trin_change = -trin_change  # Invert for standard TRIN
                    
                # Look for divergences
                if price_change > 0.02 and trin_change < -0.3:
                    # Price up but TRIN bearish
                    result['bearish_divergence'] = True
                elif price_change < -0.02 and trin_change > 0.3:
                    # Price down but TRIN bullish
                    result['bullish_divergence'] = True
        else:
            # No TRIN data, try to create a proxy
            if 'volume' in df.columns and 'close' in df.columns and len(df) >= 10:
                df_copy = df.copy()
                
                # Create a simple proxy using volume and price changes
                df_copy['price_change'] = df_copy['close'].diff()
                df_copy['up_day'] = df_copy['price_change'] > 0
                df_copy['down_day'] = df_copy['price_change'] < 0
                
                # Calculate up/down volume
                df_copy['proxy_up_vol'] = np.where(df_copy['up_day'], df_copy['volume'], 0)
                df_copy['proxy_down_vol'] = np.where(df_copy['down_day'], df_copy['volume'], 0)
                
                # Calculate 5-day sums for a rough proxy
                if len(df_copy) >= 5:
                    up_days = df_copy['up_day'].rolling(window=5).sum()
                    down_days = df_copy['down_day'].rolling(window=5).sum()
                    up_vol = df_copy['proxy_up_vol'].rolling(window=5).sum()
                    down_vol = df_copy['proxy_down_vol'].rolling(window=5).sum()
                    
                    # Calculate proxy TRIN
                    day_ratio = up_days / down_days.replace(0, 0.001)
                    vol_ratio = up_vol / down_vol.replace(0, 0.001)
                    df_copy['proxy_trin'] = day_ratio / vol_ratio
                    
                    # Store results
                    if not pd.isna(df_copy['proxy_trin'].iloc[-1]):
                        result['proxy_trin'] = df_copy['proxy_trin'].iloc[-1]
                        
                        # Create normalized signal from proxy
                        # TRIN typically ranges from 0.5 to 3.0
                        # Convert to [-1, 1] range (invert since low TRIN is bullish)
                        norm_proxy = -np.clip((df_copy['proxy_trin'].iloc[-1] - 1.0) / 2.0, -1.0, 1.0)
                        result['normalized_proxy'] = norm_proxy
        
        return result
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to calculate TRIN metrics and generate signals
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
            
        # Calculate TRIN metrics
        self.trin_data = self._calculate_trin(df_subset)
        
        # Generate signal components
        signal_components = []
        
        # Primary component: TRIN reading
        # For TRIN, values < 1 are generally bullish, values > 1 are bearish
        # Need to convert to [-1, 1] range
        if 'latest_trin' in self.trin_data:
            latest = self.trin_data['latest_trin']
            
            # Use log scale for more balanced scaling
            log_trin = np.log(latest)
            
            # Scale to [-1, 1] range (log(0.5) ≈ -0.693, log(2) ≈ 0.693)
            # Invert because low TRIN is bullish
            normalized_trin = -np.clip(log_trin / np.log(3), -1.0, 1.0)
            
            signal_components.append(normalized_trin)
            
        elif 'normalized_proxy' in self.trin_data:
            signal_components.append(self.trin_data['normalized_proxy'] * 0.7)  # Lower weight for proxy
            
        # TRIN moving average signals
        short_ma_key = f'trin_{self.short_ma_period}ma'
        long_ma_key = f'trin_{self.long_ma_period}ma'
        
        if short_ma_key in self.trin_data and long_ma_key in self.trin_data:
            short_ma = self.trin_data[short_ma_key]
            long_ma = self.trin_data[long_ma_key]
            
            # Calculate MA relationship (depends on whether using inverse TRIN)
            if self.use_inverse_trin:
                ma_ratio = short_ma / long_ma if long_ma > 0 else 1.0
                ma_signal = np.clip((ma_ratio - 1.0) * 2.0, -1.0, 1.0)
            else:
                ma_ratio = long_ma / short_ma if short_ma > 0 else 1.0
                ma_signal = np.clip((ma_ratio - 1.0) * 2.0, -1.0, 1.0)
                
            signal_components.append(ma_signal)
            
        # Crossover signals
        if self.trin_data.get('bullish_crossover', False):
            signal_components.append(0.7)  # Strong bullish
            
        if self.trin_data.get('bearish_crossover', False):
            signal_components.append(-0.7)  # Strong bearish
            
        # Extreme readings
        if self.trin_data.get('is_overbought', False):
            # Overbought in TRIN is bullish
            signal_components.append(0.6)
            
        if self.trin_data.get('is_oversold', False):
            # Oversold in TRIN is bearish
            signal_components.append(-0.6)
            
        # Momentum signals
        if 'normalized_momentum' in self.trin_data:
            signal_components.append(self.trin_data['normalized_momentum'])
            
        # Divergence signals
        if self.trin_data.get('bullish_divergence', False):
            signal_components.append(0.8)  # Strong bullish
            
        if self.trin_data.get('bearish_divergence', False):
            signal_components.append(-0.8)  # Strong bearish
            
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
        Predict Arms Index (TRIN) signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate bullish TRIN readings (low TRIN values)
          * Negative values indicate bearish TRIN readings (high TRIN values)
          * Values near zero indicate neutral TRIN readings
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Arms Index (TRIN) Agent" 