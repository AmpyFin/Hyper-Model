"""
Breadth Divergence Agent
~~~~~~~~~~~~~~~~~~
Detects divergences between price action and market breadth indicators to identify
potential market reversals and confirm or reject the strength of price movements.

Logic:
1. Track multiple breadth indicators (advance-decline, new highs/lows, etc.)
2. Compare the direction of price movement with breadth indicators
3. Identify when price makes new highs/lows but breadth fails to confirm
4. Generate signals when significant divergences are detected

Input: DataFrame containing market breadth data. Output ∈ [-1, +1] where:
* Positive values: Bullish breadth divergence (price down, breadth improving)
* Negative values: Bearish breadth divergence (price up, breadth deteriorating)
* Values near zero: No significant divergence detected
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class BreadthDivergenceAgent:
    def __init__(
        self,
        lookback_period: int = 60,          # Period for divergence analysis
        short_price_period: int = 10,       # Short-term price comparison period
        long_price_period: int = 30,        # Long-term price comparison period
        divergence_threshold: float = 0.15, # Threshold for divergence significance
        signal_smoothing: int = 3,          # Periods for signal smoothing
        divergence_types: List[str] = None, # Types of breadth to analyze for divergence
        fallback_mode: bool = True          # Use proxy indicators when breadth data unavailable
    ):
        self.lookback_period = lookback_period
        self.short_price_period = short_price_period
        self.long_price_period = long_price_period
        self.divergence_threshold = divergence_threshold
        self.signal_smoothing = signal_smoothing
        self.fallback_mode = fallback_mode
        
        # Default divergence types to analyze
        if divergence_types is None:
            self.divergence_types = ['advance_decline', 'new_highs_lows', 'up_down_volume', 'mcclellan']
        else:
            self.divergence_types = divergence_types
            
        self.latest_signal = 0.0
        self.signal_history = []
        self.divergence_data = {}
        
    def _calculate_divergences(self, df: pd.DataFrame) -> Dict:
        """Calculate divergences between price and breadth indicators"""
        result = {}
        
        # Check if we have price data
        if 'close' not in df.columns:
            return result
            
        df_copy = df.copy()
        
        # Calculate price trends
        # Short-term trend
        if len(df_copy) >= self.short_price_period:
            short_price_change = (df_copy['close'].iloc[-1] / df_copy['close'].iloc[-self.short_price_period]) - 1
            result['short_price_change'] = short_price_change
            
        # Long-term trend
        if len(df_copy) >= self.long_price_period:
            long_price_change = (df_copy['close'].iloc[-1] / df_copy['close'].iloc[-self.long_price_period]) - 1
            result['long_price_change'] = long_price_change
            
        # Check if price is making new highs or lows
        if len(df_copy) >= 20:
            # Calculate 20-period high and low
            period_high = df_copy['high'].iloc[-20:].max()
            period_low = df_copy['low'].iloc[-20:].min()
            
            latest_close = df_copy['close'].iloc[-1]
            
            result['near_high'] = (latest_close / period_high) > 0.98  # Within 2% of high
            result['near_low'] = (latest_close / period_low) < 1.02    # Within 2% of low
        
        # Check each divergence type
        for div_type in self.divergence_types:
            if div_type == 'advance_decline':
                # Advance-Decline divergence
                if all(col in df_copy.columns for col in ['advances', 'declines']):
                    # Calculate AD Line if not already present
                    if 'ad_line' not in df_copy.columns:
                        df_copy['net_advances'] = df_copy['advances'] - df_copy['declines']
                        df_copy['ad_line'] = df_copy['net_advances'].cumsum()
                        
                    # Compare AD line trend with price
                    if len(df_copy) >= 20:
                        # Short-term AD Line change
                        ad_short_change = (df_copy['ad_line'].iloc[-1] - df_copy['ad_line'].iloc[-10])
                        
                        # Normalize by average daily net advances
                        if len(df_copy) >= 30:
                            avg_net = abs(df_copy['net_advances'].iloc[-30:].mean())
                            if avg_net > 0:
                                ad_short_change = ad_short_change / (avg_net * 10)
                                
                        # Long-term AD Line change
                        ad_long_change = (df_copy['ad_line'].iloc[-1] - df_copy['ad_line'].iloc[-20])
                        if len(df_copy) >= 30:
                            if avg_net > 0:
                                ad_long_change = ad_long_change / (avg_net * 20)
                        
                        # Store changes
                        result['ad_short_change'] = ad_short_change
                        result['ad_long_change'] = ad_long_change
                        
                        # Check for divergences
                        # Bearish divergence: price up, AD line down
                        if ('short_price_change' in result and 
                            result['short_price_change'] > self.divergence_threshold and 
                            ad_short_change < -self.divergence_threshold):
                            result['ad_bearish_divergence'] = True
                            
                        # Bullish divergence: price down, AD line up
                        if ('short_price_change' in result and 
                            result['short_price_change'] < -self.divergence_threshold and 
                            ad_short_change > self.divergence_threshold):
                            result['ad_bullish_divergence'] = True
                            
            elif div_type == 'new_highs_lows':
                # New Highs/Lows divergence
                has_hl_data = False
                
                # Check different column naming conventions
                if all(col in df_copy.columns for col in ['new_highs', 'new_lows']):
                    high_col = 'new_highs'
                    low_col = 'new_lows'
                    has_hl_data = True
                elif all(col in df_copy.columns for col in ['new_52wk_highs', 'new_52wk_lows']):
                    high_col = 'new_52wk_highs'
                    low_col = 'new_52wk_lows'
                    has_hl_data = True
                    
                if has_hl_data and len(df_copy) >= 20:
                    # Calculate high-low difference
                    df_copy['hl_diff'] = df_copy[high_col] - df_copy[low_col]
                    
                    # Calculate high-low index if possible
                    if 'total_issues' in df_copy.columns:
                        df_copy['hl_index'] = df_copy['hl_diff'] / df_copy['total_issues']
                    else:
                        total_issues = df_copy[high_col] + df_copy[low_col]
                        if 'unchanged' in df_copy.columns:
                            total_issues += df_copy['unchanged']
                        df_copy['hl_index'] = df_copy['hl_diff'] / total_issues
                        
                    # Calculate HL Index changes
                    hl_short_change = df_copy['hl_index'].iloc[-1] - df_copy['hl_index'].iloc[-10]
                    hl_long_change = df_copy['hl_index'].iloc[-1] - df_copy['hl_index'].iloc[-20]
                    
                    result['hl_short_change'] = hl_short_change
                    result['hl_long_change'] = hl_long_change
                    
                    # Check for divergences
                    # Bearish: Price making new highs but fewer stocks at new highs
                    if result.get('near_high', False) and df_copy[high_col].iloc[-1] < df_copy[high_col].iloc[-20:].max():
                        result['hl_bearish_divergence'] = True
                        
                    # Bullish: Price making new lows but fewer stocks at new lows
                    if result.get('near_low', False) and df_copy[low_col].iloc[-1] < df_copy[low_col].iloc[-20:].max():
                        result['hl_bullish_divergence'] = True
                        
            elif div_type == 'up_down_volume':
                # Up/Down Volume divergence
                has_ud_data = False
                
                # Check different column naming conventions
                if all(col in df_copy.columns for col in ['up_volume', 'down_volume']):
                    up_vol_col = 'up_volume'
                    down_vol_col = 'down_volume'
                    has_ud_data = True
                elif all(col in df_copy.columns for col in ['advancing_volume', 'declining_volume']):
                    up_vol_col = 'advancing_volume'
                    down_vol_col = 'declining_volume'
                    has_ud_data = True
                    
                if has_ud_data and len(df_copy) >= 20:
                    # Calculate up/down volume ratio
                    df_copy['ud_ratio'] = df_copy[up_vol_col] / df_copy[down_vol_col].replace(0, 0.1)
                    
                    # Calculate 10-day average of ratio
                    df_copy['ud_ratio_10ma'] = df_copy['ud_ratio'].rolling(window=10).mean()
                    
                    if not pd.isna(df_copy['ud_ratio_10ma'].iloc[-1]) and not pd.isna(df_copy['ud_ratio_10ma'].iloc[-10]):
                        # Calculate changes
                        ud_change = np.log(df_copy['ud_ratio_10ma'].iloc[-1] / df_copy['ud_ratio_10ma'].iloc[-10])
                        result['ud_vol_change'] = ud_change
                        
                        # Check for divergences
                        # Bearish: Price up but volume ratio down
                        if ('short_price_change' in result and 
                            result['short_price_change'] > self.divergence_threshold and 
                            ud_change < -self.divergence_threshold):
                            result['ud_bearish_divergence'] = True
                            
                        # Bullish: Price down but volume ratio up
                        if ('short_price_change' in result and 
                            result['short_price_change'] < -self.divergence_threshold and 
                            ud_change > self.divergence_threshold):
                            result['ud_bullish_divergence'] = True
                            
            elif div_type == 'mcclellan':
                # McClellan Oscillator divergence
                if 'mcclellan_oscillator' in df_copy.columns and len(df_copy) >= 20:
                    # Calculate changes
                    mcc_short_change = df_copy['mcclellan_oscillator'].iloc[-1] - df_copy['mcclellan_oscillator'].iloc[-5]
                    result['mcclellan_change'] = mcc_short_change
                    
                    # Check for divergences
                    # Bearish: Price up but oscillator down
                    if ('short_price_change' in result and 
                        result['short_price_change'] > self.divergence_threshold and 
                        mcc_short_change < -20):  # Typical threshold for McClellan
                        result['mcclellan_bearish_divergence'] = True
                        
                    # Bullish: Price down but oscillator up
                    if ('short_price_change' in result and 
                        result['short_price_change'] < -self.divergence_threshold and 
                        mcc_short_change > 20):
                        result['mcclellan_bullish_divergence'] = True
        
        # Count total divergences
        bullish_divs = sum(1 for k in result if k.endswith('bullish_divergence') and result[k])
        bearish_divs = sum(1 for k in result if k.endswith('bearish_divergence') and result[k])
        
        result['bullish_divergence_count'] = bullish_divs
        result['bearish_divergence_count'] = bearish_divs
        
        # Fallback mode: Create proxy divergence signals from individual stock data
        if (self.fallback_mode and bullish_divs == 0 and bearish_divs == 0 and 
            'close' in df.columns and 'volume' in df.columns and len(df) >= 30):
            
            df_copy = df.copy()
            
            # Adjust thresholds for minute data (much smaller price movements)
            minute_price_threshold = self.divergence_threshold * 0.1  # 0.015 instead of 0.15
            
            # Create proxy breadth indicators from price and volume
            # 1. Price momentum vs Volume momentum divergence
            if len(df_copy) >= 20:
                # Calculate price momentum (rate of change)
                price_roc_short = (df_copy['close'].iloc[-1] / df_copy['close'].iloc[-5]) - 1
                price_roc_long = (df_copy['close'].iloc[-1] / df_copy['close'].iloc[-10]) - 1
                
                # Calculate volume momentum (average volume change)
                vol_ma_recent = df_copy['volume'].iloc[-5:].mean()
                vol_ma_older = df_copy['volume'].iloc[-15:-10].mean()
                vol_momentum = (vol_ma_recent / vol_ma_older) - 1 if vol_ma_older > 0 else 0
                
                # Check for divergences with adjusted thresholds
                # Bearish: Price up but volume momentum down (lack of confirmation)
                if price_roc_short > minute_price_threshold and vol_momentum < -0.1:
                    result['proxy_bearish_divergence'] = True
                    result['bearish_divergence_count'] += 1
                    
                # Bullish: Price down but volume momentum up (accumulation)
                if price_roc_short < -minute_price_threshold and vol_momentum > 0.1:
                    result['proxy_bullish_divergence'] = True
                    result['bullish_divergence_count'] += 1
                    
            # 2. High-Low range vs Price direction divergence
            if len(df_copy) >= 15:
                # Calculate average true range momentum
                df_copy['hl_range'] = df_copy['high'] - df_copy['low']
                atr_recent = df_copy['hl_range'].iloc[-5:].mean()
                atr_older = df_copy['hl_range'].iloc[-15:-10].mean()
                atr_momentum = (atr_recent / atr_older) - 1 if atr_older > 0 else 0
                
                # Price direction
                price_direction = (df_copy['close'].iloc[-1] / df_copy['close'].iloc[-10]) - 1
                
                # Bearish: Price up but volatility (uncertainty) increasing
                if price_direction > minute_price_threshold and atr_momentum > 0.2:
                    result['proxy_volatility_bearish'] = True
                    result['bearish_divergence_count'] += 1
                    
                # Bullish: Price down but volatility decreasing (stabilizing)
                if price_direction < -minute_price_threshold and atr_momentum < -0.1:
                    result['proxy_volatility_bullish'] = True
                    result['bullish_divergence_count'] += 1
                    
            # 3. Moving average convergence/divergence proxy
            if len(df_copy) >= 30:
                # Calculate short and long moving averages
                ma_short = df_copy['close'].rolling(window=5).mean()
                ma_long = df_copy['close'].rolling(window=15).mean()
                
                # Calculate MACD-like indicator
                macd_proxy = ma_short - ma_long
                macd_signal = macd_proxy.rolling(window=3).mean()
                
                # Check for MACD divergences
                if not pd.isna(macd_proxy.iloc[-1]) and not pd.isna(macd_proxy.iloc[-10]):
                    macd_change = macd_proxy.iloc[-1] - macd_proxy.iloc[-10]
                    price_change_10d = (df_copy['close'].iloc[-1] / df_copy['close'].iloc[-10]) - 1
                    
                    # Bearish: Price making higher highs but MACD making lower highs
                    if price_change_10d > minute_price_threshold and macd_change < -0.1:
                        result['proxy_macd_bearish'] = True
                        result['bearish_divergence_count'] += 1
                        
                    # Bullish: Price making lower lows but MACD making higher lows
                    if price_change_10d < -minute_price_threshold and macd_change > 0.1:
                        result['proxy_macd_bullish'] = True
                        result['bullish_divergence_count'] += 1
                        
            # 4. Simple price-volume relationship fallback
            # If no specific divergences found, generate a weak signal based on general price-volume relationship
            if result['bullish_divergence_count'] == 0 and result['bearish_divergence_count'] == 0:
                # Calculate simple price momentum
                price_momentum_5 = (df_copy['close'].iloc[-1] / df_copy['close'].iloc[-5]) - 1
                
                # Calculate volume ratio (recent vs older)
                vol_recent = df_copy['volume'].iloc[-3:].mean()
                vol_older = df_copy['volume'].iloc[-10:-7].mean()
                vol_ratio = vol_recent / vol_older if vol_older > 0 else 1.0
                
                # Generate weak signals based on price-volume relationship
                if price_momentum_5 > 0.0005 and vol_ratio > 1.2:  # Price up with volume
                    result['proxy_price_volume_bullish'] = True
                    result['bullish_divergence_count'] += 1
                elif price_momentum_5 < -0.0005 and vol_ratio < 0.8:  # Price down with low volume (potential bottom)
                    result['proxy_price_volume_bullish'] = True
                    result['bullish_divergence_count'] += 1
                elif price_momentum_5 > 0.0005 and vol_ratio < 0.8:  # Price up with low volume (weak move)
                    result['proxy_price_volume_bearish'] = True
                    result['bearish_divergence_count'] += 1
                elif price_momentum_5 < -0.0005 and vol_ratio > 1.2:  # Price down with volume (selling pressure)
                    result['proxy_price_volume_bearish'] = True
                    result['bearish_divergence_count'] += 1
        
        return result
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to detect breadth divergences and generate signals
        """
        # Need enough bars for calculation
        if len(historical_df) < self.long_price_period:
            self.latest_signal = 0.0
            return
            
        # Use lookback window if specified
        if self.lookback_period and len(historical_df) > self.lookback_period:
            df_subset = historical_df.iloc[-self.lookback_period:]
        else:
            df_subset = historical_df
            
        # Calculate divergences
        self.divergence_data = self._calculate_divergences(df_subset)
        
        # Generate signal based on divergences
        
        # Base signal components
        signal_components = []
        
        # Individual divergence signals
        # Advance-Decline divergences
        if self.divergence_data.get('ad_bearish_divergence', False):
            signal_components.append(-0.6)  # Bearish signal
            
        if self.divergence_data.get('ad_bullish_divergence', False):
            signal_components.append(0.6)   # Bullish signal
            
        # New Highs/Lows divergences
        if self.divergence_data.get('hl_bearish_divergence', False):
            signal_components.append(-0.7)  # Bearish signal
            
        if self.divergence_data.get('hl_bullish_divergence', False):
            signal_components.append(0.7)   # Bullish signal
            
        # Up/Down Volume divergences
        if self.divergence_data.get('ud_bearish_divergence', False):
            signal_components.append(-0.5)  # Bearish signal
            
        if self.divergence_data.get('ud_bullish_divergence', False):
            signal_components.append(0.5)   # Bullish signal
            
        # McClellan Oscillator divergences
        if self.divergence_data.get('mcclellan_bearish_divergence', False):
            signal_components.append(-0.4)  # Bearish signal
            
        if self.divergence_data.get('mcclellan_bullish_divergence', False):
            signal_components.append(0.4)   # Bullish signal
            
        # Proxy divergence signals (fallback mode)
        if self.divergence_data.get('proxy_bearish_divergence', False):
            signal_components.append(-0.3)  # Moderate bearish signal
            
        if self.divergence_data.get('proxy_bullish_divergence', False):
            signal_components.append(0.3)   # Moderate bullish signal
            
        if self.divergence_data.get('proxy_volatility_bearish', False):
            signal_components.append(-0.2)  # Weak bearish signal
            
        if self.divergence_data.get('proxy_volatility_bullish', False):
            signal_components.append(0.2)   # Weak bullish signal
            
        if self.divergence_data.get('proxy_macd_bearish', False):
            signal_components.append(-0.4)  # Moderate bearish signal
            
        if self.divergence_data.get('proxy_macd_bullish', False):
            signal_components.append(0.4)   # Moderate bullish signal
            
        # Simple price-volume relationship signals (weakest)
        if self.divergence_data.get('proxy_price_volume_bearish', False):
            signal_components.append(-0.1)  # Weak bearish signal
            
        if self.divergence_data.get('proxy_price_volume_bullish', False):
            signal_components.append(0.1)   # Weak bullish signal
            
        # Weight by number of confirming divergences
        bullish_count = self.divergence_data.get('bullish_divergence_count', 0)
        bearish_count = self.divergence_data.get('bearish_divergence_count', 0)
        
        # Multiple divergences strengthen the signal
        if bullish_count > 1:
            signal_components.append(min(0.3 * bullish_count, 0.9))
            
        if bearish_count > 1:
            signal_components.append(max(-0.3 * bearish_count, -0.9))
            
        # If no divergences but we have breadth data, add a weak neutral component
        # to avoid extreme readings in absence of breadth data
        if not signal_components:
            if 'ad_short_change' in self.divergence_data or 'hl_short_change' in self.divergence_data:
                signal_components.append(0.0)
                
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
        Predict breadth divergence signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate bullish divergence (price down, breadth up)
          * Negative values indicate bearish divergence (price up, breadth down)
          * Values near zero indicate no significant divergences
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Breadth Divergence Agent" 