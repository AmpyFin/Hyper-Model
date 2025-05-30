"""
Market Phase Agent
~~~~~~~~~~~~~~
Classifies the market into one of the four Wyckoff market phases:
- Accumulation: Sideways movement after a decline, with smart money accumulating positions
- Markup: Rising prices as the market trends upward with increased participation
- Distribution: Sideways movement after a rise, with smart money distributing positions
- Markdown: Falling prices as the market trends downward with increased selling

Logic:
1. Analyze price action and volume characteristics for each phase
2. Identify phase transitions using key price and volume patterns
3. Calculate probability scores for each market phase
4. Monitor for Wyckoff sequence confirmation and phase shifts
5. Adjust signals based on strength of evidence for current phase

Input: OHLCV DataFrame. Output ∈ [-1, +1] where:
* +1.0 to +0.5: Strong markup phase
* +0.5 to +0.1: Accumulation phase
* +0.1 to -0.1: Phase transition/indeterminate
* -0.1 to -0.5: Distribution phase
* -0.5 to -1.0: Strong markdown phase
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    ACCUMULATION = 1
    MARKUP = 2
    DISTRIBUTION = 3
    MARKDOWN = 4
    INDETERMINATE = 0


class MarketPhaseAgent:
    def __init__(
        self,
        price_window: int = 50,       # Window for price trend analysis
        volume_window: int = 20,      # Window for volume analysis
        volatility_window: int = 14,  # Window for volatility analysis
        ma_fast: int = 20,            # Fast moving average period
        ma_slow: int = 50,            # Slow moving average period
        ma_trend: int = 200,          # Trend moving average period
        threshold: float = 0.6        # Threshold for phase confirmation
    ):
        self.price_window = price_window
        self.volume_window = volume_window
        self.volatility_window = volatility_window
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.ma_trend = ma_trend
        self.threshold = threshold
        self.latest_signal = 0.0
        self.current_phase = MarketPhase.INDETERMINATE
        self.phase_probabilities = {
            MarketPhase.ACCUMULATION: 0.0,
            MarketPhase.MARKUP: 0.0,
            MarketPhase.DISTRIBUTION: 0.0,
            MarketPhase.MARKDOWN: 0.0
        }
        
    def _calculate_phase_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate metrics for market phase identification"""
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        result = {}
        
        # Calculate moving averages
        if len(df_copy) >= self.ma_trend:
            df_copy[f'ma_{self.ma_fast}'] = df_copy['close'].rolling(window=self.ma_fast).mean()
            df_copy[f'ma_{self.ma_slow}'] = df_copy['close'].rolling(window=self.ma_slow).mean()
            df_copy[f'ma_{self.ma_trend}'] = df_copy['close'].rolling(window=self.ma_trend).mean()
            
            # Price relative to moving averages
            result['close_vs_fast'] = df_copy['close'].iloc[-1] / df_copy[f'ma_{self.ma_fast}'].iloc[-1] - 1
            result['close_vs_slow'] = df_copy['close'].iloc[-1] / df_copy[f'ma_{self.ma_slow}'].iloc[-1] - 1
            result['close_vs_trend'] = df_copy['close'].iloc[-1] / df_copy[f'ma_{self.ma_trend}'].iloc[-1] - 1
            
            # Moving average alignment
            result['fast_vs_slow'] = df_copy[f'ma_{self.ma_fast}'].iloc[-1] / df_copy[f'ma_{self.ma_slow}'].iloc[-1] - 1
            result['slow_vs_trend'] = df_copy[f'ma_{self.ma_slow}'].iloc[-1] / df_copy[f'ma_{self.ma_trend}'].iloc[-1] - 1
            
            # Moving average slopes
            result['fast_slope'] = (df_copy[f'ma_{self.ma_fast}'].iloc[-1] / 
                                    df_copy[f'ma_{self.ma_fast}'].iloc[-10] - 1) * 100
            result['slow_slope'] = (df_copy[f'ma_{self.ma_slow}'].iloc[-1] / 
                                   df_copy[f'ma_{self.ma_slow}'].iloc[-10] - 1) * 100
            result['trend_slope'] = (df_copy[f'ma_{self.ma_trend}'].iloc[-1] / 
                                    df_copy[f'ma_{self.ma_trend}'].iloc[-10] - 1) * 100
        
        # Calculate price trend
        if len(df_copy) >= self.price_window:
            # Short-term trend (linear regression slope over price_window)
            x = np.arange(self.price_window)
            y = df_copy['close'].iloc[-self.price_window:].values
            try:
                # First check if the arrays are compatible
                if len(x) == len(y) and len(x) > 0:
                    polyfit_result = np.polyfit(x, y, 1, full=True)
                    slope = polyfit_result[0][0]  # First element is slope
                    intercept = polyfit_result[0][1]  # Second element is intercept
                    
                    # Normalized slope
                    result['price_slope'] = slope / np.mean(y) if np.mean(y) != 0 else 0
                    
                    # Calculate trend strength manually
                    y_mean = np.mean(y)
                    if y_mean != 0 and np.sum((y - y_mean) ** 2) != 0:
                        y_pred = slope * x + intercept
                        ss_total = np.sum((y - y_mean) ** 2)
                        ss_residual = np.sum((y - y_pred) ** 2)
                        result['trend_strength'] = 1 - (ss_residual / ss_total)
                    else:
                        result['trend_strength'] = 0
                else:
                    # If arrays are incompatible, set default values
                    result['price_slope'] = 0
                    result['trend_strength'] = 0
            except Exception as e:
                logger.warning(f"Error in linear regression: {e}")
                result['price_slope'] = 0
                result['trend_strength'] = 0
                
            # Determine if price is in an uptrend/downtrend/sideways
            if 'price_slope' in result:
                if abs(result['price_slope']) < 0.0001:  # Very low slope
                    result['price_trend'] = 'sideways'
                else:
                    result['price_trend'] = 'uptrend' if result['price_slope'] > 0 else 'downtrend'
        
        # Calculate volume analysis
        if len(df_copy) >= self.volume_window and 'volume' in df_copy.columns:
            # Volume trend
            df_copy['vol_ma'] = df_copy['volume'].rolling(window=self.volume_window).mean()
            result['volume_trend'] = (df_copy['volume'].iloc[-1] / 
                                     df_copy['vol_ma'].iloc[-1] - 1)
            
            # Volume on up days vs down days
            df_copy['up_day'] = df_copy['close'] > df_copy['close'].shift(1)
            recent_vol = df_copy.iloc[-self.volume_window:]
            up_vol = recent_vol.loc[recent_vol['up_day'], 'volume'].mean()
            down_vol = recent_vol.loc[~recent_vol['up_day'], 'volume'].mean()
            
            if not np.isnan(up_vol) and not np.isnan(down_vol) and down_vol != 0:
                result['up_down_vol_ratio'] = up_vol / down_vol
            else:
                result['up_down_vol_ratio'] = 1.0
                
            # Volume with price spread correlation
            df_copy['price_spread'] = df_copy['high'] - df_copy['low']
            vol_spread_corr = df_copy['volume'].iloc[-self.volume_window:].corr(
                df_copy['price_spread'].iloc[-self.volume_window:])
            result['vol_spread_corr'] = vol_spread_corr
            
            # Rising/falling volume with trend
            if 'price_trend' in result:
                if result['price_trend'] == 'uptrend':
                    # In uptrend, rising volume is bullish
                    result['volume_alignment'] = result['volume_trend']
                elif result['price_trend'] == 'downtrend':
                    # In downtrend, rising volume is bearish
                    result['volume_alignment'] = -result['volume_trend']
                else:
                    # In sideways, look at volume extremes
                    result['volume_alignment'] = 0
        
        # Calculate volatility
        if len(df_copy) >= self.volatility_window:
            # True Range
            df_copy['hl'] = df_copy['high'] - df_copy['low']
            df_copy['hc'] = abs(df_copy['high'] - df_copy['close'].shift(1))
            df_copy['lc'] = abs(df_copy['low'] - df_copy['close'].shift(1))
            df_copy['tr'] = df_copy[['hl', 'hc', 'lc']].max(axis=1)
            
            # ATR and normalized ATR
            df_copy['atr'] = df_copy['tr'].rolling(window=self.volatility_window).mean()
            df_copy['natr'] = df_copy['atr'] / df_copy['close']
            
            # Get latest values
            result['atr'] = df_copy['atr'].iloc[-1]
            result['natr'] = df_copy['natr'].iloc[-1]
            
            # Volatility trend
            vol_trend = (df_copy['natr'].iloc[-1] / 
                         df_copy['natr'].iloc[-self.volatility_window//2] - 1)
            result['volatility_trend'] = vol_trend
            
            # Recent volatility compared to longer term
            long_natr = df_copy['natr'].rolling(window=self.volatility_window*3).mean().iloc[-1]
            if not np.isnan(long_natr) and long_natr != 0:
                result['relative_volatility'] = result['natr'] / long_natr - 1
        
        # Price patterns
        if len(df_copy) >= self.price_window:
            # Check for higher highs/higher lows (uptrend)
            half_window = self.price_window // 2
            early_high = df_copy['high'].iloc[-self.price_window:-half_window].max()
            late_high = df_copy['high'].iloc[-half_window:].max()
            early_low = df_copy['low'].iloc[-self.price_window:-half_window].min()
            late_low = df_copy['low'].iloc[-half_window:].min()
            
            result['higher_highs'] = late_high > early_high
            result['higher_lows'] = late_low > early_low
            result['lower_highs'] = late_high < early_high
            result['lower_lows'] = late_low < early_low
        
        return result
    
    def _evaluate_market_phase(self, metrics: Dict) -> Dict[MarketPhase, float]:
        """
        Evaluate probabilities for each market phase
        Returns a dictionary of {phase: probability}
        """
        # Default phase probabilities
        probabilities = {
            MarketPhase.ACCUMULATION: 0.0,
            MarketPhase.MARKUP: 0.0,
            MarketPhase.DISTRIBUTION: 0.0,
            MarketPhase.MARKDOWN: 0.0
        }
        
        # Not enough metrics to determine phase
        if not metrics:
            return probabilities
        
        # ------ Accumulation Phase Characteristics ------
        accumulation_evidence = []
        
        # Price is in a sideways pattern after a decline
        if 'price_trend' in metrics and metrics['price_trend'] == 'sideways':
            if 'trend_slope' in metrics and metrics['trend_slope'] < 0:
                accumulation_evidence.append(0.7)  # Sideways after downtrend
            elif 'lower_lows' in metrics and not metrics['lower_lows']:
                accumulation_evidence.append(0.5)  # Sideways with stable lows
                
        # Price is near or slightly above its long-term moving average
        if 'close_vs_trend' in metrics:
            if -0.05 < metrics['close_vs_trend'] < 0.05:
                accumulation_evidence.append(0.6)  # Price near long term MA
            
        # Volume characteristics
        if 'up_down_vol_ratio' in metrics and metrics['up_down_vol_ratio'] > 1.2:
            accumulation_evidence.append(0.5)  # More volume on up days
            
        # Decreasing volatility
        if 'volatility_trend' in metrics and metrics['volatility_trend'] < -0.1:
            accumulation_evidence.append(0.4)  # Contracting volatility
            
        # Calculate accumulation probability
        if accumulation_evidence:
            probabilities[MarketPhase.ACCUMULATION] = sum(accumulation_evidence) / len(accumulation_evidence)
        
        # ------ Markup Phase Characteristics ------
        markup_evidence = []
        
        # Price is trending up
        if 'price_trend' in metrics and metrics['price_trend'] == 'uptrend':
            markup_evidence.append(0.7)
            
        # Moving averages are properly aligned (fast > slow > trend)
        if 'fast_vs_slow' in metrics and 'slow_vs_trend' in metrics:
            if metrics['fast_vs_slow'] > 0 and metrics['slow_vs_trend'] > 0:
                markup_evidence.append(0.8)  # Moving averages properly aligned
                
        # Price above moving averages
        if 'close_vs_fast' in metrics and 'close_vs_slow' in metrics and 'close_vs_trend' in metrics:
            if (metrics['close_vs_fast'] > 0 and metrics['close_vs_slow'] > 0 and 
                metrics['close_vs_trend'] > 0):
                markup_evidence.append(0.6)  # Price above all MAs
                
        # Higher highs and higher lows
        if 'higher_highs' in metrics and 'higher_lows' in metrics:
            if metrics['higher_highs'] and metrics['higher_lows']:
                markup_evidence.append(0.7)  # HH and HL pattern
                
        # Volume expanding with price increases
        if 'volume_alignment' in metrics and metrics['volume_alignment'] > 0.1:
            markup_evidence.append(0.5)  # Increasing volume with uptrend
            
        # Calculate markup probability
        if markup_evidence:
            probabilities[MarketPhase.MARKUP] = sum(markup_evidence) / len(markup_evidence)
            
        # ------ Distribution Phase Characteristics ------
        distribution_evidence = []
        
        # Price is in a sideways pattern after a rise
        if 'price_trend' in metrics and metrics['price_trend'] == 'sideways':
            if 'trend_slope' in metrics and metrics['trend_slope'] > 0:
                distribution_evidence.append(0.7)  # Sideways after uptrend
            elif 'higher_highs' in metrics and not metrics['higher_highs']:
                distribution_evidence.append(0.5)  # Sideways with stable highs
                
        # Volume characteristics (selling into strength)
        if 'up_down_vol_ratio' in metrics and metrics['up_down_vol_ratio'] < 0.8:
            distribution_evidence.append(0.6)  # More volume on down days
            
        # Slowing momentum
        if 'fast_slope' in metrics and 'slow_slope' in metrics:
            if metrics['fast_slope'] < metrics['slow_slope'] and metrics['fast_slope'] < 0:
                distribution_evidence.append(0.5)  # Fast MA turning down
                
        # Price struggles at resistance levels
        if 'lower_highs' in metrics and metrics['lower_highs']:
            distribution_evidence.append(0.4)  # Lower highs forming
            
        # Calculate distribution probability
        if distribution_evidence:
            probabilities[MarketPhase.DISTRIBUTION] = sum(distribution_evidence) / len(distribution_evidence)
            
        # ------ Markdown Phase Characteristics ------
        markdown_evidence = []
        
        # Price is trending down
        if 'price_trend' in metrics and metrics['price_trend'] == 'downtrend':
            markdown_evidence.append(0.7)
            
        # Moving averages are negatively aligned (fast < slow < trend)
        if 'fast_vs_slow' in metrics and 'slow_vs_trend' in metrics:
            if metrics['fast_vs_slow'] < 0 and metrics['slow_vs_trend'] < 0:
                markdown_evidence.append(0.8)  # Moving averages negatively aligned
                
        # Price below moving averages
        if 'close_vs_fast' in metrics and 'close_vs_slow' in metrics and 'close_vs_trend' in metrics:
            if (metrics['close_vs_fast'] < 0 and metrics['close_vs_slow'] < 0 and 
                metrics['close_vs_trend'] < 0):
                markdown_evidence.append(0.6)  # Price below all MAs
                
        # Lower highs and lower lows
        if 'lower_highs' in metrics and 'lower_lows' in metrics:
            if metrics['lower_highs'] and metrics['lower_lows']:
                markdown_evidence.append(0.7)  # LH and LL pattern
                
        # Volume expanding with price decreases
        if 'volume_alignment' in metrics and metrics['volume_alignment'] < -0.1:
            markdown_evidence.append(0.5)  # Increasing volume with downtrend
            
        # Calculate markdown probability
        if markdown_evidence:
            probabilities[MarketPhase.MARKDOWN] = sum(markdown_evidence) / len(markdown_evidence)
            
        return probabilities
        
    def _determine_signal(self, probabilities: Dict[MarketPhase, float]) -> Tuple[MarketPhase, float]:
        """
        Determine the current market phase and signal value
        Returns (market_phase, signal_value)
        """
        # Find the phase with highest probability
        max_prob = max(probabilities.values())
        if max_prob < self.threshold:
            return MarketPhase.INDETERMINATE, 0.0
            
        # Get the phase with max probability
        for phase, prob in probabilities.items():
            if prob == max_prob:
                current_phase = phase
                break
        
        # Convert phase to signal value in [-1, +1] range
        if current_phase == MarketPhase.ACCUMULATION:
            # Accumulation: slightly positive (0.1 to 0.5)
            signal = 0.1 + (0.4 * (max_prob - self.threshold) / (1 - self.threshold))
        elif current_phase == MarketPhase.MARKUP:
            # Markup: strongly positive (0.5 to 1.0)
            signal = 0.5 + (0.5 * (max_prob - self.threshold) / (1 - self.threshold))
        elif current_phase == MarketPhase.DISTRIBUTION:
            # Distribution: slightly negative (-0.1 to -0.5)
            signal = -0.1 - (0.4 * (max_prob - self.threshold) / (1 - self.threshold))
        elif current_phase == MarketPhase.MARKDOWN:
            # Markdown: strongly negative (-0.5 to -1.0)
            signal = -0.5 - (0.5 * (max_prob - self.threshold) / (1 - self.threshold))
        else:
            signal = 0.0
            
        return current_phase, signal
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to identify the current market phase
        """
        # Need enough bars for calculation
        required_bars = max(self.price_window, self.volume_window, self.volatility_window, self.ma_trend)
        if len(historical_df) < required_bars:
            self.latest_signal = 0.0
            return
        
        # Calculate phase metrics
        metrics = self._calculate_phase_metrics(historical_df)
        
        # Evaluate market phase probabilities
        self.phase_probabilities = self._evaluate_market_phase(metrics)
        
        # Determine current phase and signal
        self.current_phase, self.latest_signal = self._determine_signal(self.phase_probabilities)
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict the market phase signal based on latest data
        Returns a float in the range [-1, 1] where:
          * +1.0 to +0.5: Strong markup phase
          * +0.5 to +0.1: Accumulation phase
          * +0.1 to -0.1: Phase transition/indeterminate
          * -0.1 to -0.5: Distribution phase
          * -0.5 to -1.0: Strong markdown phase
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Market Phase Agent" 