"""
Market Cycle Agent
~~~~~~~~~~~~~~
Identifies major market cycles (bull market, bear market, and sideways market)
based on long-term price action, trend analysis, and cycle indicators.

Logic:
1. Analyze long-term trends using multiple timeframes
2. Compare price to key moving averages (50, 200)
3. Analyze cycle indicators (death cross, golden cross)
4. Measure momentum and trend strength
5. Determine the probability of being in each market cycle

Input: OHLCV DataFrame. Output ∈ [-1, +1] where:
* Values near +1: Strong bull market
* Values near 0: Sideways/transition market
* Values near -1: Strong bear market
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketCycle(Enum):
    BULL = 1
    BEAR = 2
    SIDEWAYS = 3
    UNKNOWN = 0


class MarketCycleAgent:
    def __init__(
        self,
        fast_ma: int = 50,             # Fast moving average (typically 50-day)
        slow_ma: int = 200,            # Slow moving average (typically 200-day)
        bull_threshold: float = 0.1,   # Threshold for bull market (% above slow MA)
        bear_threshold: float = -0.1,  # Threshold for bear market (% below slow MA)
        cycle_lookback: int = 252,     # Lookback period for cycle analysis (approx 1 year)
        momentum_period: int = 90,     # Period for momentum measurement
        signal_smoothing: int = 14     # Smoothing period for final signal
    ):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.cycle_lookback = cycle_lookback
        self.momentum_period = momentum_period
        self.signal_smoothing = signal_smoothing
        self.latest_signal = 0.0
        self.current_cycle = MarketCycle.UNKNOWN
        self.cycle_metrics = {}
        
    def _calculate_cycle_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate metrics for market cycle identification"""
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        result = {}
        
        # Calculate moving averages
        if len(df_copy) >= self.slow_ma:
            df_copy[f'ma_{self.fast_ma}'] = df_copy['close'].rolling(window=self.fast_ma).mean()
            df_copy[f'ma_{self.slow_ma}'] = df_copy['close'].rolling(window=self.slow_ma).mean()
            
            # Price relative to moving averages
            result['close_vs_fast'] = df_copy['close'].iloc[-1] / df_copy[f'ma_{self.fast_ma}'].iloc[-1] - 1
            result['close_vs_slow'] = df_copy['close'].iloc[-1] / df_copy[f'ma_{self.slow_ma}'].iloc[-1] - 1
            
            # Fast MA relative to Slow MA
            result['fast_vs_slow'] = df_copy[f'ma_{self.fast_ma}'].iloc[-1] / df_copy[f'ma_{self.slow_ma}'].iloc[-1] - 1
            
            # Detect recent crosses
            result['golden_cross'] = False
            result['death_cross'] = False
            
            # Check last 20 bars for crosses
            cross_window = min(20, len(df_copy) - self.slow_ma)
            if cross_window > 0:
                # Check for golden cross (fast MA crosses above slow MA)
                fast_ma_series = df_copy[f'ma_{self.fast_ma}'].iloc[-cross_window:]
                slow_ma_series = df_copy[f'ma_{self.slow_ma}'].iloc[-cross_window:]
                
                # Check if current fast > slow but previously fast < slow
                for i in range(1, cross_window):
                    if (fast_ma_series.iloc[i] > slow_ma_series.iloc[i] and 
                        fast_ma_series.iloc[i-1] <= slow_ma_series.iloc[i-1]):
                        result['golden_cross'] = True
                        break
                        
                # Check for death cross (fast MA crosses below slow MA)
                for i in range(1, cross_window):
                    if (fast_ma_series.iloc[i] < slow_ma_series.iloc[i] and 
                        fast_ma_series.iloc[i-1] >= slow_ma_series.iloc[i-1]):
                        result['death_cross'] = True
                        break
        
        # Calculate major trend
        if len(df_copy) >= self.cycle_lookback:
            # Long-term trend using linear regression
            x = np.arange(self.cycle_lookback)
            y = df_copy['close'].iloc[-self.cycle_lookback:].values
            try:
                # The issue is that np.polyfit is returning a tuple with 2 values, not 5
                polyfit_result = np.polyfit(x, y, 1, full=True)
                slope = polyfit_result[0][0]  # First element of first array is slope
                intercept = polyfit_result[0][1]  # Second element of first array is intercept
                
                # Calculate r_value manually if needed
                y_mean = np.mean(y)
                y_pred = slope * x + intercept
                ss_total = np.sum((y - y_mean) ** 2)
                ss_residual = np.sum((y - y_pred) ** 2)
                r_squared = 1 - (ss_residual / ss_total)
                r_value = np.sqrt(r_squared) * (1 if slope > 0 else -1)
                
                # Normalized slope
                result['long_term_slope'] = slope / np.mean(y)
                
                # Trend strength (R-squared)
                result['trend_strength'] = r_squared
            except Exception as e:
                logger.warning(f"Error in linear regression: {e}")
                result['long_term_slope'] = 0.0
                result['trend_strength'] = 0.0
            
            # Calculate price drawdown from peak
            rolling_max = df_copy['close'].rolling(window=self.cycle_lookback).max()
            current_drawdown = (df_copy['close'].iloc[-1] / rolling_max.iloc[-1]) - 1
            result['drawdown'] = current_drawdown
            
            # Calculate price drawup from trough
            rolling_min = df_copy['close'].rolling(window=self.cycle_lookback).min()
            current_drawup = (df_copy['close'].iloc[-1] / rolling_min.iloc[-1]) - 1
            result['drawup'] = current_drawup
        
        # Calculate momentum indicators
        if len(df_copy) >= self.momentum_period:
            # Rate of change
            df_copy['roc'] = df_copy['close'].pct_change(periods=self.momentum_period)
            result['momentum'] = df_copy['roc'].iloc[-1]
            
            # Percentage of positive days
            recent_returns = df_copy['close'].pct_change().iloc[-self.momentum_period:]
            result['percent_positive'] = (recent_returns > 0).mean()
            
            # Directional movement
            up_moves = recent_returns[recent_returns > 0].sum()
            down_moves = abs(recent_returns[recent_returns < 0].sum())
            
            if up_moves + down_moves > 0:
                result['directional_strength'] = (up_moves - down_moves) / (up_moves + down_moves)
            else:
                result['directional_strength'] = 0
        
        # Volume analysis
        if len(df_copy) >= self.momentum_period and 'volume' in df_copy.columns:
            # Volume trend
            df_copy['vol_ma'] = df_copy['volume'].rolling(window=self.momentum_period).mean()
            result['volume_trend'] = df_copy['volume'].iloc[-1] / df_copy['vol_ma'].iloc[-1] - 1
            
            # Up/down volume
            df_copy['up_day'] = df_copy['close'] > df_copy['close'].shift(1)
            up_vol = df_copy.loc[df_copy['up_day'], 'volume'].iloc[-self.momentum_period:].mean()
            down_vol = df_copy.loc[~df_copy['up_day'], 'volume'].iloc[-self.momentum_period:].mean()
            
            if down_vol > 0:
                result['up_down_volume'] = up_vol / down_vol
            else:
                result['up_down_volume'] = 1.0
            
        return result
    
    def _evaluate_market_cycle(self, metrics: Dict) -> Tuple[MarketCycle, float]:
        """
        Evaluate the current market cycle and signal value
        Returns (market_cycle, signal_value)
        """
        # Default to unknown cycle if insufficient data
        if not metrics or 'close_vs_slow' not in metrics:
            return MarketCycle.UNKNOWN, 0.0
        
        # ------ Bull Market Evidence ------
        bull_evidence = []
        
        # Price above moving averages
        if metrics['close_vs_slow'] > self.bull_threshold:
            bull_evidence.append(0.6)
            
        if metrics['close_vs_fast'] > 0:
            bull_evidence.append(0.4)
            
        # Fast MA above Slow MA
        if metrics['fast_vs_slow'] > 0:
            bull_evidence.append(0.7)
            
        # Recent golden cross
        if metrics.get('golden_cross', False):
            bull_evidence.append(0.8)
            
        # Positive long-term slope
        if 'long_term_slope' in metrics and metrics['long_term_slope'] > 0:
            # Scale by strength (0 to 0.7)
            slope_evidence = min(metrics['long_term_slope'] * 100, 0.7)
            if slope_evidence > 0.1:  # Only count if significant
                bull_evidence.append(slope_evidence)
                
        # Strong upward momentum
        if 'momentum' in metrics and metrics['momentum'] > 0:
            # Scale momentum (0 to 0.6)
            momentum_evidence = min(metrics['momentum'] * 5, 0.6)
            if momentum_evidence > 0.1:  # Only count if significant
                bull_evidence.append(momentum_evidence)
                
        # Majority of days positive
        if 'percent_positive' in metrics and metrics['percent_positive'] > 0.5:
            # Scale from 0.5-1.0 to 0-0.5
            pos_evidence = (metrics['percent_positive'] - 0.5) * 1.0
            bull_evidence.append(pos_evidence)
            
        # Significant drawup from lows
        if 'drawup' in metrics and metrics['drawup'] > 0.2:
            # Drawup evidence (max 0.5)
            drawup_evidence = min(metrics['drawup'] * 0.5, 0.5)
            bull_evidence.append(drawup_evidence)
            
        # ------ Bear Market Evidence ------
        bear_evidence = []
        
        # Price below moving averages
        if metrics['close_vs_slow'] < self.bear_threshold:
            bear_evidence.append(0.6)
            
        if metrics['close_vs_fast'] < 0:
            bear_evidence.append(0.4)
            
        # Fast MA below Slow MA
        if metrics['fast_vs_slow'] < 0:
            bear_evidence.append(0.7)
            
        # Recent death cross
        if metrics.get('death_cross', False):
            bear_evidence.append(0.8)
            
        # Negative long-term slope
        if 'long_term_slope' in metrics and metrics['long_term_slope'] < 0:
            # Scale by strength (0 to 0.7)
            slope_evidence = min(abs(metrics['long_term_slope'] * 100), 0.7)
            if slope_evidence > 0.1:  # Only count if significant
                bear_evidence.append(slope_evidence)
                
        # Strong downward momentum
        if 'momentum' in metrics and metrics['momentum'] < 0:
            # Scale momentum (0 to 0.6)
            momentum_evidence = min(abs(metrics['momentum'] * 5), 0.6)
            if momentum_evidence > 0.1:  # Only count if significant
                bear_evidence.append(momentum_evidence)
                
        # Majority of days negative
        if 'percent_positive' in metrics and metrics['percent_positive'] < 0.5:
            # Scale from 0.5-0 to 0-0.5
            neg_evidence = (0.5 - metrics['percent_positive']) * 1.0
            bear_evidence.append(neg_evidence)
            
        # Significant drawdown from peak
        if 'drawdown' in metrics and metrics['drawdown'] < -0.2:
            # Drawdown evidence (max 0.5)
            drawdown_evidence = min(abs(metrics['drawdown'] * 0.5), 0.5)
            bear_evidence.append(drawdown_evidence)
            
        # ------ Sideways Market Evidence ------
        sideways_evidence = []
        
        # Price near moving averages
        if abs(metrics['close_vs_slow']) < self.bull_threshold:
            sideways_evidence.append(0.5)
            
        # Fast MA near Slow MA
        if abs(metrics['fast_vs_slow']) < 0.03:  # Within 3%
            sideways_evidence.append(0.6)
            
        # Low long-term slope
        if 'long_term_slope' in metrics and abs(metrics['long_term_slope']) < 0.0001:
            sideways_evidence.append(0.7)
            
        # Weak trend strength
        if 'trend_strength' in metrics and metrics['trend_strength'] < 0.3:
            # Scale inverted R-squared (0.3-0 to 0-0.6)
            sideways_evidence.append((0.3 - metrics['trend_strength']) * 2)
            
        # Low momentum
        if 'momentum' in metrics and abs(metrics['momentum']) < 0.05:
            # Scale inverted momentum (0.05-0 to 0-0.5)
            sideways_evidence.append((0.05 - abs(metrics['momentum'])) * 10)
            
        # Calculate average evidence scores
        bull_score = sum(bull_evidence) / max(len(bull_evidence), 1) if bull_evidence else 0
        bear_score = sum(bear_evidence) / max(len(bear_evidence), 1) if bear_evidence else 0
        sideways_score = sum(sideways_evidence) / max(len(sideways_evidence), 1) if sideways_evidence else 0
        
        # Determine the dominant cycle
        if bull_score > bear_score and bull_score > sideways_score and bull_score > 0.4:
            cycle = MarketCycle.BULL
            signal = bull_score
        elif bear_score > bull_score and bear_score > sideways_score and bear_score > 0.4:
            cycle = MarketCycle.BEAR
            signal = -bear_score
        elif sideways_score > 0.3:
            cycle = MarketCycle.SIDEWAYS
            
            # For sideways markets, signal is biased slightly by bull/bear scores
            signal = 0.0
            if bull_score > bear_score:
                signal = sideways_score * 0.2  # Slightly positive
            elif bear_score > bull_score:
                signal = -sideways_score * 0.2  # Slightly negative
        else:
            # Indeterminate - not enough strong evidence for any cycle
            cycle = MarketCycle.UNKNOWN
            
            # Signal based on balance of bull/bear evidence
            signal = (bull_score - bear_score) * 0.5  # Scaled down for uncertainty
        
        # Ensure signal is in [-1, 1] range
        signal = max(-1.0, min(1.0, signal))
        
        return cycle, signal
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to identify the current market cycle
        """
        # Need enough bars for calculation
        required_bars = max(self.fast_ma, self.slow_ma, self.cycle_lookback)
        if len(historical_df) < required_bars:
            self.latest_signal = 0.0
            self.current_cycle = MarketCycle.UNKNOWN
            return
        
        # Calculate cycle metrics
        self.cycle_metrics = self._calculate_cycle_metrics(historical_df)
        
        # Evaluate market cycle
        self.current_cycle, raw_signal = self._evaluate_market_cycle(self.cycle_metrics)
        
        # Apply simple signal smoothing if history is available
        if not hasattr(self, 'signal_history'):
            self.signal_history = []
            
        self.signal_history.append(raw_signal)
        if len(self.signal_history) > self.signal_smoothing:
            self.signal_history.pop(0)
            
        # Smoothed signal is the average of recent raw signals
        self.latest_signal = sum(self.signal_history) / len(self.signal_history)
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict the market cycle signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate bull market (stronger the more positive)
          * Negative values indicate bear market (stronger the more negative)
          * Values near zero indicate sideways market or indeterminate
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Market Cycle Agent" 