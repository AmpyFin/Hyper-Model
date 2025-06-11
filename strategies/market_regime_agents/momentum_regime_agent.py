"""
Momentum Regime Agent
~~~~~~~~~~~~~~~~~
Identifies when the market is in a momentum regime where price continues to move in
the same direction and momentum strategies are likely to be effective.

Logic:
1. Track momentum across multiple timeframes (short/medium/long-term)
2. Measure consistency and strength of momentum
3. Analyze autocorrelation of returns for momentum persistence
4. Compare the effectiveness of momentum vs mean-reversion strategies
5. Generate signals when strong momentum is detected

Input: OHLCV DataFrame. Output ∈ [-1, +1] where:
* Positive values: Strong upward momentum regime
* Negative values: Strong downward momentum regime
* Values near zero: No clear momentum regime detected
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from scipy import stats


class MomentumRegimeAgent:
    def __init__(
        self,
        short_period: int = 20,      # Short-term momentum period
        medium_period: int = 60,     # Medium-term momentum period
        long_period: int = 120,      # Long-term momentum period
        correlation_period: int = 30, # Period for autocorrelation calculation
        signal_threshold: float = 0.3, # Threshold for significant momentum
        consistency_weight: float = 0.4 # Weight for momentum consistency vs strength
    ):
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        self.correlation_period = correlation_period
        self.signal_threshold = signal_threshold
        self.consistency_weight = consistency_weight
        self.latest_signal = 0.0
        self.momentum_metrics = {}
        
    def _calculate_momentum_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate various momentum metrics"""
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # Calculate returns
        df_copy['returns'] = df_copy['close'].pct_change()
        
        # Price momentum over different periods
        result = {}
        periods = [self.short_period, self.medium_period, self.long_period]
        
        for period in periods:
            if len(df_copy) < period:
                continue
                
            # Price ROC (Rate of Change)
            df_copy[f'roc_{period}'] = df_copy['close'].pct_change(period)
            
            # Normalize to daily rate for comparison
            df_copy[f'roc_daily_{period}'] = (1 + df_copy[f'roc_{period}']) ** (1 / period) - 1
            
            # Momentum z-score (how unusual is current momentum relative to history)
            rolling_mean = df_copy[f'roc_{period}'].rolling(window=max(period * 3, 120)).mean()
            rolling_std = df_copy[f'roc_{period}'].rolling(window=max(period * 3, 120)).std()
            df_copy[f'roc_zscore_{period}'] = (df_copy[f'roc_{period}'] - rolling_mean) / rolling_std
            
            # Store latest values
            result[f'roc_{period}'] = df_copy[f'roc_{period}'].iloc[-1]
            result[f'roc_daily_{period}'] = df_copy[f'roc_daily_{period}'].iloc[-1]
            result[f'roc_zscore_{period}'] = df_copy[f'roc_zscore_{period}'].iloc[-1]
        
        # Directional consistency
        # What percentage of days are moving in the same direction as the overall trend
        for period in periods:
            if len(df_copy) < period:
                continue
                
            overall_direction = np.sign(df_copy[f'roc_{period}'].iloc[-1])
            
            # Count days in same direction as trend
            days_in_direction = sum(np.sign(df_copy['returns'].iloc[-period:]) == overall_direction)
            
            # Calculate consistency ratio
            result[f'consistency_{period}'] = days_in_direction / period
        
        # Autocorrelation (momentum persistence)
        if len(df_copy) > self.correlation_period:
            returns = df_copy['returns'].iloc[-self.correlation_period:].dropna()
            if len(returns) > 5:  # Need sufficient data
                # Lag-1 autocorrelation
                result['autocorrelation'] = returns.autocorr(lag=1)
                
                # Higher order autocorrelations
                for lag in [3, 5]:
                    if len(returns) > lag + 5:
                        result[f'autocorrelation_{lag}'] = returns.autocorr(lag=lag)
        
        return result
    
    def _evaluate_momentum_regime(self, metrics: Dict) -> Tuple[float, float]:
        """
        Evaluate the momentum regime based on calculated metrics
        Returns (signal_strength, signal_direction)
        """
        # Default values if insufficient data
        signal_strength = 0.0
        signal_direction = 0.0
        
        # Check if we have momentum metrics
        if not metrics:
            return signal_strength, signal_direction
        
        # Direction of momentum (based on medium-term)
        if 'roc_medium_period' in metrics:
            signal_direction = np.sign(metrics['roc_{}'.format(self.medium_period)])
        elif 'roc_{}'.format(self.short_period) in metrics:
            signal_direction = np.sign(metrics['roc_{}'.format(self.short_period)])
        else:
            return signal_strength, signal_direction
        
        # Calculate momentum strength based on z-scores across timeframes
        zscore_sum = 0
        zscore_count = 0
        
        for period in ['short', 'medium', 'long']:
            period_value = getattr(self, f'{period}_period')
            if f'roc_zscore_{period_value}' in metrics:
                zscore_sum += abs(metrics[f'roc_zscore_{period_value}'])
                zscore_count += 1
        
        if zscore_count == 0:
            return signal_strength, signal_direction
            
        avg_zscore = zscore_sum / zscore_count
        
        # Calculate consistency across timeframes
        consistency_sum = 0
        consistency_count = 0
        
        for period in ['short', 'medium', 'long']:
            period_value = getattr(self, f'{period}_period')
            if f'consistency_{period_value}' in metrics:
                # Rescale from [0.5, 1.0] to [0, 1.0]
                # 0.5 = random, 1.0 = perfect consistency
                rescaled = (metrics[f'consistency_{period_value}'] - 0.5) * 2
                if rescaled > 0:  # Only consider positive consistency
                    consistency_sum += rescaled
                    consistency_count += 1
        
        avg_consistency = consistency_sum / max(consistency_count, 1)
        
        # Check autocorrelation (positive = momentum, negative = mean reversion)
        autocorrelation_factor = 1.0
        if 'autocorrelation' in metrics:
            # If autocorrelation is negative, reduce momentum signal
            autocorrelation_factor = max(0.0, (metrics['autocorrelation'] + 0.5) / 1.5)
        
        # Combined score
        strength_component = min(1.0, avg_zscore / 2.0) * (1.0 - self.consistency_weight)
        consistency_component = avg_consistency * self.consistency_weight
        
        signal_strength = (strength_component + consistency_component) * autocorrelation_factor
        
        # Apply threshold
        if signal_strength < self.signal_threshold:
            signal_strength = 0.0
            
        return signal_strength, signal_direction
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to identify the current momentum regime
        """
        # Need enough bars for calculation
        required_bars = max(self.short_period, self.medium_period, self.long_period)
        if len(historical_df) < required_bars:
            self.latest_signal = 0.0
            return
        
        # Calculate momentum metrics
        self.momentum_metrics = self._calculate_momentum_metrics(historical_df)
        
        # Evaluate momentum regime
        strength, direction = self._evaluate_momentum_regime(self.momentum_metrics)
        
        # Set the signal (-1 to +1)
        self.latest_signal = strength * direction
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict the momentum regime signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate strong upward momentum regime
          * Negative values indicate strong downward momentum regime
          * Values near zero indicate no clear momentum regime
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Momentum Regime Agent" 