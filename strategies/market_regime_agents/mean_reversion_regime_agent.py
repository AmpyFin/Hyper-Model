"""
Mean Reversion Regime Agent
~~~~~~~~~~~~~~~~~~~~~~~
Detects when the market is in a mean-reverting regime where prices tend to revert 
toward historical averages after moving away from them.

Logic:
1. Calculate multiple moving averages to establish equilibrium price levels
2. Identify extreme deviations from these moving averages
3. Analyze recent price behavior to detect mean-reverting tendencies
4. Calculate the Hurst Exponent to quantify mean-reversion vs trend strength
5. Combine these factors to determine the probability of mean reversion

Input: OHLCV DataFrame. Output ∈ [-1, +1] where:
* Positive values: Likely upward reversion (price is below equilibrium)
* Negative values: Likely downward reversion (price is above equilibrium)
* Values near zero: No strong mean reversion signal/trending regime
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


class MeanReversionRegimeAgent:
    def __init__(
        self,
        fast_ma: int = 20,           # Fast moving average period
        slow_ma: int = 50,           # Slow moving average period
        long_ma: int = 200,          # Long moving average period
        deviation_lookback: int = 100, # Lookback for deviation calculation
        hurst_period: int = 50,      # Period for Hurst exponent
        zscore_threshold: float = 2.0, # Z-score threshold for extreme deviation
        max_signal_strength: float = 0.8 # Maximum signal strength [0.0-1.0]
    ):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.long_ma = long_ma
        self.deviation_lookback = deviation_lookback
        self.hurst_period = hurst_period
        self.zscore_threshold = zscore_threshold
        self.max_signal_strength = max_signal_strength
        self.latest_signal = 0.0
        self.latest_hurst = 0.5
        self.latest_deviation = 0.0
        
    def _calculate_hurst_exponent(self, price_series: pd.Series) -> float:
        """
        Calculate the Hurst exponent to detect mean reversion vs trending
        H < 0.5: mean-reverting, H > 0.5: trending, H ≈ 0.5: random walk
        """
        # Need sufficient data
        if len(price_series) < self.hurst_period:
            return 0.5
            
        # Log returns
        returns = np.log(price_series / price_series.shift(1)).dropna()
        
        # If too few returns, return default
        if len(returns) < 20:
            return 0.5
            
        # Calculate Hurst using R/S analysis
        # Define range of tau values (lags)
        tau = [10, 15, 20, 25, 30, 40, 50]
        tau = [t for t in tau if t < len(returns)]
        
        if not tau:
            return 0.5
            
        # Calculate R/S values for different tau
        rs_values = []
        for lag in tau:
            # Split returns into subseries
            segments = len(returns) // lag
            rs_temp = []
            
            # Skip if not enough segments
            if segments == 0:
                continue
                
            # Calculate R/S for each segment
            for i in range(segments):
                series = returns.iloc[i*lag:(i+1)*lag].values
                
                # Mean-adjusted series
                mean_adj = series - np.mean(series)
                
                # Cumulative sum
                cum_series = np.cumsum(mean_adj)
                
                # Range (max - min of cumulative sum)
                r = max(cum_series) - min(cum_series)
                
                # Standard deviation
                s = np.std(series)
                
                # Skip if zero standard deviation
                if s == 0:
                    continue
                    
                # R/S value
                rs_temp.append(r / s)
            
            # Skip if no valid R/S values
            if not rs_temp:
                continue
                
            # Average R/S for this tau
            rs_values.append(np.mean(rs_temp))
            
        # Perform linear regression on log-log plot of tau vs R/S
        if len(tau) < 2 or len(rs_values) < 2:
            return 0.5
            
        log_tau = np.log10(tau[:len(rs_values)])
        log_rs = np.log10(rs_values)
        
        # Linear regression
        hurst = np.polyfit(log_tau, log_rs, 1)[0]
        
        return hurst
        
    def _calculate_ma_deviation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate deviations from moving averages"""
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # Calculate moving averages
        df_copy[f'ma_{self.fast_ma}'] = df_copy['close'].rolling(window=self.fast_ma).mean()
        df_copy[f'ma_{self.slow_ma}'] = df_copy['close'].rolling(window=self.slow_ma).mean()
        df_copy[f'ma_{self.long_ma}'] = df_copy['close'].rolling(window=self.long_ma).mean()
        
        # Calculate weighted average of the MAs
        df_copy['equilibrium'] = (
            df_copy[f'ma_{self.fast_ma}'] * 0.2 +
            df_copy[f'ma_{self.slow_ma}'] * 0.3 +
            df_copy[f'ma_{self.long_ma}'] * 0.5
        )
        
        # Calculate price deviation from equilibrium
        df_copy['deviation'] = (df_copy['close'] - df_copy['equilibrium']) / df_copy['equilibrium']
        
        # Calculate z-score of deviation
        lookback = min(self.deviation_lookback, len(df_copy))
        df_copy['deviation_zscore'] = (
            (df_copy['deviation'] - df_copy['deviation'].rolling(window=lookback).mean()) /
            df_copy['deviation'].rolling(window=lookback).std()
        )
        
        return df_copy
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to identify the current mean reversion regime
        """
        # Need enough bars for calculation
        required_bars = max(self.fast_ma, self.slow_ma, self.long_ma, self.hurst_period)
        if len(historical_df) < required_bars:
            self.latest_signal = 0.0
            return
        
        # Calculate moving average deviations
        df = self._calculate_ma_deviation(historical_df)
        
        # Get the latest values
        self.latest_deviation = df['deviation'].iloc[-1]
        latest_zscore = df['deviation_zscore'].iloc[-1]
        
        # Calculate Hurst exponent
        self.latest_hurst = self._calculate_hurst_exponent(historical_df['close'].iloc[-self.hurst_period:])
        
        # Determine mean reversion signal
        # If Hurst < 0.5, market is mean-reverting
        # Strength of mean reversion (0 to 1)
        mean_reversion_strength = max(0, 0.5 - self.latest_hurst) * 2
        
        # If deviation is extreme (high z-score), stronger mean reversion signal
        if abs(latest_zscore) > self.zscore_threshold:
            # Direction of likely reversion (negative = down, positive = up)
            direction = -1 if self.latest_deviation > 0 else 1
            
            # Strength based on z-score and mean reversion tendency
            deviation_factor = min(abs(latest_zscore) / self.zscore_threshold, 2.0) * 0.5
            
            # Combined signal
            signal_strength = mean_reversion_strength * deviation_factor * self.max_signal_strength
            
            # Apply direction
            self.latest_signal = signal_strength * direction
        else:
            # No strong deviation, weak signal
            self.latest_signal = 0.0
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict the mean reversion regime signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate likely upward mean reversion
          * Negative values indicate likely downward mean reversion
          * Values near zero indicate trending or random market
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Mean Reversion Regime Agent" 