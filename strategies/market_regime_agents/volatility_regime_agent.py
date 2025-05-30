"""
Volatility Regime Agent
~~~~~~~~~~~~~~~~~~~~
Classifies the current market volatility regime (low, medium, high) using historical volatility
measurements and adaptive thresholds.

Logic:
1. Calculate historical volatility using standard deviation of returns
2. Compare current volatility to a lookback window to determine relative regime
3. Apply adaptive thresholds based on the asset's typical volatility profile
4. Classify the market into:
   - Low volatility regime (consolidation/complacency)
   - Medium volatility regime (normal trading conditions)
   - High volatility regime (fear/panic/euphoria)

Input: OHLCV DataFrame. Output ∈ [-1, +1] where:
* Values near 0: Normal/medium volatility
* Values near -1: Low volatility regime
* Values near +1: High volatility regime

Note: The signal indicates the regime, not a directional prediction.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


class VolatilityRegimeAgent:
    def __init__(
        self,
        volatility_window: int = 20,     # Window for volatility calculation
        regime_lookback: int = 252,      # Lookback for regime determination
        high_quantile: float = 0.75,     # Threshold for high volatility
        low_quantile: float = 0.25,      # Threshold for low volatility
        use_atr: bool = True,            # Use ATR instead of close prices
        scale_factor: float = 2.0        # Scale factor for output normalization
    ):
        self.volatility_window = volatility_window
        self.regime_lookback = regime_lookback
        self.high_quantile = high_quantile
        self.low_quantile = low_quantile
        self.use_atr = use_atr
        self.scale_factor = scale_factor
        self.latest_signal = 0.0
        self.latest_vol = 0.0
        self.vol_percentile = 0.5  # Default to medium
        
    def _calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate historical volatility using different methods"""
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        if self.use_atr:
            # Calculate ATR (Average True Range)
            df_copy['hl'] = df_copy['high'] - df_copy['low']
            df_copy['hc'] = abs(df_copy['high'] - df_copy['close'].shift(1))
            df_copy['lc'] = abs(df_copy['low'] - df_copy['close'].shift(1))
            df_copy['tr'] = df_copy[['hl', 'hc', 'lc']].max(axis=1)
            df_copy['atr'] = df_copy['tr'].rolling(window=self.volatility_window).mean()
            
            # Normalize ATR by dividing by price level
            df_copy['norm_atr'] = df_copy['atr'] / df_copy['close']
            
            # Store as volatility
            df_copy['volatility'] = df_copy['norm_atr']
        else:
            # Calculate volatility using standard deviation of returns
            df_copy['returns'] = df_copy['close'].pct_change()
            df_copy['volatility'] = df_copy['returns'].rolling(window=self.volatility_window).std() * np.sqrt(252)
            
        return df_copy
    
    def _classify_regime(self, volatility: float, vol_series: pd.Series) -> Tuple[float, float]:
        """
        Classify the volatility regime
        Returns (regime_signal, percentile)
        """
        # If we don't have enough data, return middle value
        if len(vol_series.dropna()) < 10:
            return 0.0, 0.5
            
        # Calculate percentile of current volatility
        percentile = sum(vol_series <= volatility) / sum(~vol_series.isna())
        
        # Classify based on percentile thresholds
        if percentile >= self.high_quantile:
            # High volatility regime
            # Scale from self.high_quantile (0) to 1.0 (1.0)
            regime_signal = self.scale_factor * (percentile - self.high_quantile) / (1.0 - self.high_quantile)
            regime_signal = min(1.0, regime_signal)  # Cap at 1.0
        elif percentile <= self.low_quantile:
            # Low volatility regime
            # Scale from 0.0 (-1.0) to self.low_quantile (0)
            regime_signal = -self.scale_factor * (self.low_quantile - percentile) / self.low_quantile
            regime_signal = max(-1.0, regime_signal)  # Cap at -1.0
        else:
            # Medium volatility regime
            # Scale from low_quantile to high_quantile to range from -0.1 to +0.1
            normalized = (percentile - self.low_quantile) / (self.high_quantile - self.low_quantile)
            regime_signal = 0.2 * (normalized - 0.5)
            
        return regime_signal, percentile
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to identify the current volatility regime
        """
        # Need enough bars for calculation
        if len(historical_df) < self.volatility_window:
            self.latest_signal = 0.0
            return
        
        # Calculate volatility
        df = self._calculate_volatility(historical_df)
        
        # Get the latest volatility
        self.latest_vol = df['volatility'].iloc[-1]
        
        # Get historical volatility for regime classification
        lookback = min(self.regime_lookback, len(df))
        historical_vol = df['volatility'].iloc[-lookback:-1]  # Exclude current point
        
        # Classify volatility regime
        self.latest_signal, self.vol_percentile = self._classify_regime(self.latest_vol, historical_vol)
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict the volatility regime signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate high volatility regime
          * Negative values indicate low volatility regime
          * Values near zero indicate normal volatility
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Volatility Regime Agent" 