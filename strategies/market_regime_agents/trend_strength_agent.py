"""
Trend Strength Agent
~~~~~~~~~~~~~~~~
Detects the strength and direction of market trends using ADX (Average Directional Index) and DMI
(Directional Movement Index).

Logic:
1. Calculate ADX to measure trend strength regardless of direction
2. Calculate +DI and -DI to determine trend direction
3. Combine these indicators to classify the market regime:
   - Strong uptrend (high ADX, +DI > -DI)
   - Strong downtrend (high ADX, -DI > +DI) 
   - Weak/no trend (low ADX)

Input: OHLCV DataFrame. Output ∈ [-1, +1] where:
* Positive values: Strong uptrend (higher value = stronger trend)
* Negative values: Strong downtrend (lower value = stronger trend)
* Near zero values: Weak/no trend (ranging market)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


class TrendStrengthAgent:
    def __init__(
        self,
        adx_period: int = 14,         # Period for ADX calculation
        di_period: int = 14,          # Period for DI calculation
        smoothing: int = 14,          # Smoothing period for ADX
        adx_threshold: float = 25.0,  # ADX threshold for trend strength
        strong_threshold: float = 40.0 # ADX threshold for very strong trend
    ):
        self.adx_period = adx_period
        self.di_period = di_period
        self.smoothing = smoothing
        self.adx_threshold = adx_threshold
        self.strong_threshold = strong_threshold
        self.latest_signal = 0.0
        self.latest_adx = 0.0
        self.latest_plus_di = 0.0
        self.latest_minus_di = 0.0
        
    def _calculate_dmi_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Directional Movement Index (DMI) and Average Directional Index (ADX)"""
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # Calculate True Range (TR)
        df_copy['high_minus_low'] = df_copy['high'] - df_copy['low']
        df_copy['high_minus_prev_close'] = abs(df_copy['high'] - df_copy['close'].shift(1))
        df_copy['low_minus_prev_close'] = abs(df_copy['low'] - df_copy['close'].shift(1))
        df_copy['tr'] = df_copy[['high_minus_low', 'high_minus_prev_close', 'low_minus_prev_close']].max(axis=1)
        
        # Calculate Directional Movement (DM)
        df_copy['up_move'] = df_copy['high'] - df_copy['high'].shift(1)
        df_copy['down_move'] = df_copy['low'].shift(1) - df_copy['low']
        
        # Positive and negative DM
        df_copy['+dm'] = np.where(
            (df_copy['up_move'] > df_copy['down_move']) & (df_copy['up_move'] > 0),
            df_copy['up_move'],
            0
        )
        df_copy['-dm'] = np.where(
            (df_copy['down_move'] > df_copy['up_move']) & (df_copy['down_move'] > 0),
            df_copy['down_move'],
            0
        )
        
        # Calculate smoothed TR and DM using Wilder's smoothing
        df_copy['tr_{}'.format(self.di_period)] = df_copy['tr'].rolling(window=self.di_period).sum()
        df_copy['+dm_{}'.format(self.di_period)] = df_copy['+dm'].rolling(window=self.di_period).sum()
        df_copy['-dm_{}'.format(self.di_period)] = df_copy['-dm'].rolling(window=self.di_period).sum()
        
        # Calculate +DI and -DI
        df_copy['+di'] = 100 * (df_copy['+dm_{}'.format(self.di_period)] / df_copy['tr_{}'.format(self.di_period)])
        df_copy['-di'] = 100 * (df_copy['-dm_{}'.format(self.di_period)] / df_copy['tr_{}'.format(self.di_period)])
        
        # Calculate direction index (DX)
        df_copy['di_diff'] = abs(df_copy['+di'] - df_copy['-di'])
        df_copy['di_sum'] = df_copy['+di'] + df_copy['-di']
        df_copy['dx'] = 100 * (df_copy['di_diff'] / df_copy['di_sum'])
        
        # Calculate ADX
        df_copy['adx'] = df_copy['dx'].rolling(window=self.smoothing).mean()
        
        return df_copy
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """
        Analyze historical data to identify the current trend regime
        """
        # Need enough bars for calculation
        if len(historical_df) < max(self.adx_period, self.di_period) + self.smoothing:
            self.latest_signal = 0.0
            return
        
        # Calculate DMI and ADX
        df = self._calculate_dmi_adx(historical_df)
        
        # Get the latest values
        self.latest_adx = df['adx'].iloc[-1]
        self.latest_plus_di = df['+di'].iloc[-1]
        self.latest_minus_di = df['-di'].iloc[-1]
        
        # Determine trend strength signal
        if np.isnan(self.latest_adx):
            self.latest_signal = 0.0
            return
        
        # Scale ADX to [0, 1] range using thresholds
        trend_strength = min((self.latest_adx - self.adx_threshold) / 
                             (self.strong_threshold - self.adx_threshold), 1.0)
        
        # If ADX below threshold, weak trend
        if self.latest_adx < self.adx_threshold:
            trend_strength = 0.0
        
        # Determine trend direction
        if self.latest_plus_di > self.latest_minus_di:
            # Uptrend
            self.latest_signal = trend_strength
        else:
            # Downtrend
            self.latest_signal = -trend_strength
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Predict the trend regime signal based on latest data
        Returns a float in the range [-1, 1] where:
          * Positive values indicate strong uptrend
          * Negative values indicate strong downtrend
          * Values near zero indicate weak/no trend
        """
        # Process the latest data
        self.fit(historical_df)
        
        return self.latest_signal
    
    def __str__(self) -> str:
        """String representation of the agent"""
        return "Trend Strength Agent (ADX)" 