"""
VWAP Pressure Agent
~~~~~~~~~~~~~~~~~~~
Scores how far the last close is from the intraday VWAP, with additional
factors like volume profile and price momentum. The z-scored distance is
passed through a non-linear function, with sensitivity to volume and trend.

Input : OHLCV DataFrame (any bar size).  Output ∈ [-1, +1].
"""

from __future__ import annotations
import numpy as np
import pandas as pd

class VWAP_Pressure_Agent:
    def __init__(
        self, 
        lookback: int = 15,     # Reduced for minute data
        vol_impact: float = 0.3, # Volume impact weight
        trend_impact: float = 0.2 # Trend impact weight
    ):
        self.lookback = lookback
        self.vol_impact = vol_impact
        self.trend_impact = trend_impact
        
    def _calculate_vwap(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Calculate VWAP and typical price series"""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        cumul_tp_vol = (typical_price * df["volume"]).cumsum()
        cumul_vol = df["volume"].cumsum()
        vwap = cumul_tp_vol / cumul_vol
        return vwap, typical_price
        
    def _calculate_volume_profile(self, df: pd.DataFrame, vwap: pd.Series) -> float:
        """Calculate volume profile score (-1 to 1)"""
        # Split volume into above and below VWAP
        above_vwap = df["close"] > vwap
        vol_above = df.loc[above_vwap, "volume"].sum()
        vol_below = df.loc[~above_vwap, "volume"].sum()
        total_vol = vol_above + vol_below
        
        if total_vol == 0:
            return 0.0
            
        # Calculate volume imbalance
        vol_score = (vol_below - vol_above) / total_vol
        return vol_score
        
    def _calculate_trend_score(self, df: pd.DataFrame, vwap: pd.Series) -> float:
        """Calculate trend score based on VWAP slope and price momentum"""
        # VWAP slope
        vwap_change = (vwap.iloc[-1] - vwap.iloc[0]) / vwap.iloc[0]
        
        # Price momentum (shorter window)
        price_momentum = (df["close"].iloc[-1] - df["close"].iloc[-3]) / df["close"].iloc[-3]
        
        # Combine with more weight on recent momentum
        trend_score = (vwap_change + 2 * price_momentum) / 3
        return np.tanh(trend_score * 100)  # Normalize to [-1, 1]
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        # No training needed for this rule-based agent
        pass
        
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        if len(historical_df) < self.lookback:
            return 0.0
            
        # Get recent window
        window = historical_df.iloc[-self.lookback:]
        
        # Calculate VWAP and typical price
        vwap, typical_price = self._calculate_vwap(window)
        current_vwap = vwap.iloc[-1]
        
        # Calculate distance from VWAP
        dist = current_price - current_vwap
        
        # Calculate historical distances for z-score
        historical_dists = window["close"] - vwap
        
        # Calculate adaptive standard deviation
        rolling_std = historical_dists.rolling(window=5).std()
        adaptive_std = np.maximum(rolling_std.mean(), historical_dists.std())
        
        # Z-score with minimum std dev
        z_score = dist / (adaptive_std if adaptive_std > 0 else 1e-6)
        
        # Calculate base signal (-1 when above VWAP, +1 when below)
        base_signal = -np.tanh(z_score / 2.0)  # Reduced sensitivity
        
        # Calculate volume profile
        vol_profile = self._calculate_volume_profile(window, vwap)
        
        # Calculate trend score
        trend_score = self._calculate_trend_score(window, vwap)
        
        # Combine signals
        # 1. Base signal from VWAP distance
        # 2. Volume profile confirmation
        # 3. Trend confirmation or contradiction
        signal = base_signal
        
        # Volume confirmation
        if np.sign(vol_profile) == np.sign(base_signal):
            signal *= (1.0 + self.vol_impact * abs(vol_profile))
        else:
            signal *= (1.0 - self.vol_impact * abs(vol_profile))
            
        # Trend adjustment
        if abs(trend_score) > 0.2:  # Only consider significant trends
            if np.sign(trend_score) == np.sign(signal):
                # Trend confirms signal
                signal *= (1.0 + self.trend_impact * abs(trend_score))
            else:
                # Trend contradicts signal
                signal *= (1.0 - self.trend_impact * abs(trend_score))
        
        # Add small mean reversion component for strong deviations
        if abs(z_score) > 2.0:
            mean_rev = -np.sign(z_score) * 0.1
            signal = 0.9 * signal + 0.1 * mean_rev
            
        return float(np.clip(signal, -1.0, 1.0))
