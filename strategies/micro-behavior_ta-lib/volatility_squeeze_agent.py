"""
volatility_squeeze_agent.py
===========================

Volatility‑Squeeze Agent
------------------------
Detects volatility contraction (squeeze) using multiple indicators and scores
breakouts with volume and momentum confirmation. Optimized for intraday trading.

Logic
~~~~~
1. Primary squeeze detection:
   - Bollinger Bands (adaptive window, 2σ)
   - Keltner Channels (adaptive window)
   - ATR-based volatility analysis
2. Squeeze metrics:
   - BB/KC width ratio
   - Volatility percentile
   - Volume profile
3. Breakout scoring:
   - Price momentum
   - Volume surge
   - Trend confirmation
4. Signal scaling based on:
   - Squeeze intensity
   - Breakout strength
   - Volume confirmation

Output ∈ [‑1, +1].

Dependencies
~~~~~~~~~~~~
pip install ta
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

try:
    import ta
except ModuleNotFoundError as e:
    raise ImportError('Install ta: pip install ta') from e


class Volatility_Squeeze_Agent:
    def __init__(
        self,
        bb_window: int = 12,        # Reduced for minute data
        kc_window: int = 12,        # Keltner Channel window
        atr_window: int = 10,       # Reduced for minute data
        lookback_mult: int = 5,     # Reduced multiplier for minute data
        squeeze_threshold: float = 0.85,  # Threshold for squeeze detection
        vol_impact: float = 0.3     # Volume impact on signal
    ):
        self.bb_w = bb_window
        self.kc_w = kc_window
        self.atr_w = atr_window
        self.lb_mult = lookback_mult
        self.squeeze_thresh = squeeze_threshold
        self.vol_impact = vol_impact

    def _calculate_bands(
        self, 
        df: pd.DataFrame,
        volatility: float
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands and Keltner Channels"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Adaptive window based on volatility
        bb_window = max(int(self.bb_w * (1 + volatility)), 8)
        kc_window = max(int(self.kc_w * (1 + volatility)), 8)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=close, 
            window=bb_window,
            window_dev=2
        )
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_mid = bb.bollinger_mavg()
        
        # Calculate ATR for Keltner Channels
        atr = ta.volatility.AverageTrueRange(
            high=high,
            low=low,
            close=close,
            window=self.atr_w
        ).average_true_range()
        
        # Calculate Keltner Channels manually for better control
        typical_price = (high + low + close) / 3
        kc_mid = typical_price.rolling(window=kc_window).mean()
        kc_upper = kc_mid + (1.5 * atr)  # Standard multiplier
        kc_lower = kc_mid - (1.5 * atr)
        
        return bb_upper, bb_lower, bb_mid, kc_upper, kc_lower

    def _detect_squeeze(
        self,
        df: pd.DataFrame,
        bb_upper: pd.Series,
        bb_lower: pd.Series,
        bb_mid: pd.Series,
        kc_upper: pd.Series,
        kc_lower: pd.Series
    ) -> Tuple[bool, float]:
        """Detect squeeze condition and calculate intensity"""
        # Calculate normalized widths
        bb_width = (bb_upper - bb_lower) / bb_mid
        kc_width = (kc_upper - kc_lower) / bb_mid
        
        # Calculate rolling medians with minimum periods
        bb_width_med = bb_width.rolling(
            window=self.bb_w * self.lb_mult,
            min_periods=self.bb_w
        ).median()
        kc_width_med = kc_width.rolling(
            window=self.kc_w * self.lb_mult,
            min_periods=self.kc_w
        ).median()
        
        # Get latest values with safe fallbacks
        bb_now = float(bb_width.iloc[-1])
        bb_med = float(bb_width_med.iloc[-1] or bb_now)
        kc_now = float(kc_width.iloc[-1])
        kc_med = float(kc_width_med.iloc[-1] or kc_now)
        
        # Calculate relative widths
        bb_rel = bb_now / bb_med if bb_med > 0 else 1.0
        kc_rel = kc_now / kc_med if kc_med > 0 else 1.0
        
        # Detect squeeze conditions
        bb_squeeze = bb_rel < self.squeeze_thresh
        kc_squeeze = kc_rel < self.squeeze_thresh
        
        # Calculate squeeze intensity (0 to 1)
        bb_intensity = max(0.0, 1.0 - bb_rel / self.squeeze_thresh)
        kc_intensity = max(0.0, 1.0 - kc_rel / self.squeeze_thresh)
        intensity = (bb_intensity + kc_intensity) / 2
        
        return (bb_squeeze and kc_squeeze), intensity

    def _calculate_breakout_score(
        self,
        df: pd.DataFrame,
        current_price: float,
        bb_upper: pd.Series,
        bb_lower: pd.Series,
        squeeze_intensity: float
    ) -> Tuple[float, float]:
        """Calculate breakout score and momentum"""
        close = df['close']
        
        # Get latest values
        upper_now = float(bb_upper.iloc[-1])
        lower_now = float(bb_lower.iloc[-1])
        mid_price = (upper_now + lower_now) / 2
        
        # Calculate normalized distance from bands
        band_width = upper_now - lower_now
        if band_width > 0:
            # Normalize distance by band width
            if current_price > upper_now:
                dist = (current_price - upper_now) / band_width
                sign = 1
            elif current_price < lower_now:
                dist = (lower_now - current_price) / band_width
                sign = -1
            else:
                # Price inside bands
                dist = abs(current_price - mid_price) / (band_width / 2)
                sign = 1 if current_price > mid_price else -1
                # Reduce signal strength for non-breakouts
                dist *= 0.5
        else:
            # Fallback if bands are identical
            dist = 0
            sign = 0
            
        # Calculate momentum over adaptive lookback
        lookback = max(3, min(int(5 * (1 - squeeze_intensity)), 8))
        if len(close) > lookback:
            momentum = (current_price - close.iloc[-lookback]) / close.iloc[-lookback]
        else:
            momentum = 0
            
        # Scale distance by squeeze intensity
        score = sign * dist * (0.5 + 0.5 * squeeze_intensity)
        
        return score, momentum

    def _calculate_volume_score(self, df: pd.DataFrame) -> float:
        """Calculate volume confirmation score"""
        if 'volume' not in df.columns or len(df) < 10:
            return 0.0
            
        volume = df['volume']
        close = df['close']
        
        # Calculate adaptive lookback based on data
        recent_window = min(3, len(volume) - 1)
        base_window = min(7, len(volume) - recent_window - 1)
        
        if recent_window <= 0 or base_window <= 0:
            return 0.0
        
        # Recent volume vs baseline
        recent_vol = volume.iloc[-recent_window:].mean()
        base_vol = volume.iloc[-base_window-recent_window:-recent_window].mean()
        
        # Volume surge ratio with safety check
        if base_vol > 0:
            vol_ratio = (recent_vol / base_vol) - 1
        else:
            vol_ratio = 0
            
        # Price direction
        if len(close) >= recent_window + 1:
            price_change = (close.iloc[-1] - close.iloc[-recent_window-1]) / close.iloc[-recent_window-1]
        else:
            price_change = 0
        
        # Volume score considers both surge and price direction
        vol_score = np.tanh(vol_ratio) * np.sign(price_change)
        
        return vol_score

    def fit(self, historical_df: pd.DataFrame) -> None:
        # No training needed for this rule-based agent
        pass

    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        if len(historical_df) < self.bb_w * self.lb_mult + 2:
            return 0.0
            
        # Calculate base volatility
        returns = historical_df['close'].pct_change()
        volatility = float(returns.std())
        
        # Calculate bands
        bb_upper, bb_lower, bb_mid, kc_upper, kc_lower = self._calculate_bands(
            historical_df, volatility
        )
        
        # Detect squeeze
        is_squeeze, squeeze_intensity = self._detect_squeeze(
            historical_df, bb_upper, bb_lower, bb_mid, kc_upper, kc_lower
        )
        
        # Calculate breakout score and momentum
        base_score, momentum = self._calculate_breakout_score(
            historical_df, current_price, bb_upper, bb_lower, squeeze_intensity
        )
        
        # Calculate volume confirmation
        vol_score = self._calculate_volume_score(historical_df)
        
        # Start with base score
        signal = base_score
        
        # Scale signal based on squeeze presence
        if is_squeeze:
            # Amplify signal in squeeze
            signal *= (1.0 + squeeze_intensity)
        else:
            # Reduce signal outside squeeze
            signal *= 0.3
            
        # Add momentum impact (reduced from previous version)
        signal *= (1.0 + np.sign(signal) * min(abs(momentum) * 2, 0.3))
        
        # Add volume confirmation (if volume agrees with signal)
        if np.sign(vol_score) == np.sign(signal):
            signal *= (1.0 + self.vol_impact * abs(vol_score))
        else:
            signal *= (1.0 - self.vol_impact * abs(vol_score))
            
        return float(np.clip(signal, -1.0, 1.0))
