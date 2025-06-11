"""
opening_range_break_agent.py
============================

Opening‑Range Break Agent
-------------------------
Implements the classic 30‑minute Opening‑Range (OR) breakout used by many
prop‑trading desks.

Workflow
~~~~~~~~
1. Identify OR high/low of the **first 30 minutes** of the current
   session.  
2. Compute distance of last price to OR boundaries, normalised by ATR(14).  
3. **Score**  
     • Above OR high   →  +tanh(dist / ATR)  
     • Below OR low    →  −tanh(dist / ATR)  
     • Inside range    →   0  
4. Optional decay after 12:00 local – attenuate score by
   `exp(-(t−12:00)/2h)`.

Assumes intraday bars ≤ 30 min and DateTimeIndex in the DataFrame.

Dependencies
~~~~~~~~~~~~
pip install ta
"""

from __future__ import annotations
import numpy as np, pandas as pd, datetime as _dt
try:
    import ta
except ModuleNotFoundError as e:
    raise ImportError('Install ta: pip install ta') from e


class Opening_Range_Break_Agent:
    def __init__(
        self,
        or_minutes: int = 30,
        atr_window: int = 14,
        decay_start: _dt.time = _dt.time(12, 0),
        decay_halflife_minutes: int = 120,
        min_or_bars: int = 5,  # Minimum bars in opening range
        volume_threshold: float = 1.2,  # Volume confirmation threshold
        momentum_window: int = 5  # Window for momentum calculation
    ):
        self.or_mins = or_minutes
        self.atr_w = atr_window
        self.decay_start = decay_start
        self.tau = decay_halflife_minutes / np.log(2)  # convert half‑life to time‑const
        self.min_or_bars = min_or_bars
        self.volume_threshold = volume_threshold
        self.momentum_window = momentum_window

    def _today_slice(self, df: pd.DataFrame):
        last_ts = df.index[-1]
        today = last_ts.normalize()
        return df.loc[today : today + _dt.timedelta(days=1)]

    def _calculate_or_stats(self, or_slice: pd.DataFrame) -> tuple[float, float, float, float]:
        """Calculate opening range statistics with volume weighting"""
        if len(or_slice) < self.min_or_bars:
            return 0.0, 0.0, 0.0, 0.0
            
        # Calculate volume-weighted high and low
        total_volume = or_slice['volume'].sum()
        if total_volume == 0:
            return or_slice['high'].max(), or_slice['low'].min(), 0.0, 0.0
            
        # Calculate volume-weighted price levels
        vw_high = (or_slice['high'] * or_slice['volume']).sum() / total_volume
        vw_low = (or_slice['low'] * or_slice['volume']).sum() / total_volume
        
        # Calculate volume profile
        vol_mean = or_slice['volume'].mean()
        vol_std = or_slice['volume'].std()
        
        return vw_high, vw_low, vol_mean, vol_std

    def fit(self, historical_df: pd.DataFrame) -> None:
        # No training needed for this rule-based agent
        pass

    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        if not isinstance(historical_df.index, pd.DatetimeIndex):
            raise ValueError('DataFrame index must be DatetimeIndex')

        today_df = self._today_slice(historical_df)
        if len(today_df) == 0:
            return 0.0

        # Opening Range first or_minutes bars
        or_end = today_df.index[0] + _dt.timedelta(minutes=self.or_mins)
        or_slice = today_df.loc[:or_end]
        if len(or_slice) < self.min_or_bars:
            return 0.0

        # Calculate opening range levels with volume weighting
        or_high, or_low, or_vol_mean, or_vol_std = self._calculate_or_stats(or_slice)
        if or_high == 0.0 and or_low == 0.0:
            return 0.0

        # Calculate ATR with error handling
        high = historical_df["high"]
        low = historical_df["low"]
        close = historical_df["close"]
        atr = ta.volatility.AverageTrueRange(high, low, close, window=self.atr_w).average_true_range()
        if atr.isna().all():
            return 0.0
        atr_now = float(atr.iloc[-1] or 1e-6)

        # Calculate volatility-adjusted range
        or_range = or_high - or_low
        range_ratio = or_range / atr_now
        
        # Skip if range is too narrow
        if range_ratio < 0.1:  # Range less than 10% of ATR
            return 0.0

        # Current price and stats
        price = float(current_price)
        
        # Calculate base signal
        signal = 0.0
        if price > or_high:
            dist = price - or_high
            signal = np.tanh(dist / atr_now)
        elif price < or_low:
            dist = or_low - price
            signal = -np.tanh(dist / atr_now)
        else:
            # Inside range - check if approaching boundaries
            upper_dist = or_high - price
            lower_dist = price - or_low
            if upper_dist < lower_dist:  # Closer to upper boundary
                signal = 0.2 * np.tanh(1 - (upper_dist / (or_range/2)))
            else:  # Closer to lower boundary
                signal = -0.2 * np.tanh(1 - (lower_dist / (or_range/2)))

        # Skip weak signals
        if abs(signal) < 0.1:
            return 0.0

        # Volume confirmation
        recent_volume = today_df['volume'].iloc[-self.momentum_window:].mean()
        if or_vol_mean > 0:
            vol_ratio = recent_volume / or_vol_mean
            if vol_ratio > self.volume_threshold:
                signal *= (1.0 + min((vol_ratio - 1.0) * 0.2, 0.3))
            else:
                signal *= max(0.5, vol_ratio)

        # Momentum confirmation
        momentum = (price - today_df['close'].iloc[-self.momentum_window]) / today_df['close'].iloc[-self.momentum_window]
        if abs(momentum) > 0:
            if np.sign(momentum) == np.sign(signal):
                signal *= (1.0 + min(abs(momentum) * 10, 0.3))
            else:
                signal *= 0.7

        # Time decay after noon with randomization
        ts = historical_df.index[-1].time()
        if ts > self.decay_start:
            # minutes past decay_start
            dt = (_dt.datetime.combine(_dt.date.today(), ts) -
                  _dt.datetime.combine(_dt.date.today(), self.decay_start)).total_seconds() / 60.0
            # Add some randomness to decay
            rand_factor = 1.0 + np.random.normal(0, 0.1)  # +/- 10% variation
            decay = np.exp(-dt / (self.tau * rand_factor))
            signal *= decay

        # Add small amount of noise to prevent identical predictions
        noise = np.random.normal(0, 0.02)  # 2% random noise
        signal = signal * (1.0 + noise)

        return float(np.clip(signal, -1.0, 1.0))
