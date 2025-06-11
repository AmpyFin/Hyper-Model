"""
double_top_bottom_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Detects Double-Top and Double-Bottom reversal patterns and outputs a
continuous trading signal in **[-1.0, +1.0]**.

Key upgrades vs the initial version
-----------------------------------
* **Parametrised everything** – all magic numbers are constructor args.
* **Volatility-adaptive prominence** – peak detection scales with ATR.
* **Trend filter** – requires an up-trend before a Double-Top and a
  down-trend before a Double-Bottom (reduces false positives).
* **Flexible breakout window** – default 50 % of pattern duration.
* **Continuous volume quality** – rewards / penalises breakout volume
  smoothly.
* **Anticipatory signal** – emits a small ± signal when the pattern
  exists but the neckline has not yet broken.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class DTBConfig:
    window_size: int = 120                # bars inspected each call
    peak_distance: int = 5                # min distance between peaks
    peak_prominence_factor: float = 0.4   # × ATR for adaptive prominence (reduced from 0.8)
    level_tolerance: float = 0.005        # max % diff between peaks (reduced from 0.02)
    min_pattern_bars: int = 10
    max_pattern_bars: int = 120
    breakout_window_factor: float = 0.5   # search window after second peak
    breakout_threshold: float = 0.002     # min % move beyond neckline (reduced from 0.01)
    anticipate_scale: float = 0.25        # signal if pattern but no break
    use_volume: bool = True
    volume_cap: float = 1.5               # cap for volume multiplier
    min_trend_slope: float = 5e-4         # slope in price / bar (reduced from 1e-3)
    quality_weight: float = 0.3
    breakout_weight: float = 0.7
    trend_lookback: int = 20              # bars for trend slope


class DoubleTopBottomAgent:
    """Pattern-based reversal agent."""

    def __init__(self, **kwargs):
        self.cfg = DTBConfig(**kwargs)
        self.detected_patterns: List[Dict[str, Any]] = []
        self._last_bar_id: int | None = None  # crude memoisation

    # ------------------------------------------------------------------ #
    #  Low-level utilities
    # ------------------------------------------------------------------ #
    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return float(tr.tail(period).mean(skipna=True))

    def _trend_slope(self, series: pd.Series) -> float:
        y = series.values
        x = np.arange(len(y))
        if len(y) < 2:
            return 0.0
        # simple linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return float(slope / y.mean())  # normalised per-price unit

    # ------------------------------------------------------------------ #
    #  Peak / trough detection
    # ------------------------------------------------------------------ #
    def _find_peaks_troughs(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        atr = self._atr(df) or 1e-8
        prominence = self.cfg.peak_prominence_factor * atr

        highs = df["high"].values
        lows = df["low"].values

        peaks, _ = find_peaks(highs, distance=self.cfg.peak_distance,
                              prominence=prominence)
        troughs, _ = find_peaks(-lows, distance=self.cfg.peak_distance,
                                prominence=prominence)
                            
        
        
        return peaks.tolist(), troughs.tolist()

    # ------------------------------------------------------------------ #
    #  Pattern search helpers
    # ------------------------------------------------------------------ #
    def _double_top_patterns(self, df: pd.DataFrame,
                             peaks: List[int],
                             offset: int) -> List[Dict[str, Any]]:
        patterns: List[Dict[str, Any]] = []
        highs = df["high"].values
        lows = df["low"].values

        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                p1, p2 = peaks[i], peaks[j]
                sep = p2 - p1
                if not (self.cfg.min_pattern_bars <= sep <= self.cfg.max_pattern_bars):
                    continue

                level_diff = abs(highs[p1] - highs[p2]) / highs[p1]
                if level_diff > self.cfg.level_tolerance:
                    continue

                # trend filter – need prior up-trend
                slope = self._trend_slope(df["close"].iloc[max(0, p1 - self.cfg.trend_lookback):p1])
                if slope < self.cfg.min_trend_slope:
                    continue

                trough_segment = lows[p1:p2]
                trough_idx_rel = np.argmin(trough_segment) + p1
                neckline = lows[trough_idx_rel]
                height = ((highs[p1] + highs[p2]) / 2) - neckline
                height_pct = height / neckline

                symmetry = 1.0 - level_diff
                time_q = 1.0 - sep / self.cfg.max_pattern_bars
                geometry_q = np.clip(height_pct * 10.0, 0, 1)

                quality = symmetry * 0.4 + geometry_q * 0.4 + time_q * 0.2

                pattern = {
                    "type": "double_top",
                    "first_peak": p1 + offset,
                    "second_peak": p2 + offset,
                    "trough": trough_idx_rel + offset,
                    "neckline": neckline,
                    "height_pct": height_pct,
                    "quality": quality,
                }
                patterns.append(pattern)
        return patterns

    def _double_bottom_patterns(self, df: pd.DataFrame,
                                troughs: List[int],
                                offset: int) -> List[Dict[str, Any]]:
        patterns: List[Dict[str, Any]] = []
        highs = df["high"].values
        lows = df["low"].values

        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                t1, t2 = troughs[i], troughs[j]
                sep = t2 - t1
                if not (self.cfg.min_pattern_bars <= sep <= self.cfg.max_pattern_bars):
                    continue

                level_diff = abs(lows[t1] - lows[t2]) / lows[t1]
                if level_diff > self.cfg.level_tolerance:
                    continue

                # trend filter – need prior down-trend
                slope = self._trend_slope(df["close"].iloc[max(0, t1 - self.cfg.trend_lookback):t1])
                if slope > -self.cfg.min_trend_slope:
                    continue

                peak_segment = highs[t1:t2]
                peak_idx_rel = np.argmax(peak_segment) + t1
                neckline = highs[peak_idx_rel]
                height = neckline - ((lows[t1] + lows[t2]) / 2)
                height_pct = height / neckline

                symmetry = 1.0 - level_diff
                time_q = 1.0 - sep / self.cfg.max_pattern_bars
                geometry_q = np.clip(height_pct * 10.0, 0, 1)

                quality = symmetry * 0.4 + geometry_q * 0.4 + time_q * 0.2

                pattern = {
                    "type": "double_bottom",
                    "first_trough": t1 + offset,
                    "second_trough": t2 + offset,
                    "peak": peak_idx_rel + offset,
                    "neckline": neckline,
                    "height_pct": height_pct,
                    "quality": quality,
                }
                patterns.append(pattern)
        return patterns

    # ------------------------------------------------------------------ #
    #  Break-out evaluation
    # ------------------------------------------------------------------ #
    def _breakout(self, df: pd.DataFrame, pat: Dict[str, Any]) -> Tuple[bool, float]:
        """Return (breakout?, strength)."""
        if pat["type"] == "double_top":
            end_idx = pat["second_peak"]
            dir_ = -1
            pattern_len = pat["second_peak"] - pat["first_peak"]
            neckline = pat["neckline"]
            cmp = lambda c: c < neckline
            diff = lambda c: (neckline - c) / neckline
        else:
            end_idx = pat["second_trough"]
            dir_ = +1
            pattern_len = pat["second_trough"] - pat["first_trough"]
            neckline = pat["neckline"]
            cmp = lambda c: c > neckline
            diff = lambda c: (c - neckline) / neckline

        # Get the slice after the pattern end using integer location
        post_slice = df.iloc[end_idx:]
        limit = int(pattern_len * self.cfg.breakout_window_factor)
       
        
        for k, (_, row) in enumerate(post_slice.iterrows()):
            if k > limit:
                break
            if cmp(row["close"]):
                strength = diff(row["close"])
                if strength < self.cfg.breakout_threshold:
                    continue

                # volume multiplier
                if self.cfg.use_volume and "volume" in df.columns:
                    if pat["type"] == "double_top":
                        start_idx = pat["first_peak"]
                        end_pattern_idx = pat["second_peak"]
                    else:
                        start_idx = pat["first_trough"]
                        end_pattern_idx = pat["second_trough"]
                        
                    pat_slice = df.iloc[start_idx:end_pattern_idx + 1]
                    avg_vol = pat_slice["volume"].mean()
                    mult = np.clip(row["volume"] / avg_vol, 0.5, self.cfg.volume_cap)
                    strength *= mult

                return True, strength
                
        return False, 0.0

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Detect patterns in the latest window."""
        if historical_df.empty:
            self.detected_patterns = []
            return

        # crude memoisation: skip work if no new bars
        last_idx = int(historical_df.index[-1].toordinal()) if hasattr(historical_df.index[-1], "toordinal") else len(historical_df)  # type: ignore
        if last_idx == self._last_bar_id:
            return
        self._last_bar_id = last_idx

        window_df = historical_df.tail(self.cfg.window_size)
        offset = len(historical_df) - len(window_df)

        peaks, troughs = self._find_peaks_troughs(window_df)

        dtops = self._double_top_patterns(window_df, peaks, offset)
        dbots = self._double_bottom_patterns(window_df, troughs, offset)


        self.detected_patterns = []
        for pat in dtops + dbots:
            brk, brk_strength = self._breakout(historical_df, pat)
            pat["breakout"] = brk
            pat["breakout_strength"] = brk_strength
            if brk:
                self.detected_patterns.append(pat)

        # keep patterns without breakout for anticipatory mode
        if not self.detected_patterns and (dtops or dbots):
            best = sorted(dtops + dbots, key=lambda p: p["quality"], reverse=True)[0]
            best["breakout"] = False
            best["breakout_strength"] = 0.0
            self.detected_patterns.append(best)
        
        

    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Return a signal in [-1, +1]:
        * +ve  → bullish (double-bottom)
        * -ve  → bearish (double-top)
        """
        self.fit(historical_df)

        if not self.detected_patterns:
            return 0.0

        # use most recent pattern by second_peak / second_trough
        latest = max(
            self.detected_patterns,
            key=lambda p: p.get("second_peak", p.get("second_trough"))
        )

        q = latest["quality"]
        b = latest["breakout_strength"]
        cfg = self.cfg

        if latest["breakout"]:
            strength = cfg.quality_weight * q + cfg.breakout_weight * b
        else:
            # anticipatory: scale by distance to neckline
            neck = latest["neckline"]
            dist = abs(current_price - neck) / neck
            strength = cfg.anticipate_scale * (1.0 - dist) * q

        strength = np.clip(strength, 0.0, 1.0)
        return strength if latest["type"] == "double_bottom" else -strength

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #
    def parameters(self) -> Dict[str, Any]:
        """Return current hyper-parameters."""
        return asdict(self.cfg)

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.parameters().items())
        return f"DoubleTopBottomAgent({p})"

