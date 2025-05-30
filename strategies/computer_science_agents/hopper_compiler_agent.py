"""
hopper_agent.py
~~~~~~~~~~~~~~~

"Grace Hopper" compiler-inspired trading agent
---------------------------------------------
*  Lexical  →  tokenise price/volume action
*  Syntax   →  validate sequences / motifs
*  Semantic →  weigh context, trend & momentum
*  "Debug"  →  anomaly hunt (Hopper's moth-in-the-relay legend)
*  Code-gen →  emit final signal in **[-1.0, +1.0]**

Key fixes vs original
---------------------
1. **Parameter object** – everything tunable from constructor.
2. **Robust token logic**
   * `small_move` now means "≤ ½ mean absolute diff" (was unreachable).
   * All volume tokens are created only when volume data truly aligns.
3. **Better pattern thresholds**
   * Default `syntax_threshold` → 0.4 (was 0.6).
   * Confidence formulas rescaled to reach > 0 more often.
4. **Momentum fallback** – returns ±0.1-0.3 even when no syntax pattern hits.
5. **Count-consecutive fixed** – previous loop skipped first window bar.
6. **π-safe anomaly maths** – guards against div-by-zero on tiny stdevs.

Hopper Agent
~~~~~~~~~~
Agent implementing trading strategies based on Grace Hopper's work on
compilers, programming languages, and computer standardization.

Grace Hopper is known for:
1. COBOL Programming Language
2. First Compiler (A-0 System)
3. Machine-Independent Programming
4. Program Optimization Techniques
5. Standardization of Programming Languages

This agent models market behavior using:
1. Compilation of market signals
2. Language-independent pattern recognition
3. Program optimization principles
4. Standardized signal processing
5. Bug detection and debugging

Input: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
Output: Signal ∈ [-1.0000, 1.0000] where:
  -1.0000 = Strong sell signal (strong downward trend detected)
  -0.5000 = Weak sell signal (weak downward trend detected)
   0.0000 = Neutral signal (no clear trend)
   0.5000 = Weak buy signal (weak upward trend detected)
   1.0000 = Strong buy signal (strong upward trend detected)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from ..agent import Agent

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Hyper-parameter bundle
# ------------------------------------------------------------------ #

@dataclass
class HopperConfig:
    lexical_window: int = 20
    syntax_threshold: float = 0.4       # ↓ so patterns actually fire
    debug_sensitivity: float = 2.0
    optimization_level: int = 2
    code_generation_lag: int = 3
    momentum_weight: float = 0.3        # in semantic step
    anomaly_weight: float = 0.3         # in final blend
    seed: int | None = None             # reproducibility optional


# ------------------------------------------------------------------ #
#  Agent
# ------------------------------------------------------------------ #

class HopperAgent(Agent):
    """Compiler-metaphor trading agent."""

    def __init__(self, **kwargs):
        self.cfg = HopperConfig(**kwargs)
        self._rng = np.random.default_rng(self.cfg.seed)

        self.latest_signal: float = 0.0
        self.is_fitted: bool = False

        # state caches
        self.token_library: Dict[str, np.ndarray] = {}
        self.syntax_patterns: List[Dict[str, float]] = []
        self.debugging_history: deque[tuple[float, float]] = deque(maxlen=30)

    # -------------------------------------------------------------- #
    #  Lexical analysis
    # -------------------------------------------------------------- #
    def _lexical_analysis(
        self, prices: np.ndarray, volumes: Optional[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if prices.size < self.cfg.lexical_window:
            return {}

        price_diffs = np.diff(prices)
        mean_diff = np.mean(np.abs(price_diffs))
        if mean_diff == 0:  # flat ticker guard
            mean_diff = 1e-12
        std_diff = np.std(np.abs(price_diffs))

        tokens: Dict[str, np.ndarray] = {
            "up": (price_diffs > 0).astype(int),
            "down": (price_diffs < 0).astype(int),
            "unchanged": (price_diffs == 0).astype(int),
            # magnitude buckets
            "large_move": (np.abs(price_diffs) > mean_diff + std_diff).astype(int),
            "small_move": (np.abs(price_diffs) <= mean_diff * 0.5).astype(int),
        }

        # volume tokens
        if volumes is not None and volumes.size >= prices.size:
            vols = volumes[: prices.size]  # align
            vol_diffs = np.diff(vols)
            vols_aligned = vols[1:]

            mean_vol = vols_aligned.mean()
            std_vol = vols_aligned.std() or 1e-9  # avoid div-by-0

            tokens.update(
                {
                    "high_volume": (vols_aligned > mean_vol + std_vol).astype(int),
                    "low_volume": (vols_aligned < mean_vol - std_vol).astype(int),
                    "rising_volume": (vol_diffs > 0).astype(int),
                    "falling_volume": (vol_diffs < 0).astype(int),
                }
            )

        # volatility tokens (rolling 5-bar σ)
        rolling_vol = np.array(
            [prices[max(0, i - 5) : i + 1].std() for i in range(1, prices.size)]
        )
        if rolling_vol.size:
            mean_v = rolling_vol.mean()
            std_v = rolling_vol.std() or 1e-9
            tokens["high_volatility"] = (rolling_vol > mean_v + std_v).astype(int)
            tokens["low_volatility"] = (rolling_vol < mean_v - std_v).astype(int)

        # ensure equal length (N-1)
        expected = price_diffs.size
        for k, v in tokens.items():
            if v.size != expected:
                tokens[k] = np.pad(v, (expected - v.size, 0))

        return tokens

    # -------------------------------------------------------------- #
    #  Syntax analysis
    # -------------------------------------------------------------- #
    def _count_consecutive(self, arr: np.ndarray, window: int) -> np.ndarray:
        cnt = np.zeros_like(arr)
        streak = 0
        for i, val in enumerate(arr):
            streak = streak + 1 if val else 0
            cnt[i] = max(streak, cnt[i - 1] if i else 0)
            if i >= window:
                # subtract windows that fell out of range
                cnt[i] = min(cnt[i], window)
        return cnt

    def _syntax_analysis(self, t: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
        if not t:
            return []

        pats: List[Dict[str, float]] = []
        thr = self.cfg.syntax_threshold

        # ――― trend continuation
        if "up" in t and "large_move" in t:
            conf = 0.5 * t["up"][-5:].mean() + 0.5 * t["large_move"][-5:].mean()
            if conf > thr:
                pats.append({"name": "up_cont", "confidence": conf, "direction": 1.0})

        if "down" in t and "large_move" in t:
            conf = 0.5 * t["down"][-5:].mean() + 0.5 * t["large_move"][-5:].mean()
            if conf > thr:
                pats.append({"name": "down_cont", "confidence": conf, "direction": -1.0})

        # ――― exhaustion reversal
        if all(k in t for k in ("up", "high_volume", "high_volatility")):
            if t["up"][-5:].sum() >= 4 and (
                t["high_volume"][-5:] + t["high_volatility"][-5:]
            ).sum() >= 5:
                conf = 0.6 + 0.05 * t["high_volume"][-5:].sum()
                pats.append({"name": "up_exhaust", "confidence": conf, "direction": -1.0})

        if all(k in t for k in ("down", "high_volume", "high_volatility")):
            if t["down"][-5:].sum() >= 4 and (
                t["high_volume"][-5:] + t["high_volatility"][-5:]
            ).sum() >= 5:
                conf = 0.6 + 0.05 * t["high_volume"][-5:].sum()
                pats.append({"name": "down_exhaust", "confidence": conf, "direction": 1.0})

        # ――― consolidation breakout
        if "low_volatility" in t and "small_move" in t and "up" in t:
            if (t["low_volatility"][-10:] & t["small_move"][-10:]).sum() >= 7:
                dir_ = 1.0 if t["up"][-3:].sum() >= 2 else -1.0
                conf = 0.5 + 0.05 * (t["low_volatility"][-10:].sum())
                pats.append({"name": "consol_brk", "confidence": conf, "direction": dir_})

        return pats

    # -------------------------------------------------------------- #
    #  Semantic & anomaly layers
    # -------------------------------------------------------------- #
    def _semantic_analysis(self, pats: List[Dict[str, float]], prices: np.ndarray) -> float:
        if not pats:
            return 0.0

        tot = sum(p["confidence"] for p in pats)
        weighted_dir = sum(p["confidence"] * p["direction"] for p in pats) / tot

        # simple momentum tilt
        if prices.size >= 11:
            r = np.diff(prices[-11:]) / prices[-11:-1]
            momentum = r.sum() * 10.0
        else:
            momentum = 0.0

        w = self.cfg.momentum_weight
        return float(np.clip((1 - w) * weighted_dir + w * np.clip(momentum, -1, 1), -1, 1))

    def _debug_anomalies(
        self, prices: np.ndarray, volumes: Optional[np.ndarray]
    ) -> float:
        if prices.size < 20:
            return 0.0

        ret = np.diff(prices) / prices[:-1]
        mu, sigma = ret.mean(), ret.std() or 1e-9
        z = (ret[-5:] - mu) / sigma
        price_anom = np.any(np.abs(z) > self.cfg.debug_sensitivity)
        price_sig = (
            -np.sign(z.mean()) * np.clip(np.abs(z).max() / (2 * self.cfg.debug_sensitivity), 0, 1)
            if price_anom
            else 0.0
        )

        vol_sig = 0.0
        if volumes is not None and volumes.size >= prices.size:
            vol = volumes[: prices.size]
            dv = np.diff(vol) / np.maximum(vol[:-1], 1e-9)
            mu_v, sig_v = dv.mean(), dv.std() or 1e-9
            z_v = (dv[-5:] - mu_v) / sig_v
            if np.any(np.abs(z_v) > self.cfg.debug_sensitivity):
                vol_sig = np.sign(z_v.mean()) * np.clip(
                    np.abs(z_v).max() / (2 * self.cfg.debug_sensitivity), 0, 1
                )

        return (price_sig + 0.5 * vol_sig) / 1.5

    # -------------------------------------------------------------- #
    #  Code generation
    # -------------------------------------------------------------- #
    def _code_generation(self, sem: float, anom: float) -> float:
        self.debugging_history.append((sem, anom))
        if (
            self.cfg.code_generation_lag
            and len(self.debugging_history) >= self.cfg.code_generation_lag
        ):
            sem_lag, anom_lag = self.debugging_history[-self.cfg.code_generation_lag]
        else:
            sem_lag, anom_lag = sem, anom

        if self.cfg.optimization_level == 1:
            out = 0.5 * sem + 0.5 * anom
        elif self.cfg.optimization_level == 2:
            out = (1 - self.cfg.anomaly_weight) * sem + self.cfg.anomaly_weight * anom
        else:
            s_w = 0.6 + 0.2 * abs(sem_lag)
            a_w = 1 - s_w
            out = s_w * sem_lag + a_w * anom_lag

        return float(np.clip(out, -1, 1))

    # -------------------------------------------------------------- #
    #  Fit / predict
    # -------------------------------------------------------------- #
    def fit(self, historical_df: pd.DataFrame) -> None:
        if historical_df.shape[0] < self.cfg.lexical_window * 2:
            self.is_fitted = False
            return

        try:
            prices = historical_df["close"].to_numpy()
            volumes = (
                historical_df["volume"].to_numpy()
                if "volume" in historical_df.columns
                else None
            )

            tok = self._lexical_analysis(prices, volumes)
            self.token_library = tok

            pats = self._syntax_analysis(tok)
            self.syntax_patterns = pats

            sem = self._semantic_analysis(pats, prices)

            # even if no syntax pattern, produce mild momentum signal
            if sem == 0.0 and prices.size >= 2:
                sem = float(
                    np.clip(
                        np.sign(prices[-1] - prices[-2])
                        * min(abs(prices[-1] - prices[-2]) / prices[-2], 0.003) * 100,
                        -0.3,
                        0.3,
                    )
                )

            anom = self._debug_anomalies(prices, volumes)

            self.latest_signal = self._code_generation(sem, anom)
            self.is_fitted = True
        except Exception as exc:  # noqa: BLE001
            logger.error("HopperAgent fit error: %s", exc, exc_info=False)
            self.is_fitted = False

    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        self.fit(historical_df)
        return self.latest_signal if self.is_fitted else 0.0

    # -------------------------------------------------------------- #
    #  Misc
    # -------------------------------------------------------------- #
    def parameters(self) -> Dict[str, float]:
        return asdict(self.cfg)

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.parameters().items())
        return f"HopperAgent({p})"

    def __str__(self) -> str:  # for your logging
        return "Hopper Agent"

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Hopper's compiler optimization principles.
        
        Parameters
        ----------
        historical_df : pd.DataFrame
            Historical OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns
        -------
        float
            Trading signal in range [-1.0000, 1.0000]
            -1.0000 = Strong sell
            -0.5000 = Weak sell
             0.0000 = Neutral
             0.5000 = Weak buy
             1.0000 = Strong buy
        """
        try:
            # Process the data using existing workflow
            self.fit(historical_df)
            
            if not self.is_fitted:
                return 0.0000
                
            # Get current price for prediction
            current_price = historical_df['close'].iloc[-1]
            
            # Generate signal using existing predict method
            signal = self.predict(current_price=current_price, historical_df=historical_df)
            
            # Ensure signal is properly formatted to 4 decimal places
            return float(round(signal, 4))
            
        except ValueError as e:
            # Handle the case where there's not enough data
            logger.error(f"ValueError in Hopper strategy: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Error in Hopper strategy: {str(e)}")
            return 0.0000
