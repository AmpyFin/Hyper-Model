"""
cumulative_delta_agent.py

Cumulative Delta Agent
~~~~~~~~~~~~~~~~~~~~~~
Analyzes the cumulative difference between buying and selling volume over time,
a key order-flow indicator for detecting shifts in market dominance.

* Output range: -1 … +1 (>0 = buying pressure, <0 = selling pressure)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class CumulativeDeltaAgent:
    # --------------------------------------------------------------------- #
    # INITIALISATION                                                        #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        lookback_period: int = 100,        # bars used for rolling stats
        signal_period: int = 20,           # bars used for smoothing
        divergence_threshold: float = 0.5, # (kept for compatibility)
        delta_smoothing: int = 5,          # EWMA span for raw delta
        signal_smoothing: int = 3,         # EWMA span for final signal
        use_delta_slope: bool = True,      # (kept for compatibility)
        volume_scale: float = 1.0,         # normalisation factor
    ) -> None:
        self.lookback_period = lookback_period
        self.signal_period = signal_period
        self.divergence_threshold = divergence_threshold
        self.delta_smoothing = delta_smoothing
        self.signal_smoothing = signal_smoothing
        self.use_delta_slope = use_delta_slope
        self.volume_scale = volume_scale

        self.latest_signal: float = 0.0
        self.signal_history: list[float] = []
        self.delta_data: dict = {}

    # --------------------------------------------------------------------- #
    # PRIVATE UTILITIES                                                     #
    # --------------------------------------------------------------------- #
    def _calculate_cumulative_delta(self, df: pd.DataFrame) -> Dict[str, float]:
        """Return a dict with cumulative-delta-based features."""
        df = df.copy()

        # -------- trade classification (bid/ask preferred) ----------------
        has_quote = {"bid", "ask"}.issubset(df.columns)
        
        if has_quote:
            # Classify trades based on execution price vs bid/ask
            df["trade_type"] = "neutral"  # default
            
            # Aggressive buys: trades at or above ask
            df.loc[df["close"] >= df["ask"], "trade_type"] = "aggr_buy"
            
            # Aggressive sells: trades at or below bid
            df.loc[df["close"] <= df["bid"], "trade_type"] = "aggr_sell"

            # Calculate delta based on trade type
            df["delta"] = np.where(df["trade_type"] == "aggr_buy", df["volume"],
                                 np.where(df["trade_type"] == "aggr_sell", -df["volume"], 0.0))
        else:
            # fall-back proxy using price change sign
            price_chg = df["close"].diff().fillna(0)
            df["delta"] = np.sign(price_chg) * df["volume"]

        # -------- core cumulative-delta maths -----------------------------
        df["cum_delta"] = df["delta"].cumsum()
        
        # Calculate delta stats using total values
        total_delta = df["delta"].sum()
        mean_delta = df["delta"].mean()
        std_delta = df["delta"].std()
        
        # Z-score based on total stats
        z_score = (total_delta - mean_delta) / std_delta if std_delta > 0 else 0.0

        # Calculate ROC using percentage changes
        price_roc = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) if len(df) > 1 else 0.0
        
        # Calculate delta ROC using absolute changes to avoid division by zero
        first_cum_delta = df["cum_delta"].iloc[0]
        last_cum_delta = df["cum_delta"].iloc[-1]
        delta_roc = (last_cum_delta - first_cum_delta) / (abs(first_cum_delta) + 1.0)  # Add 1.0 to avoid division by zero
        
        # Calculate divergence
        divergence = delta_roc - price_roc

        result = {
            "cum_delta": float(last_cum_delta),
            "delta_z": float(z_score),
            "delta_roc": float(delta_roc),
            "divergence": float(divergence)
        }
        return result

    # --------------------------------------------------------------------- #
    # PUBLIC API                                                            #
    # --------------------------------------------------------------------- #
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Update internal state from a historical OHLCV(+bid/ask) DataFrame."""
        if len(historical_df) < 10:
            self.latest_signal = 0.0
            return

        # Calculate delta metrics
        self.delta_data = self._calculate_cumulative_delta(historical_df)
        if not self.delta_data:
            self.latest_signal = 0.0
            return

        # ------------ build a composite signal ----------------------------
        comp, w = [], []

        # Z-score component
        z_score_comp = np.tanh(self.delta_data["delta_z"] * 0.6)
        comp.append(z_score_comp)
        w.append(0.4)

        # Delta ROC component
        roc_comp = np.tanh(self.delta_data["delta_roc"] * 5.0)
        comp.append(roc_comp)
        w.append(0.3)

        # Divergence component
        div_comp = np.tanh(self.delta_data["divergence"] * 3.0)
        comp.append(div_comp)
        w.append(0.3)

        raw = float(np.dot(comp, np.array(w) / sum(w)))

        # ------------ smooth it (adaptive EWMA length) --------------------
        self.signal_history.append(raw)
        max_hist = max(3, self.signal_smoothing)
        self.signal_history = self.signal_history[-max_hist:]

        weights = np.exp(np.linspace(-2.0, 0.0, len(self.signal_history)))
        weights /= weights.sum()

        smoothed = float(np.tanh(np.dot(self.signal_history, weights) * 2.0))
        self.latest_signal = smoothed

    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """Return the latest signal in the range -1…+1."""
        self.fit(historical_df)
        return self.latest_signal

    # --------------------------------------------------------------------- #
    def __str__(self) -> str:
        if not self.delta_data:
            return "No data"
        return (
            f"Δcum={self.delta_data['cum_delta']:.0f} | "
            f"z={self.delta_data['delta_z']:.2f} | "
            f"roc={self.delta_data['delta_roc']:.2%} | "
            f"div={self.delta_data['divergence']:.2%} | "
            f"sig={self.latest_signal:.2f}"
        )
