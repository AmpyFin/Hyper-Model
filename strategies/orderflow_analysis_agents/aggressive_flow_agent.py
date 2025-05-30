"""
aggressive_flow_agent.py

Aggressive Flow Agent
~~~~~~~~~~~~~~~~~~~~~
Detects whether traders are aggressively hitting bids or lifting offers, a
common precursor to momentum ignition.

* Output range: -1 … +1 (>0 = aggressive buying, <0 = aggressive selling)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class AggressiveFlowAgent:
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        lookback_period: int = 50,        # bars for recent-flow stats
        aggression_threshold: float = 0.7, # (kept for compatibility)
        volume_weighted: bool = True,     # (kept for compatibility)
        flow_smoothing: int = 5,          # EWMA span for flow series
        signal_smoothing: int = 3,        # EWMA span for final signal
        use_proxy: bool = True,           # use price proxy if no quotes
    ) -> None:
        self.lookback_period = lookback_period
        self.aggression_threshold = aggression_threshold
        self.volume_weighted = volume_weighted
        self.flow_smoothing = flow_smoothing
        self.signal_smoothing = signal_smoothing
        self.use_proxy = use_proxy

        self.latest_signal: float = 0.0
        self.signal_history: list[float] = []
        self.flow_data: dict = {}

    # --------------------------------------------------------------------- #
    def _calculate_aggressive_flow(self, df: pd.DataFrame) -> Dict[str, float]:
        df = df.copy()
        has_quote = {"bid", "ask"}.issubset(df.columns)

        if has_quote:
            # Calculate price changes and spread
            df["price_chg"] = df["close"].diff().fillna(0)
            df["spread"] = df["ask"] - df["bid"]
            df["spread_pct"] = df["spread"] / df["close"]
            
            # Default to neutral
            df["trade_type"] = "neutral"
            
            # Identify significant price moves relative to spread
            sig_move_threshold = df["spread_pct"].mean() * 0.5  # 50% of avg spread
            
            # Aggressive buys: significant up moves or trades above midpoint with volume
            buy_mask = (
                (df["price_chg"] > 0) & 
                (abs(df["price_chg"]) > sig_move_threshold * df["close"])
            )
            df.loc[buy_mask, "trade_type"] = "aggr_buy"
            
            # Aggressive sells: significant down moves or trades below midpoint with volume
            sell_mask = (
                (df["price_chg"] < 0) & 
                (abs(df["price_chg"]) > sig_move_threshold * df["close"])
            )
            df.loc[sell_mask, "trade_type"] = "aggr_sell"

            # Debug print only for verification during development
            print("\nTrade classification summary:")
            type_counts = df["trade_type"].value_counts()
            print(type_counts)
            
            print("\nPrice change stats:")
            print(f"Mean spread %: {df['spread_pct'].mean()*100:.4f}%")
            print(f"Sig move threshold: {sig_move_threshold*100:.4f}%")
            print(f"Mean abs price change %: {(abs(df['price_chg'])/df['close']).mean()*100:.4f}%")
            
            print("\nSample classifications:")
            sample = df[["close", "bid", "ask", "price_chg", "trade_type"]].head()
            print(sample)

            df["agg_buy"]  = np.where(df["trade_type"] == "aggr_buy",  df["volume"], 0.0)
            df["agg_sell"] = np.where(df["trade_type"] == "aggr_sell", df["volume"], 0.0)
        else:
            # proxy: price change sign
            price_chg_sign = np.sign(df["close"].diff().fillna(0))
            df["agg_buy"]  = np.where(price_chg_sign > 0, df["volume"], 0.0)
            df["agg_sell"] = np.where(price_chg_sign < 0, df["volume"], 0.0)

        df["net_agg_vol"] = df["agg_buy"] - df["agg_sell"]

        # ---------------- recent-window stats -----------------------------
        window = min(self.lookback_period, len(df))  # Use full available data for small samples
        vol_sum = df["volume"].sum()  # Use total volume for small samples
        
        # normalised net aggression (-1 … +1)
        net_agg = df["net_agg_vol"].sum()  # Use total net aggression
        norm_agg = net_agg / vol_sum if vol_sum > 0 else 0.0

        latest = {
            "norm_agg": float(norm_agg),
            "agg_buy_ratio": float(df["agg_buy"].sum() / vol_sum if vol_sum > 0 else 0.0),
            "agg_sell_ratio": float(df["agg_sell"].sum() / vol_sum if vol_sum > 0 else 0.0)
        }

        latest["net_ratio"] = latest["agg_buy_ratio"] - latest["agg_sell_ratio"]
        
        # Debug print only for verification
        print(f"\nFlow metrics:")
        print(f"Net aggression: {norm_agg:.4f}")
        print(f"Buy ratio: {latest['agg_buy_ratio']:.4f}")
        print(f"Sell ratio: {latest['agg_sell_ratio']:.4f}")
        return latest

    # --------------------------------------------------------------------- #
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Update internal state from a historical OHLCV(+bid/ask) DataFrame."""
        if len(historical_df) < 10:
            self.latest_signal = 0.0
            return

        # Calculate flow metrics
        self.flow_data = self._calculate_aggressive_flow(historical_df)
        if not self.flow_data:
            self.latest_signal = 0.0
            return

        # ------------ build a composite signal ----------------------------
        comp, w = [], []

        # Normalized aggression component
        norm_agg_comp = np.tanh(self.flow_data["norm_agg"] * 3.0)
        comp.append(norm_agg_comp)
        w.append(0.6)

        # Net ratio component
        net_ratio_comp = np.tanh(self.flow_data["net_ratio"] * 4.0)
        comp.append(net_ratio_comp)
        w.append(0.4)

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
        self.fit(historical_df)
        return self.latest_signal

    # --------------------------------------------------------------------- #
    def __str__(self) -> str:
        if not self.flow_data:
            return "No data"
        return (
            f"norm={self.flow_data['norm_agg']:.2f} | "
            f"buy={self.flow_data['agg_buy_ratio']:.2%} | "
            f"sell={self.flow_data['agg_sell_ratio']:.2%} | "
            f"sig={self.latest_signal:.2f}"
        )
