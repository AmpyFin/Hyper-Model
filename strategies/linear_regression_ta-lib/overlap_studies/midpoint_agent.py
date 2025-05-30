"""
MIDPOINT Agent
==============

Uses the rolling midpoint price:

    midpoint_t = ( max(high[window]) + min(low[window]) ) / 2

Default window = 14 bars.

Features
--------
* **Mid Divergence** – (close − midpoint) / midpoint
* **Mid Slope**      – % slope of the midpoint
* **Price ROC-5**    – 5-bar rate-of-change of the close

Model
-----
LogisticRegression (scikit-learn) → probability close_{t+1} > close_t  
Mapped linearly to **score ∈ [-1, 1]**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ───────────────────────── helper function ─────────────────────────────
def midpoint(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    max_high = high.rolling(period).max()
    min_low = low.rolling(period).min()
    return (max_high + min_low) / 2


# ───────────────────────────── agent class ─────────────────────────────
class MIDPOINT_Agent:
    """LogReg learner on midpoint divergence, slope, and 5-bar ROC."""

    def __init__(self, period: int = 14):
        self.period = period
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # ----------------------- feature engineering ---------------------- #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must include columns: {required}")

        df = df.copy()
        df["mid"] = midpoint(df["high"], df["low"], self.period)
        df["mid_div"] = (df["close"] - df["mid"]) / df["mid"]
        df["mid_slope"] = df["mid"].pct_change()
        df["roc_5"] = df["close"].pct_change(periods=5)

        feats = ["mid_div", "mid_slope", "roc_5"]
        df[feats] = (
            df[feats]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
        )
        return df.dropna(subset=feats)

    # --------------------------- training ----------------------------- #
    def fit(self, ohlcv: pd.DataFrame) -> None:
        df = self._add_features(ohlcv)
        if len(df) < self.period + 10:
            raise ValueError("Not enough rows to train MIDPOINT_Agent.")

        X = df[["mid_div", "mid_slope", "roc_5"]][:-1]
        y = (df["close"].shift(-1) > df["close"]).astype(int)[:-1]
        self.model.fit(X, y)
        self.fitted = True

    # ---------------------------- predict ----------------------------- #
    def predict(
        self,
        *,
        current_price: float,
        historical_df: pd.DataFrame,
    ) -> float:
        if not self.fitted:
            self.fit(historical_df)

        last = self._add_features(historical_df).iloc[-1:][
            ["mid_div", "mid_slope", "roc_5"]
        ]
        prob_up = float(self.model.predict_proba(last)[0, 1])
        return (prob_up * 2) - 1

