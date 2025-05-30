"""
DEMA Agent
==========

Uses the Double Exponential Moving Average (DEMA, period 20 by default)
to learn price direction.

Features
--------
* **Divergence**   – (close − dema) / dema
* **DEMA Slope**   – 1-bar pct slope of the DEMA itself
* **Price ROC-5**  – 5-bar rate-of-change for extra momentum context

Model
-----
LogisticRegression (scikit-learn) → probability price will rise next bar,
mapped linearly to **score ∈ [-1, 1]**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ───────────────────────── indicator helper ────────────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def dema(series: pd.Series, period: int = 20) -> pd.Series:
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    return 2 * ema1 - ema2


# ───────────────────────────── agent class ──────────────────────────────
class DEMA_Agent:
    """LogReg learner on DEMA divergence, slope, and 5-period ROC."""

    def __init__(self, period: int = 20):
        self.period = period
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # ----------------------- feature engineering ----------------------- #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        df = df.copy()
        df["dema"] = dema(df["close"], self.period)

        # Divergence: relative difference
        df["dema_div"] = (df["close"] - df["dema"]) / df["dema"]

        # DEMA slope (trend proxy)
        df["dema_slope"] = df["dema"].pct_change()

        # Price momentum context
        df["roc_5"] = df["close"].pct_change(periods=5)

        feats = ["dema_div", "dema_slope", "roc_5"]
        df[feats] = (
            df[feats]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
        )

        return df.dropna(subset=feats)

    # ---------------------------- train ------------------------------- #
    def fit(self, ohlcv: pd.DataFrame) -> None:
        df = self._add_features(ohlcv)
        if len(df) < 30:
            raise ValueError("Need at least 30 rows to train DEMA_Agent.")

        X = df[["dema_div", "dema_slope", "roc_5"]][:-1]
        y = (df["close"].shift(-1) > df["close"]).astype(int)[:-1]

        self.model.fit(X, y)
        self.fitted = True

    # --------------------------- predict ------------------------------ #
    def predict(
        self,
        *,
        current_price: float,
        historical_df: pd.DataFrame,
    ) -> float:
        if not self.fitted:
            self.fit(historical_df)

        last_row = self._add_features(historical_df).iloc[-1:][
            ["dema_div", "dema_slope", "roc_5"]
        ]

        prob_up = float(self.model.predict_proba(last_row)[0, 1])
        return (prob_up * 2) - 1

