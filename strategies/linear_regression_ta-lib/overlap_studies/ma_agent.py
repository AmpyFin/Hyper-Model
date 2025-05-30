"""
MA Agent
========

Classic simple moving-average (SMA, default 20-period)–based model.

Features
--------
* **MA Divergence** – (close − ma) / ma
* **MA Slope**      – 1-bar % slope of the MA
* **Price ROC-5**   – 5-bar rate-of-change of the close

Model
-----
LogisticRegression → probability price will rise next bar, mapped linearly to
**score ∈ [-1, 1]**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ────────────────────────── agent class ────────────────────────────────
class MA_Agent:
    """Logistic-regression learner on MA divergence, slope, and 5-bar ROC."""

    def __init__(self, period: int = 20):
        self.period = period
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # ------------------------ feature builder ------------------------- #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close'")

        df = df.copy()
        df["ma"] = df["close"].rolling(self.period).mean()
        df["ma_div"] = (df["close"] - df["ma"]) / df["ma"]
        df["ma_slope"] = df["ma"].pct_change()
        df["roc_5"] = df["close"].pct_change(periods=5)

        feats = ["ma_div", "ma_slope", "roc_5"]
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
            raise ValueError("Not enough rows to train MA_Agent.")

        X = df[["ma_div", "ma_slope", "roc_5"]][:-1]
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

        last = self._add_features(historical_df).iloc[-1:][
            ["ma_div", "ma_slope", "roc_5"]
        ]
        prob_up = float(self.model.predict_proba(last)[0, 1])
        return (prob_up * 2) - 1

