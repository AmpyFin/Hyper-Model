"""
EMA Agent
=========

Uses a single-period Exponential Moving Average (EMA, default 20)
to model price direction.

Features
--------
* **EMA Divergence** – (close − ema) / ema
* **EMA Slope**      – 1-bar pct slope of the EMA itself
* **Price ROC-5**    – 5-bar rate-of-change of the close

Model
-----
LogisticRegression → probability of an up move next bar, mapped to
**score ∈ [-1, 1]**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ───────────────────────── indicator helper ────────────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ───────────────────────────── agent class ──────────────────────────────
class EMA_Agent:
    """LogReg learner on EMA divergence, slope, and 5-bar ROC."""

    def __init__(self, period: int = 20):
        self.period = period
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # ----------------------- feature engineering ----------------------- #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        df = df.copy()
        df["ema"] = ema(df["close"], self.period)
        df["ema_div"] = (df["close"] - df["ema"]) / df["ema"]
        df["ema_slope"] = df["ema"].pct_change()
        df["roc_5"] = df["close"].pct_change(periods=5)

        feats = ["ema_div", "ema_slope", "roc_5"]
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
            raise ValueError("Need at least 30 rows to train EMA_Agent.")

        X = df[["ema_div", "ema_slope", "roc_5"]][:-1]
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
            ["ema_div", "ema_slope", "roc_5"]
        ]
        prob_up = float(self.model.predict_proba(last)[0, 1])
        return (prob_up * 2) - 1


