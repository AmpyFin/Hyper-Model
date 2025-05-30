"""
WMA Agent
=========

Uses the classic linearly Weighted Moving Average (WMA, default 20-period)
to model price direction.

Formula
-------
For window length *n* with weights 1…n (higher weight on newest bar):

    WMA_t = Σ_{i=0}^{n-1} (w_i · price_{t-i}) / Σ_{i=0}^{n-1} w_i
    where w_i = i + 1

Features
--------
* **WMA Divergence** – (close − wma) / wma
* **WMA Slope**      – 1-bar % slope of the WMA
* **Price ROC-5**    – 5-bar rate-of-change of the close

Model
-----
LogisticRegression (scikit-learn) → probability close_{t+1} > close_t  
linearly scaled to **score ∈ [-1, 1]**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ───────────────────── Weighted MA helper ──────────────────────────────
def wma(series: pd.Series, period: int = 20) -> pd.Series:
    weights = np.arange(1, period + 1)

    def _calc(x):
        return np.dot(x, weights) / weights.sum()

    return series.rolling(period).apply(_calc, raw=True)


# ─────────────────────────── agent class ───────────────────────────────
class WMA_Agent:
    """LogReg learner on WMA divergence, slope, and 5-bar ROC."""

    def __init__(self, period: int = 20):
        self.period = period
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # ---------------------- feature engineering ----------------------- #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        df = df.copy()
        df["wma"] = wma(df["close"], self.period)
        df["wma_div"] = (df["close"] - df["wma"]) / df["wma"]
        df["wma_slope"] = df["wma"].pct_change()
        df["roc_5"] = df["close"].pct_change(periods=5)

        feats = ["wma_div", "wma_slope", "roc_5"]
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
            raise ValueError("Not enough rows to train WMA_Agent.")

        X = df[["wma_div", "wma_slope", "roc_5"]][:-1]
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
            ["wma_div", "wma_slope", "roc_5"]
        ]
        prob_up = float(self.model.predict_proba(last)[0, 1])
        return (prob_up * 2) - 1

