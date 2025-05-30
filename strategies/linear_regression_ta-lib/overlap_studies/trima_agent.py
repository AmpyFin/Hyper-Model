"""
TRIMA Agent
===========

Triangular Moving Average (TRIMA) is essentially a double-smoothed SMA.
For window *n*:

    TRIMA = SMA( SMA(price, n/2 + 1), n/2 + 1 )  if n is even
            SMA( SMA(price, (n+1)/2), (n+1)/2 )   if n is odd

Features
--------
* **TRIMA Divergence** – (close − trima) / trima
* **TRIMA Slope**      – 1-bar % slope of TRIMA
* **Price ROC-5**      – 5-bar rate of change of the close

Model
-----
LogisticRegression (scikit-learn) → probability close_{t+1} > close_t  
Mapped linearly to **score ∈ [-1, 1]**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ───────────────────── Triangular MA helper ────────────────────────────
def trima(series: pd.Series, period: int = 20) -> pd.Series:
    if period < 2:
        raise ValueError("period must be >= 2")

    if period % 2 == 0:
        sma1_len = period // 2
        sma2_len = period // 2 + 1
    else:
        sma1_len = (period + 1) // 2
        sma2_len = sma1_len

    sma1 = series.rolling(sma1_len).mean()
    return sma1.rolling(sma2_len).mean()


# ───────────────────────────── agent class ─────────────────────────────
class TRIMA_Agent:
    """LogReg learner on TRIMA divergence, slope, and 5-bar ROC."""

    def __init__(self, period: int = 20):
        self.period = period
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # --------------------- feature engineering ------------------------ #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        df = df.copy()
        df["trima"] = trima(df["close"], self.period)
        df["trima_div"] = (df["close"] - df["trima"]) / df["trima"]
        df["trima_slope"] = df["trima"].pct_change()
        df["roc_5"] = df["close"].pct_change(periods=5)

        feats = ["trima_div", "trima_slope", "roc_5"]
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
            raise ValueError("Not enough rows to train TRIMA_Agent.")

        X = df[["trima_div", "trima_slope", "roc_5"]][:-1]
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
            ["trima_div", "trima_slope", "roc_5"]
        ]
        prob_up = float(self.model.predict_proba(last)[0, 1])
        return (prob_up * 2) - 1

