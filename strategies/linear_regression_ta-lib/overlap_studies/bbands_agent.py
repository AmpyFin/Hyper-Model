"""
Bollinger-Band Agent
====================

Learns trading signals from classic 20-period Bollinger Bands.

Features
--------
* **%B**      – where price sits between lower & upper band
* **Bandwidth** – (upper − lower) / SMA
* **Band Slope** – 1-bar slope of the SMA (trend proxy)

Model
-----
LogisticRegression (scikit-learn), maps features → probability
price will rise next bar.  Probability is linearly scaled to
**score ∈ [-1, 1]**:

    -1  … strong short bias  
     0  … neutral  
    +1  … strong long bias
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ────────────────────────── indicator helper ────────────────────────────
def bollinger_bands(series: pd.Series, period: int = 20, k: float = 2.0):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + k * std
    lower = sma - k * std
    return sma, upper, lower


# ───────────────────────────── agent class ──────────────────────────────
class BBANDS_Agent:
    """LogReg learner on %B, bandwidth, and SMA slope."""

    def __init__(self, period: int = 20, k: float = 2.0):
        self.period = period
        self.k = k
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # ----------------------- feature engineering ----------------------- #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        df = df.copy()
        sma, upper, lower = bollinger_bands(df["close"], self.period, self.k)

        # %B  : 0 at lower band, 1 at upper
        df["pct_b"] = (df["close"] - lower) / (upper - lower)

        # Bandwidth: relative width
        df["band_width"] = (upper - lower) / sma

        # SMA slope: simple first difference normalized
        df["sma_slope"] = sma.diff() / sma

        feats = ["pct_b", "band_width", "sma_slope"]
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
        if len(df) < 2:
            raise ValueError("Not enough data after feature engineering.")

        X = df[["pct_b", "band_width", "sma_slope"]][:-1]
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
            ["pct_b", "band_width", "sma_slope"]
        ]

        prob_up = float(self.model.predict_proba(last_row)[0, 1])
        return (prob_up * 2) - 1


