"""
TEMA Agent
==========

Uses the Triple Exponential Moving Average (TEMA, default 20-period)
to model price direction.

Formula
-------
ema1 = EMA(price, n)
ema2 = EMA(ema1, n)
ema3 = EMA(ema2, n)
**TEMA = 3·ema1 – 3·ema2 + ema3**

Features
--------
* **TEMA Divergence** – (close − tema) / tema
* **TEMA Slope**      – 1-bar % slope of TEMA
* **Price ROC-5**     – 5-bar rate-of-change of the close

Model
-----
LogisticRegression ➜ probability price will rise next bar, mapped linearly to
**score ∈ [-1, 1]**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ───────────────────────── helper functions ────────────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def tema(series: pd.Series, period: int) -> pd.Series:
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3


# ───────────────────────────── agent class ──────────────────────────────
class TEMA_Agent:
    """LogReg learner on TEMA divergence, slope, and 5-bar ROC."""

    def __init__(self, period: int = 20):
        self.period = period
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # ----------------------- feature engineering ---------------------- #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        df = df.copy()
        df["tema"] = tema(df["close"], self.period)

        df["tema_div"] = (df["close"] - df["tema"]) / df["tema"]
        df["tema_slope"] = df["tema"].pct_change()
        df["roc_5"] = df["close"].pct_change(periods=5)

        feats = ["tema_div", "tema_slope", "roc_5"]
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
            raise ValueError("Not enough rows to train TEMA_Agent.")

        X = df[["tema_div", "tema_slope", "roc_5"]][:-1]
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

        last_row = self._add_features(historical_df).iloc[-1:][
            ["tema_div", "tema_slope", "roc_5"]
        ]
        prob_up = float(self.model.predict_proba(last_row)[0, 1])
        return (prob_up * 2) - 1


