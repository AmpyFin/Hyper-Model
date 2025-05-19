"""
T3 Agent
========

Implements Tim Tillson’s Triple Exponential Moving Average (T3).
Default parameters: **period = 20**, **vfactor = 0.7**.

T3 construction
---------------
Let EMA_n = EMA(series, period).

Tillson’s recursive chain:

    e1 = EMA(price)
    e2 = EMA(e1)
    e3 = EMA(e2)
    e4 = EMA(e3)
    e5 = EMA(e4)
    e6 = EMA(e5)

Coefficients:

    c1 = -v³
    c2 = 3·v² + 3·v³
    c3 = -6·v² − 3·v − 3·v³
    c4 = 1 + 3·v + 3·v² + v³

Then:

    T3 = c1·e6 + c2·e5 + c3·e4 + c4·e3

Features
--------
* **T3 Divergence** – (close − t3) / t3
* **T3 Slope**      – % slope of T3
* **Price ROC-5**   – 5-bar rate-of-change of the close

Model
-----
LogisticRegression (scikit-learn) → probability close_{t+1} > close_t  
scaled to **score ∈ [-1, 1]**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ───────────────────────── helper functions ────────────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def t3(series: pd.Series, period: int = 20, vfactor: float = 0.7) -> pd.Series:
    """Compute Tillson’s T3 moving average."""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    e4 = ema(e3, period)
    e5 = ema(e4, period)
    e6 = ema(e5, period)

    v = vfactor
    c1 = -v**3
    c2 = 3 * v**2 + 3 * v**3
    c3 = -6 * v**2 - 3 * v - 3 * v**3
    c4 = 1 + 3 * v + 3 * v**2 + v**3

    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


# ───────────────────────────── agent class ──────────────────────────────
class T3_Agent:
    """LogReg learner on T3 divergence, slope, and 5-bar ROC."""

    def __init__(self, period: int = 20, vfactor: float = 0.7):
        self.period = period
        self.vfactor = vfactor
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # ----------------------- feature engineering ---------------------- #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close'")

        df = df.copy()
        df["t3"] = t3(df["close"], self.period, self.vfactor)
        df["t3_div"] = (df["close"] - df["t3"]) / df["t3"]
        df["t3_slope"] = df["t3"].pct_change()
        df["roc_5"] = df["close"].pct_change(periods=5)

        feats = ["t3_div", "t3_slope", "roc_5"]
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
        if len(df) < self.period + 15:
            raise ValueError("Not enough rows to train T3_Agent.")

        X = df[["t3_div", "t3_slope", "roc_5"]][:-1]
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
            ["t3_div", "t3_slope", "roc_5"]
        ]
        prob_up = float(self.model.predict_proba(last)[0, 1])
        return (prob_up * 2) - 1

