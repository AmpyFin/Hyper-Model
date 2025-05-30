"""
KAMA Agent
==========

Uses Kaufman Adaptive Moving Average (KAMA) to learn price direction.

Algorithm
---------
KAMA adapts its smoothing constant based on **Efficiency Ratio (ER)**:

    ER = |price − price_n| / Σ|price_i − price_{i-1}|

Smoothing constant:
    fastSC = 2 / (fast + 1)   (fast = 2)
    slowSC = 2 / (slow + 1)   (slow = 30)
    SC = (ER * (fastSC − slowSC) + slowSC)²

Then:
    KAMA[t] = KAMA[t-1] + SC * (price[t] − KAMA[t-1])

Features
--------
* **KAMA Divergence** – (close − kama) / kama
* **KAMA Slope**      – 1-bar pct slope of the KAMA
* **Efficiency Ratio** – ER itself

Model
-----
LogisticRegression → probability price will rise next bar  
Mapped to **score ∈ [-1, 1]**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ──────────────────── KAMA implementation ──────────────────────────────
def kama(series: pd.Series, er_period: int = 10, fast: int = 2, slow: int = 30):
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)

    kama_vals = np.zeros_like(series, dtype=float)
    er_vals = np.zeros_like(series, dtype=float)

    kama_vals[er_period - 1] = series.iloc[:er_period].mean()  # seed

    for i in range(er_period, len(series)):
        change = abs(series.iloc[i] - series.iloc[i - er_period])
        vol = np.sum(np.abs(np.diff(series.iloc[i - er_period : i + 1])))

        er = 0 if vol == 0 else change / vol
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        kama_vals[i] = kama_vals[i - 1] + sc * (series.iloc[i] - kama_vals[i - 1])
        er_vals[i] = er

    kama_series = pd.Series(kama_vals, index=series.index)
    er_series = pd.Series(er_vals, index=series.index)

    return kama_series, er_series


# ───────────────────────────── agent class ──────────────────────────────
class KAMA_Agent:
    """LogReg learner on KAMA divergence, slope, and Efficiency Ratio."""

    def __init__(self, er_period: int = 10, fast: int = 2, slow: int = 30):
        self.er_period = er_period
        self.fast = fast
        self.slow = slow
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # ----------------------- feature engineering ----------------------- #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        df = df.copy()
        kama_series, er_series = kama(
            df["close"], self.er_period, self.fast, self.slow
        )

        df["kama"] = kama_series
        df["er"] = er_series

        df["kama_div"] = (df["close"] - df["kama"]) / df["kama"]
        df["kama_slope"] = df["kama"].pct_change()

        feats = ["kama_div", "kama_slope", "er"]
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
        if len(df) < 50:
            raise ValueError("Need at least 50 rows to train KAMA_Agent.")

        X = df[["kama_div", "kama_slope", "er"]][:-1]
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
            ["kama_div", "kama_slope", "er"]
        ]
        prob_up = float(self.model.predict_proba(last)[0, 1])
        return (prob_up * 2) - 1
