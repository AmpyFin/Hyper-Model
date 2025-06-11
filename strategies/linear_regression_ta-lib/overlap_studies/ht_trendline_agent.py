"""
HT_TRENDLINE Agent
==================

Implements John Ehlers’ *Instantaneous Trendline* (ITrend), the core of
TA-Lib’s `HT_TRENDLINE`.  Default smoothing constant **alpha = 0.07**.

Recursive formula
-----------------
    ITrend[t] = (a1 * price[t] + a2 * price[t-1] + a3 * price[t-2])
                + 2·(1 − α)·ITrend[t-1] − (1 − α)²·ITrend[t-2]

where
    a1 = α − α² / 4
    a2 = α² / 2
    a3 = α − 3α² / 4

Features
--------
* **IT Divergence** – (close − itrend) / itrend
* **IT Slope**      – pct slope of the itrend
* **Cross Flag**    – +1 if close > itrend, −1 otherwise

Model
-----
LogisticRegression → probability close_{t+1} > close_t  
scaled to **score ∈ [-1, 1]**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ────────────────── Hilbert Trendline helper ───────────────────────────
def itrend(series: pd.Series, alpha: float = 0.07) -> pd.Series:
    a1 = alpha - (alpha ** 2) / 4
    a2 = (alpha ** 2) / 2
    a3 = alpha - (3 * alpha ** 2) / 4

    it = np.zeros_like(series, dtype=float)
    it[0:2] = series.iloc[0:2]  # seed first two values

    for t in range(2, len(series)):
        it[t] = (
            a1 * series.iloc[t]
            + a2 * series.iloc[t - 1]
            + a3 * series.iloc[t - 2]
            + 2 * (1 - alpha) * it[t - 1]
            - (1 - alpha) ** 2 * it[t - 2]
        )
    return pd.Series(it, index=series.index)


# ─────────────────────────── agent class ───────────────────────────────
class HTTL_Agent:
    """LogReg learner on Instantaneous Trendline divergence, slope, cross flag."""

    def __init__(self, alpha: float = 0.07):
        self.alpha = alpha
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # ---------------------- feature engineering ----------------------- #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close'")

        df = df.copy()
        df["itrend"] = itrend(df["close"], self.alpha)
        df["it_div"] = (df["close"] - df["itrend"]) / df["itrend"]
        df["it_slope"] = df["itrend"].pct_change()
        df["cross_flag"] = np.where(df["close"] > df["itrend"], 1.0, -1.0)

        feats = ["it_div", "it_slope", "cross_flag"]
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
        if len(df) < 50:
            raise ValueError("Need at least 50 rows to train HTTL_Agent.")

        X = df[["it_div", "it_slope", "cross_flag"]][:-1]
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
            ["it_div", "it_slope", "cross_flag"]
        ]
        prob_up = float(self.model.predict_proba(last)[0, 1])
        return (prob_up * 2) - 1

