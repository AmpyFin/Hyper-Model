"""
MAVP Agent
==========

Moving Average with Variable Period (MAVP).

Here we choose the *period* dynamically from the latest **volume** level,
scaled into an integer window between 10 and 50 bars:

    vol_norm = (volume − vol_min) / (vol_max − vol_min)
    period_t = round(10 + vol_norm * 40)

You can swap this driver series for any other signal (ATR, range, etc.).

Features
--------
* **MAVP Divergence** – (close − mavp) / mavp
* **MAVP Slope**      – 1-bar % slope of the MAVP
* **Price ROC-5**     – 5-bar rate-of-change of close

Model
-----
LogisticRegression → probability close_{t+1} > close_t  
scaled to score ∈ [-1, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ───────────────────── variable-period MA helper ───────────────────────
def mavp(price: pd.Series, driver: pd.Series,
         pmin: int = 10, pmax: int = 50) -> pd.Series:
    """Variable-period SMA driven by *driver* series."""
    drv_min, drv_max = driver.min(), driver.max()
    drv_norm = (driver - drv_min) / (drv_max - drv_min + 1e-9)
    periods = (pmin + drv_norm * (pmax - pmin)).round().astype(int)

    out = np.full(len(price), np.nan)
    for idx in range(len(price)):
        n = periods.iloc[idx]
        if idx + 1 >= n:
            out[idx] = price.iloc[idx - n + 1 : idx + 1].mean()
    return pd.Series(out, index=price.index)


# ───────────────────────────── agent class ─────────────────────────────
class MAVP_Agent:
    """LogReg learner on MAVP divergence, slope, and 5-bar ROC."""

    def __init__(self, pmin: int = 10, pmax: int = 50):
        self.pmin = pmin
        self.pmax = pmax
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # --------------------- feature engineering ------------------------ #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"close", "volume"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must include columns: {required}")

        df = df.copy()
        df["mavp"] = mavp(df["close"], df["volume"], self.pmin, self.pmax)
        df["mavp_div"] = (df["close"] - df["mavp"]) / df["mavp"]
        df["mavp_slope"] = df["mavp"].pct_change()
        df["roc_5"] = df["close"].pct_change(periods=5)

        feats = ["mavp_div", "mavp_slope", "roc_5"]
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
        if len(df) < self.pmax + 20:
            raise ValueError("Not enough rows to train MAVP_Agent.")

        X = df[["mavp_div", "mavp_slope", "roc_5"]][:-1]
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
            ["mavp_div", "mavp_slope", "roc_5"]
        ]
        prob_up = float(self.model.predict_proba(last)[0, 1])
        return (prob_up * 2) - 1
