"""
SAR Agent
=========

Implements the classic *Parabolic Stop-and-Reverse* (Parabolic SAR) and
learns trading signals from its relationship to price.

Algorithm
---------
Default parameters: **step = 0.02** **max_step = 0.2**

Features
--------
* **SAR Divergence** – (close − psar) / close
* **SAR Slope**      – 1-bar % slope of the PSAR
* **Trend Flag**     – +1 if PSAR below price (up-trend), −1 otherwise

Model
-----
LogisticRegression (scikit-learn) → probability price will rise next bar,
scaled to **score ∈ [-1, 1]**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ────────────────────── Parabolic SAR helper ───────────────────────────
def psar(high: pd.Series, low: pd.Series,
         step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """
    Vectorised Parabolic SAR (classic J. Welles Wilder Jr. rules).
    Returns a pandas Series aligned with *high/low*.
    """
    length = len(high)
    psar_vals = np.zeros(length)
    bull = True  # start as up-trend
    af = step
    ep = low.iloc[0]  # extreme point

    psar_vals[0] = low.iloc[0]

    for i in range(1, length):
        prev_psar = psar_vals[i - 1]

        # tentative PSAR
        psar = prev_psar + af * (ep - prev_psar)

        # ensure PSAR does not penetrate last two bars
        if bull:
            psar = min(psar, low.iloc[i - 1], low.iloc[i - 2] if i > 1 else low.iloc[i - 1])
        else:
            psar = max(psar, high.iloc[i - 1], high.iloc[i - 2] if i > 1 else high.iloc[i - 1])

        reverse = False
        if bull:
            if low.iloc[i] < psar:          # bull → bear reversal
                bull = False
                reverse = True
                psar = ep                   # reset to previous EP
                ep = high.iloc[i]
                af = step
        else:
            if high.iloc[i] > psar:         # bear → bull reversal
                bull = True
                reverse = True
                psar = ep
                ep = low.iloc[i]
                af = step

        # update EP & AF if no reversal
        if not reverse:
            if bull:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_step)
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_step)

        psar_vals[i] = psar

    return pd.Series(psar_vals, index=high.index)


# ───────────────────────────── agent class ──────────────────────────────
class SAR_Agent:
    """LogReg learner on PSAR divergence, slope, and trend direction flag."""

    def __init__(self, step: float = 0.02, max_step: float = 0.2):
        self.step = step
        self.max_step = max_step
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # ----------------------- feature engineering ----------------------- #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must include columns: {required}")

        df = df.copy()
        df["psar"] = psar(df["high"], df["low"], self.step, self.max_step)
        df["sar_div"] = (df["close"] - df["psar"]) / df["close"]
        df["sar_slope"] = df["psar"].pct_change()
        df["trend_flag"] = np.where(df["psar"] < df["close"], 1.0, -1.0)

        feats = ["sar_div", "sar_slope", "trend_flag"]
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
            raise ValueError("Need at least 50 rows to train SAR_Agent.")

        X = df[["sar_div", "sar_slope", "trend_flag"]][:-1]
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
            ["sar_div", "sar_slope", "trend_flag"]
        ]
        prob_up = float(self.model.predict_proba(last)[0, 1])
        return (prob_up * 2) - 1


