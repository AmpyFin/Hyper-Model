"""
MAMA Agent
==========

Implements a *lightweight* version of John Ehlers’ MESA Adaptive Moving
Average (MAMA) and its following average, FAMA.

Because the full Hilbert-transform implementation is lengthy, this agent
uses the commonly accepted **“simplified dynamic alpha”** version:

    α_t = SLOW + (FAST − SLOW) · |ΔP| / (|ΔP| + ε)

where ΔP = price_t − price_{t-1}.  
`α_t` adapts between **FAST = 0.5** and **SLOW = 0.05**.  
Then

    MAMA_t = α_t · P_t + (1 − α_t) · MAMA_{t-1}
    FAMA_t = 0.5 · α_t · MAMA_t + (1 − 0.5 · α_t) · FAMA_{t-1}

Features
--------
* **MAMA Divergence** – (close − mama) / mama
* **MAMA Slope**      – pct slope of MAMA
* **MAMA–FAMA Ratio** – (mama − fama) / mama

Model
-----
LogisticRegression (scikit-learn) → probability price will rise next bar  
Mapped linearly to **score ∈ [-1, 1]**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ────────────────────────── MAMA helper ────────────────────────────────
def mama(series: pd.Series, fast: float = 0.5, slow: float = 0.05):
    """
    Return (mama, fama) using simplified adaptive alpha.
    """
    eps = 1e-10
    mama = np.zeros_like(series, dtype=float)
    fama = np.zeros_like(series, dtype=float)

    mama[0] = series.iloc[0]
    fama[0] = series.iloc[0]

    for i in range(1, len(series)):
        delta = abs(series.iloc[i] - series.iloc[i - 1])
        alpha = slow + (fast - slow) * delta / (delta + eps)

        mama[i] = alpha * series.iloc[i] + (1 - alpha) * mama[i - 1]
        fama[i] = 0.5 * alpha * mama[i] + (1 - 0.5 * alpha) * fama[i - 1]

    mama_series = pd.Series(mama, index=series.index)
    fama_series = pd.Series(fama, index=series.index)
    return mama_series, fama_series


# ───────────────────────────── agent class ──────────────────────────────
class MAMA_Agent:
    """LogReg learner on MAMA divergence, slope, and MAMA/FAMA spread."""

    def __init__(self, fast: float = 0.5, slow: float = 0.05):
        self.fast = fast
        self.slow = slow
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # ----------------------- feature engineering ----------------------- #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        df = df.copy()
        mama_ser, fama_ser = mama(df["close"], self.fast, self.slow)

        df["mama"] = mama_ser
        df["fama"] = fama_ser

        df["mama_div"] = (df["close"] - df["mama"]) / df["mama"]
        df["mama_slope"] = df["mama"].pct_change()
        df["mfs"] = (df["mama"] - df["fama"]) / df["mama"]

        feats = ["mama_div", "mama_slope", "mfs"]
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
            raise ValueError("Need at least 50 rows to train MAMA_Agent.")

        X = df[["mama_div", "mama_slope", "mfs"]][:-1]
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
            ["mama_div", "mama_slope", "mfs"]
        ]
        prob_up = float(self.model.predict_proba(last)[0, 1])
        return (prob_up * 2) - 1

