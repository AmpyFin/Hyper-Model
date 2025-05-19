"""
SAREXT Agent
============

Extended Parabolic SAR — same core as standard PSAR but with custom
acceleration factor (AF) parameters:

    start_af  – initial AF (default 0.02)
    inc_af    – increment when new extreme reached (default 0.02)
    max_af    – maximum AF cap (default 0.2)

Features
--------
* **SAREXT Divergence** – (close − psar) / close
* **SAREXT Slope**      – 1-bar % slope of PSAR
* **Trend Flag**        – +1 if PSAR < close, −1 otherwise

Model
-----
LogisticRegression → probability close_{t+1} > close_t  
scaled to score ∈ [-1, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ───────────────── extended SAR helper ────────────────────────────────
def psar_ext(high: pd.Series, low: pd.Series,
             start_af: float = 0.02,
             inc_af: float = 0.02,
             max_af: float = 0.2) -> pd.Series:

    length = len(high)
    sar = np.zeros(length)
    bull = True
    af = start_af
    ep = low.iloc[0]
    sar[0] = low.iloc[0]

    for i in range(1, length):
        prev_sar = sar[i - 1]
        sar_calc = prev_sar + af * (ep - prev_sar)

        if bull:
            sar_calc = min(sar_calc, low.iloc[i - 1],
                           low.iloc[i - 2] if i > 1 else low.iloc[i - 1])
        else:
            sar_calc = max(sar_calc, high.iloc[i - 1],
                           high.iloc[i - 2] if i > 1 else high.iloc[i - 1])

        reverse = False
        if bull and low.iloc[i] < sar_calc:
            bull = False
            reverse = True
            sar_calc = ep
            ep = high.iloc[i]
            af = start_af
        elif not bull and high.iloc[i] > sar_calc:
            bull = True
            reverse = True
            sar_calc = ep
            ep = low.iloc[i]
            af = start_af

        if not reverse:
            if bull and high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + inc_af, max_af)
            elif not bull and low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + inc_af, max_af)

        sar[i] = sar_calc

    return pd.Series(sar, index=high.index)


# ───────────────────────────── agent class ─────────────────────────────
class SAREXT_Agent:
    """LogReg learner on extended PSAR divergence, slope, trend flag."""

    def __init__(self, start_af: float = 0.02,
                 inc_af: float = 0.02, max_af: float = 0.2):
        self.start_af = start_af
        self.inc_af = inc_af
        self.max_af = max_af
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # ------------------------ feature builder ------------------------- #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must include columns: {required}")

        df = df.copy()
        df["psarx"] = psar_ext(
            df["high"], df["low"],
            self.start_af, self.inc_af, self.max_af
        )
        df["sarx_div"] = (df["close"] - df["psarx"]) / df["close"]
        df["sarx_slope"] = df["psarx"].pct_change()
        df["trend_flag"] = np.where(df["psarx"] < df["close"], 1.0, -1.0)

        feats = ["sarx_div", "sarx_slope", "trend_flag"]
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
            raise ValueError("Not enough rows to train SAREXT_Agent.")

        X = df[["sarx_div", "sarx_slope", "trend_flag"]][:-1]
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
            ["sarx_div", "sarx_slope", "trend_flag"]
        ]
        prob_up = float(self.model.predict_proba(last)[0, 1])
        return (prob_up * 2) - 1
