"""
ADX Agent (robust)
==================

Average Directional Movement Index (ADX, default 14).

Features
--------
* **ADX**        – trend strength (0-1 after /100)
* **DI Spread**  – (+DI − −DI) / 100
* **DX**         – single-period DX / 100

All NaNs / infs are converted to finite numbers so the
agent trains even on very quiet intraday data.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ─────────────────────── Wilder ADX helper (robust) ─────────────────────
def adx_parts(df: pd.DataFrame, n: int = 14):
    hi, lo, cl = df["high"], df["low"], df["close"]

    up_move   = hi.diff()
    down_move = -lo.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),  up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(n).mean().replace(0, np.nan)
    plus_di  = 100 * pd.Series(plus_dm, index=df.index).rolling(n).sum() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).sum() / atr

    den = plus_di + minus_di
    dx_array = np.where(den == 0, 0, (abs(plus_di - minus_di) / den) * 100)
    dx = pd.Series(dx_array, index=df.index)
    adx = dx.rolling(n).mean()

    # fill zeros / NaNs so later math has finite numbers
    plus_di  = plus_di.fillna(0)
    minus_di = minus_di.fillna(0)
    dx       = dx.fillna(0)
    adx      = adx.ffill().fillna(0)

    return plus_di, minus_di, dx, adx


# ───────────────────────────── agent class ─────────────────────────────
class ADX_Agent:
    """LogReg learner on ADX, DI spread, and DX."""

    def __init__(self, period: int = 14):
        self.period = period
        self.model = LogisticRegression(max_iter=1000)
        self.fitted: bool = False

    # ---------------------- feature engineering ----------------------- #
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"high", "low", "close"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must include columns: {required}")

        plus_di, minus_di, dx, adx = adx_parts(df, self.period)

        out = df[["close"]].copy()          # keep close for target later
        out["adx"]        = adx / 100.0
        out["di_spread"]  = (plus_di - minus_di) / 100.0
        out["dx"]         = dx / 100.0

        feats = ["adx", "di_spread", "dx"]
        return out.dropna(subset=feats)

    # --------------------------- training ----------------------------- #
    def fit(self, ohlcv: pd.DataFrame) -> None:
        d = self._add_features(ohlcv)

        if len(d) < self.period + 10:
            raise ValueError(
                f"Only {len(d)} usable rows after ADX features. "
                "Pull a longer window or use higher timeframe."
            )

        X = d[["adx", "di_spread", "dx"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]

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
            ["adx", "di_spread", "dx"]
        ]
        prob_up = float(self.model.predict_proba(last)[0, 1])
        return (prob_up * 2) - 1



