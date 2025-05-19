"""
ADXR Agent
==========

Average Directional Movement Index Rating (ADXR):

    ADXR_t = (ADX_t + ADX_{t-period}) / 2

Features
--------
* **ADXR / 100**   – trend strength (0-1)
* **DI Spread**    – (+DI − −DI) / 100
* **DX / 100**     – single-period DX
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression


# ───────── Wilder helper reused from robust ADX ──────────
def _adx_parts(df: pd.DataFrame, n: int = 14):
    hi, lo, cl = df["high"], df["low"], df["close"]
    up, dn = hi.diff(), -lo.diff()
    plus_dm  = np.where((up > dn) & (up > 0),  up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([hi - lo,
                    (hi - cl.shift()).abs(),
                    (lo - cl.shift()).abs()], axis=1).max(axis=1)

    atr = tr.rolling(n).mean().replace(0, np.nan)
    pdi  = 100 * pd.Series(plus_dm,  index=df.index).rolling(n).sum() / atr
    mdi  = 100 * pd.Series(minus_dm, index=df.index).rolling(n).sum() / atr
    den  = pdi + mdi
    dx   = np.where(den == 0, 0, (abs(pdi - mdi) / den) * 100)
    adx  = pd.Series(dx, index=df.index).rolling(n).mean()

    return pdi.fillna(0), mdi.fillna(0), pd.Series(dx, index=df.index).fillna(0), adx.fillna(0)


class ADXR_Agent:
    def __init__(self, period: int = 14):
        self.n = period
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False

    def _feat(self, df: pd.DataFrame):
        req = {"high", "low", "close"}
        if not req.issubset(df.columns):
            raise ValueError("Missing OHLC columns")
        pdi, mdi, dx, adx = _adx_parts(df, self.n)
        adxr = (adx + adx.shift(self.n)) / 2

        d = df[["close"]].copy()
        d["adxr"] = adxr / 100.0
        d["spread"] = (pdi - mdi) / 100.0
        d["dx"] = dx / 100.0
        feats = ["adxr", "spread", "dx"]
        return d.dropna(subset=feats)

    def fit(self, df):
        d = self._feat(df)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows for ADXR_Agent")
        X = d[["adxr", "spread", "dx"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y)
        self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["adxr", "spread", "dx"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
