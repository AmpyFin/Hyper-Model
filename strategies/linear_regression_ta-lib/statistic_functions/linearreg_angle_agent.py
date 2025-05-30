"""
LINEARREG_ANGLE Agent
=====================

30-period **linear-regression angle** (degrees):

    slope = Δy / Δt   (from polyfit)
    angle = arctan(slope) × 180 / π

Features
--------
* angle (degrees)
* angle slope  (first diff)
"""

from __future__ import annotations
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression


def _lr_angle(series: pd.Series, n: int = 30) -> pd.Series:
    idx = np.arange(n)
    def ang(window):
        y = window.values
        b, _ = np.polyfit(idx, y, 1)   # slope b
        return np.degrees(np.arctan(b))
    return series.rolling(n).apply(ang, raw=False)


class LINEARREGANGLE_Agent:
    def __init__(self, period: int = 30):
        self.n = period
        self.m = LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        a = _lr_angle(df["close"], self.n)
        d = df.copy()
        d["angle"] = a
        d["angle_slope"] = a.diff()
        return d.dropna(subset=["angle", "angle_slope"])

    def fit(self, df):
        d=self._feat(df)
        if len(d)<self.n+10:
            raise ValueError("Not enough rows for LINEARREG_ANGLE Agent")
        X=d[["angle","angle_slope"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["angle","angle_slope"]]
        return 2*self.m.predict_proba(last)[0,1]-1
