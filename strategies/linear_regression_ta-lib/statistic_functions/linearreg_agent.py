"""
LINEARREG Agent
===============

30-period **linear-regression value** (same as TA-Lib LINEARREG).

Implementation
--------------
For window N: fit y = a + b·t  (t = 0…N-1) on *close*;
return predicted value at t = N-1.

Features
--------
* linreg_val
* linreg_resid  (close − linreg_val)
"""

from __future__ import annotations
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression


def _linear_reg(series: pd.Series, n: int = 30) -> pd.Series:
    idx = np.arange(n)
    def calc(window):
        y = window.values
        b, a = np.polyfit(idx, y, 1)      # slope, intercept
        return a + b * (n - 1)
    return series.rolling(n).apply(calc, raw=False)


class LINEARREG_Agent:
    def __init__(self, period: int = 30):
        self.n = period
        self.m = LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        lr = _linear_reg(df["close"], self.n)
        d = df.copy()
        d["lr"] = lr
        d["resid"] = df["close"] - lr
        return d.dropna(subset=["lr", "resid"])

    def fit(self, df):
        d=self._feat(df)
        if len(d)<self.n+10:
            raise ValueError("Not enough rows for LINEARREG_Agent")
        X=d[["lr","resid"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["lr","resid"]]
        return 2*self.m.predict_proba(last)[0,1]-1
