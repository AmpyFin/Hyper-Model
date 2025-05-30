"""
HT_DCPERIOD Agent
=================

Dominant Cycle Period (smoothed).

Features
--------
* Period / 50        (≈ 0-1)
* Period slope
* Price ROC-3
"""

from __future__ import annotations
import numpy as np, pandas as pd
from ..utils import BaseAgent

# ─────────────────── local Hilbert helpers ────────────────────
def _analytic(series: pd.Series) -> pd.Series:
    x = series.values.astype(float)
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0: h[0]=h[n//2]=1; h[1:n//2]=2
    else:          h[0]=1; h[1:(n+1)//2]=2
    z_values = np.fft.ifft(X*h)
    return pd.Series(z_values, index=series.index)

def _hilbert_parts(close: pd.Series):
    z = _analytic(close)
    # Extract real and imaginary parts from complex values
    I = pd.Series(np.real(z.values), index=z.index)
    Q = pd.Series(np.imag(z.values), index=z.index)
    phase = (np.rad2deg(np.arctan2(Q, I))+360)%360
    dph = np.diff(phase, prepend=phase.iloc[0])
    dph[dph<=0] += 360
    dph = np.clip(dph, 1e-3, None)
    period = pd.Series(360/dph, index=close.index).rolling(10).mean()
    return phase, period

# ───────────────────────── agent ──────────────────────────────
class HT_DCPERIOD_Agent(BaseAgent):
    def __init__(self):
        super().__init__()

    def _feat(self,df):
        period=_hilbert_parts(df["close"])[1]
        d=df.copy()
        d["p"]=period/50.0
        d["slope"]=d["p"].diff()
        d["roc3"]=d["close"].pct_change(3)
        feats=["p","slope","roc3"]
        d = self.replace_inf_with_nan(d, feats)
        return d.dropna(subset=feats)

    def fit(self,df):
        d=self._feat(df); 
        if len(d)<30: raise ValueError("Not enough rows")
        X=d[["p","slope","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        
        # Scale features using the base class method
        X_scaled = self._scale_features(X)
        self.m.fit(X_scaled, y)
        self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["p","slope","roc3"]]
        
        # Scale the features using the base class method
        last_scaled = self._scale_features(last)
        return 2*self.m.predict_proba(last_scaled)[0,1]-1
