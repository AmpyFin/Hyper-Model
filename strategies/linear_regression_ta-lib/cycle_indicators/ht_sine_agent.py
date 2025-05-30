"""
HT_SINE Agent
=============

Hilbert SineWave (sine & lead-sine).

Features
--------
* Sine
* LeadSine
* Spread (sine − lead)
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
def _analytic(x: pd.Series):
    v=x.values.astype(float); n=len(v); X=np.fft.fft(v); h=np.zeros(n)
    if n%2==0: h[0]=h[n//2]=1; h[1:n//2]=2
    else: h[0]=1; h[1:(n+1)//2]=2
    z_values = np.fft.ifft(X*h)
    return pd.Series(z_values, index=x.index)
def _phase(close):
    z = _analytic(close)
    real_vals = np.real(z.values)
    imag_vals = np.imag(z.values)
    return pd.Series((np.rad2deg(np.arctan2(imag_vals, real_vals))+360)%360, index=close.index)
class HT_SINE_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        ph=_phase(df["close"]); rad=np.deg2rad(ph)
        sine=np.sin(rad); lead=np.sin(rad+np.pi/4)
        d=df.copy()
        d["sine"]=sine; d["lead"]=lead; d["spread"]=d["sine"]-d["lead"]
        feats=["sine","lead","spread"]
        d[feats]=d[feats].ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df)
        if len(d)<30: raise ValueError("Not enough rows")
        X=d[["sine","lead","spread"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["sine","lead","spread"]]
        return 2*self.m.predict_proba(last)[0,1]-1
