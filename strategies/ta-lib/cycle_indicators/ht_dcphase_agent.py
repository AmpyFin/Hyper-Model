"""
HT_DCPHASE Agent
================

Dominant Cycle Phase (0-360°).

Features
--------
* Phase / 360
* Phase slope
* Price ROC-3
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
    ph = (np.rad2deg(np.arctan2(imag_vals, real_vals))+360)%360
    return pd.Series(ph, index=close.index)
class HT_DCPHASE_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        ph=_phase(df["close"])
        d=df.copy(); d["ph"]=ph/360.0
        d["slope"]=d["ph"].diff(); d["roc3"]=d["close"].pct_change(3)
        feats=["ph","slope","roc3"]
        d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df)
        if len(d)<30: raise ValueError("Not enough rows")
        X=d[["ph","slope","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["ph","slope","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
