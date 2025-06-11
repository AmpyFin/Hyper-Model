"""
HT_PHASOR Agent
===============

Phasor components In-phase (I) and Quadrature (Q).

Features
--------
* I / close
* Q / close
* Magnitude / close
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
class HT_PHASOR_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        z = _analytic(df["close"])
        I = pd.Series(np.real(z.values), index=z.index)
        Q = pd.Series(np.imag(z.values), index=z.index)
        d=df.copy()
        d["i"]=I/df["close"]; d["q"]=Q/df["close"]
        d["mag"]=np.sqrt(I**2+Q**2)/df["close"]
        feats=["i","q","mag"]
        d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df); 
        if len(d)<30: raise ValueError("Not enough rows")
        X=d[["i","q","mag"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["i","q","mag"]]
        return 2*self.m.predict_proba(last)[0,1]-1
