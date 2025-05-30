"""
HT_TRENDMODE Agent
==================

Trend (1) vs Cycle (0) flag using phase variability.

Features
--------
* TrendFlag
* Period / 50
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
def _phase_period(close):
    z = _analytic(close)
    I = pd.Series(np.real(z.values), index=z.index)
    Q = pd.Series(np.imag(z.values), index=z.index)
    ph = (np.rad2deg(np.arctan2(Q, I))+360)%360
    dph = np.diff(ph, prepend=ph.iloc[0])
    dph[dph<=0] += 360
    dph = np.clip(dph, 1e-3, None)
    per = pd.Series(360/dph, index=close.index).rolling(10).mean()
    return ph, per
class HT_TRENDMODE_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        ph,per=_phase_period(df["close"])
        trend=((pd.Series(ph).diff().abs().rolling(7).mean())<20).astype(float)
        d=df.copy()
        d["flag"]=trend; d["p"]=per/50.0; d["roc3"]=d["close"].pct_change(3)
        feats=["flag","p","roc3"]
        d[feats]=d[feats].replace([np.inf,-np.inf],np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    def fit(self,df):
        d=self._feat(df)
        if len(d)<40: raise ValueError("Not enough rows")
        X=d[["flag","p","roc3"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","p","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
