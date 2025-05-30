"""
CDLMORNINGDOJISTAR Agent
========================

Morning Doji Star (bullish reversal):

* Bar-2: long black.
* Bar-1: **doji** gaps down below Bar-2 low.
* Bar-0: long white closes above midpoint of Bar-2 body.

Doji threshold = body ≤ 10 % of range.
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _is_doji(idx, df):
    body = abs(df["close"].iat[idx] - df["open"].iat[idx])
    rng  = df["high"].iat[idx] - df["low"].iat[idx]
    return rng > 0 and body / rng <= 0.1


def _flag(df):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    flag = pd.Series(0.0, index=df.index)
    for i in range(2, len(df)):
        if not _is_doji(i - 1, df):
            continue
        black1   = c.iat[i - 2] < o.iat[i - 2]
        gap_down = h.iat[i - 1] < l.iat[i - 2]
        white0   = c.iat[i] > o.iat[i]
        close_mid= c.iat[i] > (o.iat[i - 2] + c.iat[i - 2]) / 2
        if black1 and gap_down and white0 and close_mid:
            flag.iat[i] = 1.0
    return flag.shift().fillna(0)


class CDLMORNINGDOJISTAR_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<60: raise ValueError("Not enough rows for MORNINGDOJISTAR")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
