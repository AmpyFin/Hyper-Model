"""
CDLHARAMICROSS Agent
====================

Harami Cross:

* Same as Harami, but **Bar-1 is a doji**.

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
    o, c = df["open"], df["close"]
    idx  = df.index
    flag = pd.Series(0.0, index=idx)

    for i in range(1, len(df)):
        if not _is_doji(i, df):
            continue
        # inside previous body?
        inside = (o.iat[i]  > min(o.iat[i-1], c.iat[i-1])) & \
                 (o.iat[i]  < max(o.iat[i-1], c.iat[i-1])) & \
                 (c.iat[i]  > min(o.iat[i-1], c.iat[i-1])) & \
                 (c.iat[i]  < max(o.iat[i-1], c.iat[i-1]))
        if inside:
            flag.iat[i] = 1.0
    return flag.shift().fillna(0)


class CDLHARAMICROSS_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<30: raise ValueError("Not enough rows for HARAMICROSS")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
