"""
CDLABANDONEDBABY Agent
======================

Abandoned Baby (bullish or bearish):

* Long candle in trend direction.
* A **doji** gaps beyond bar-1 high/low.
* Third candle gaps the other way and closes deep into bar-1 body.

Simplified detection with a lenient doji threshold (body ≤ 10 % of range).
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _is_doji(df, idx):
    body = (df["close"] - df["open"]).abs().iloc[idx]
    rng = (df["high"] - df["low"]).iloc[idx]
    return rng > 0 and body / rng <= 0.1


def _flag_abandoned(df):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    idx = df.index

    flags = pd.Series(0.0, index=idx)

    for i in range(2, len(df)):
        # identify possible doji at i-1
        if not _is_doji(df, i - 1):
            continue

        gap_down = l.iloc[i - 1] > h.iloc[i - 2]
        gap_up = h.iloc[i - 1] < l.iloc[i - 2]

        bullish = (c.iloc[i - 2] < o.iloc[i - 2]) and gap_down \
                  and (c.iloc[i] > o.iloc[i]) and (o.iloc[i] > c.iloc[i - 1]) \
                  and (c.iloc[i] > (o.iloc[i - 2] + c.iloc[i - 2]) / 2)

        bearish = (c.iloc[i - 2] > o.iloc[i - 2]) and gap_up \
                  and (c.iloc[i] < o.iloc[i]) and (o.iloc[i] < c.iloc[i - 1]) \
                  and (c.iloc[i] < (o.iloc[i - 2] + c.iloc[i - 2]) / 2)

        if bullish or bearish:
            flags.iat[i] = 1.0
    return flags


class CDLABANDONEDBABY_Agent:
    def __init__(self):
        self.model=LogisticRegression(max_iter=1000); self.fitted=False
    def _features(self,df):
        flag=_flag_abandoned(df).shift()
        d=df.copy(); d["flag"]=flag; d["roc3"]=d["close"].pct_change(3)
        d[["flag","roc3"]]=d[["flag","roc3"]].fillna(0)
        return d.dropna()
    def fit(self,df):
        data=self._features(df)
        if len(data)<60: raise ValueError("Not enough rows for CDLABANDONEDBABY_Agent")
        X=data[["flag","roc3"]][:-1]; y=(data["close"].shift(-1)>data["close"]).astype(int)[:-1]
        self.model.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._features(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.model.predict_proba(last)[0,1]-1
