"""
CDLKICKING Agent
================

Kicking (gap marubozu → opposite marubozu).

Bullish
-------
* Bar-1: long **black marubozu** (open=high, close=low).
* Bar-0: long **white marubozu** (open=low,  close=high)
         **gaps up** above Bar-1 high.

Bearish = colours reversed with gap down.

Marubozu test: shadow ≤ 10 % of body.
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _is_marubozu(idx, df, color):
    o, c, h, l = (df[col].iat[idx] for col in ("open", "close", "high", "low"))
    body = abs(c - o)
    if body == 0: return False
    upper = h - max(o, c)
    lower = min(o, c) - l
    no_shadows = (upper <= body * 0.1) and (lower <= body * 0.1)
    return no_shadows and ((color == "white" and c > o) or (color == "black" and c < o))


def _flag(df):
    idx = df.index
    flag = pd.Series(0.0, index=idx)
    for i in range(1, len(df)):
        if _is_marubozu(i - 1, df, "black") and _is_marubozu(i, df, "white"):
            if df["open"].iat[i] > df["high"].iat[i - 1]:
                flag.iat[i] = 1.0
        if _is_marubozu(i - 1, df, "white") and _is_marubozu(i, df, "black"):
            if df["open"].iat[i] < df["low"].iat[i - 1]:
                flag.iat[i] = 1.0
    return flag.shift().fillna(0)


class CDLKICKING_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<50: raise ValueError("Not enough rows for KICKING")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
