"""
CDLCONCEALBABYSWALL Agent
=========================

Concealing Baby Swallow (very rare bullish reversal).

Simplified Rules
----------------
* Four consecutive **black** candles.
* Candle-2 and Candle-3 **gap down** below prior low.
* Candle-3 & Candle-4 open == high (no upper shadow).
* Candle-4 closes above Candle-3 close (swallow).

Features
--------
* flag
* roc3
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag(df: pd.DataFrame) -> pd.Series:
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]

    black = lambda idx: c.shift(idx) < o.shift(idx)

    cond = (
        black(3) & black(2) & black(1) & black(0) &
        (o.shift(2) < l.shift(3)) & (o.shift(1) < l.shift(2)) &
        (o.shift(1) == h.shift(1)) & (o == h) &
        (c > c.shift(1))
    )
    return cond.shift().fillna(0).astype(float)


class CDLCONCEALBABYSWALL_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy()
        d["flag"]=_flag(df)
        d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<120: raise ValueError("Not enough rows for CONCEALBABYSWALL")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
