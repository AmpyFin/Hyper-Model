"""
CDL3STARSINSOUTH Agent
======================

Three Stars in the South – rare bullish reversal of a down-move.

Simplified rules
----------------
* Three consecutive **black** candles with lower highs & lows.
* Bar-2 body ≤ 60 % of bar-1 body and gaps down below bar-1 low.
* Bar-3 opens within bar-2 range and closes above bar-2 close.

Feature set identical to prior agents.
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag_3stars(df):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]

    black1 = c.shift(2) < o.shift(2)
    black2 = c.shift(1) < o.shift(1)
    black3 = c < o

    lower_highs = (h.shift(1) < h.shift(2)) & (h < h.shift(1))
    lower_lows  = (l.shift(1) < l.shift(2)) & (l < l.shift(1))

    body1 = (o.shift(2) - c.shift(2)).abs()
    body2_small = (o.shift(1) - c.shift(1)).abs() <= body1 * 0.6
    gap2 = o.shift(1) < l.shift(2)

    open3_in2 = o.between(l.shift(1), h.shift(1))
    close3_above2 = c > c.shift(1)

    flag = (black1 & black2 & black3 &
            lower_highs & lower_lows &
            body2_small & gap2 &
            open3_in2 & close3_above2)
    return flag.astype(float)


class CDL3STARSINSOUTH_Agent:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000); self.fitted=False
    def _features(self, df):
        flag=_flag_3stars(df).shift()
        d=df.copy(); d["flag"]=flag; d["roc3"]=d["close"].pct_change(3)
        d[["flag","roc3"]]=d[["flag","roc3"]].fillna(0)
        return d.dropna()
    def fit(self, df):
        data=self._features(df)
        if len(data)<60: raise ValueError("Not enough rows for CDL3STARSINSOUTH_Agent")
        X=data[["flag","roc3"]][:-1]; y=(data["close"].shift(-1)>data["close"]).astype(int)[:-1]
        self.model.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._features(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.model.predict_proba(last)[0,1]-1
