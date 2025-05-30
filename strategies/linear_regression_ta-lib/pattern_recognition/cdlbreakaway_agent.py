"""
CDLBREAKAWAY Agent
==================

Breakaway (bullish or bearish) – 5-bar gap + reversal pattern.

Simplified detection
--------------------
* Bars 1-5 trend consecutively (HH/HL for bullish, LL/LH for bearish).
* Bar-5 closes into bar-1 body, signalling reversal.

This lenient version captures the core gap + reversal logic.
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag_breakaway(df):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]

    # Bullish Breakaway (down-gap start then bullish reversal)
    down_gap = o.shift(4) > c.shift(4)
    lower_seq = (l.shift(3) < l.shift(4)) & (l.shift(2) < l.shift(3)) & (l.shift(1) < l.shift(2))
    bull_break = (c > o) & (c > o.shift(4)) & down_gap & lower_seq

    # Bearish Breakaway
    up_gap = o.shift(4) < c.shift(4)
    higher_seq = (h.shift(3) > h.shift(4)) & (h.shift(2) > h.shift(3)) & (h.shift(1) > h.shift(2))
    bear_break = (c < o) & (c < o.shift(4)) & up_gap & higher_seq

    return (bull_break | bear_break).shift().fillna(0).astype(float)


class CDLBREAKAWAY_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self, df):
        d=df.copy(); d["flag"]=_flag_breakaway(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self, df):
        d=self._feat(df)
        if len(d)<100: raise ValueError("Not enough rows for BREAKAWAY")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
