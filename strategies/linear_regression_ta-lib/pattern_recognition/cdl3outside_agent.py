"""
CDL3OUTSIDE Agent
=================

Three Outside Up / Down (bullish- or bearish-engulfing continuation).

Bullish version
---------------
* Bar-1: black
* Bar-2: white **engulfing** bar-1 body
* Bar-3: white, closes above bar-2 close

Bearish version = colours reversed.

Feature set
-----------
* **flag**  – 1 when pattern completed on t-1, else 0
* **ROC-3** – 3-bar momentum context
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _flag_3outside(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]

    # Bullish setup
    bear1   = c.shift(2) < o.shift(2)
    white2  = c.shift(1) > o.shift(1)
    engulf2 = (o.shift(1) < c.shift(2)) & (c.shift(1) > o.shift(2))
    white3  = c > c.shift(1)
    bull_flag = bear1 & white2 & engulf2 & white3

    # Bearish setup
    bull1   = c.shift(2) > o.shift(2)
    black2  = c.shift(1) < o.shift(1)
    engulf2b = (o.shift(1) > c.shift(2)) & (c.shift(1) < o.shift(2))
    black3  = c < c.shift(1)
    bear_flag = bull1 & black2 & engulf2b & black3

    return (bull_flag | bear_flag).astype(float)


class CDL3OUTSIDE_Agent:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.fitted = False

    def _features(self, df):
        flag = _flag_3outside(df).shift()          # t-1 pattern predicts t
        d = df.copy()
        d["flag"] = flag
        d["roc3"] = d["close"].pct_change(3)
        d[["flag", "roc3"]] = d[["flag", "roc3"]].fillna(0)
        return d.dropna()

    def fit(self, df):
        data = self._features(df)
        if len(data) < 50:
            raise ValueError("Not enough rows for CDL3OUTSIDE_Agent")
        X = data[["flag", "roc3"]][:-1]
        y = (data["close"].shift(-1) > data["close"]).astype(int)[:-1]
        self.model.fit(X, y); self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._features(historical_df).iloc[-1:][["flag", "roc3"]]
        return 2 * self.model.predict_proba(last)[0, 1] - 1
