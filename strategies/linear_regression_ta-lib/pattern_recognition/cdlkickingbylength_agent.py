"""
CDLKICKINGBYLENGTH Agent
========================

Kicking by Length – same as Kicking but *direction* decided by which
marubozu body is longer.  We simply flag any valid Kicking and let the
model learn.

Detection uses same rules as CDLKICKING.
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Include the _is_marubozu function locally instead of importing
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
        b1_black = _is_marubozu(i - 1, df, "black")
        b1_white = _is_marubozu(i - 1, df, "white")
        b0_black = _is_marubozu(i, df, "black")
        b0_white = _is_marubozu(i, df, "white")

        if b1_black and b0_white and df["open"].iat[i] > df["high"].iat[i - 1]:
            flag.iat[i] = 1.0
        elif b1_white and b0_black and df["open"].iat[i] < df["low"].iat[i - 1]:
            flag.iat[i] = 1.0
    return flag.shift().fillna(0)


class CDLKICKINGBYLENGTH_Agent:
    def __init__(self): self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self,df):
        d=df.copy(); d["flag"]=_flag(df); d["roc3"]=d["close"].pct_change(3).fillna(0)
        return d.dropna()
    def fit(self,df):
        d=self._feat(df)
        if len(d)<50: raise ValueError("Not enough rows for KICKINGBYLENGTH")
        X=d[["flag","roc3"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["flag","roc3"]]
        return 2*self.m.predict_proba(last)[0,1]-1
