"""
Ichimoku Agent
==============

Default periods 9 / 26 / 52.

Lines
-----
* conversion (tenkan)    = (9-H + 9-L) / 2
* base (kijun)           = (26-H + 26-L) / 2
* spanA                  = (conversion + base)/2  shifted +26
* spanB                  = (52-H + 52-L)/2       shifted +26

Features
--------
* span_diff  = spanA − spanB
* price_spanA = close − spanA
* price_spanB = close − spanB
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


class Ichimoku_Agent:
    def __init__(self, conv=9, base=26, span_b=52, shift=26):
        self.cv, self.bs, self.sb, self.sh = conv, base, span_b, shift
        self.m = LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        high, low, close = df["high"], df["low"], df["close"]
        conv = (high.rolling(self.cv).max() + low.rolling(self.cv).min()) / 2
        base = (high.rolling(self.bs).max() + low.rolling(self.bs).min()) / 2
        spanA = ((conv + base) / 2).shift(self.sh)
        spanB = ((high.rolling(self.sb).max() + low.rolling(self.sb).min()) / 2).shift(self.sh)

        d = df.copy()
        d["span_diff"] = spanA - spanB
        d["price_spanA"] = close - spanA
        d["price_spanB"] = close - spanB
        return d.dropna(subset=["span_diff","price_spanA","price_spanB"])

    def fit(self, df):
        d=self._feat(df)
        if len(d)<self.sb+self.sh+10: raise ValueError("Not enough rows for Ichimoku_Agent")
        X=d[["span_diff","price_spanA","price_spanB"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["span_diff","price_spanA","price_spanB"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Ichimoku Cloud.
        
        Parameters
        ----------
        historical_df : pd.DataFrame
            Historical OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns
        -------
        float
            Trading signal in range [-1.0000, 1.0000]
            -1.0000 = Strong sell
            -0.5000 = Weak sell
             0.0000 = Neutral
             0.5000 = Weak buy
             1.0000 = Strong buy
        """
        try:
            # First ensure the model is fitted with the historical data
            if not self.fitted:
                self.fit(historical_df)
            
            # Get current price for prediction
            current_price = historical_df['close'].iloc[-1]
            
            # Generate signal using predict method
            signal = self.predict(current_price=current_price, historical_df=historical_df)
            
            # Ensure signal is properly formatted to 4 decimal places
            return float(round(signal, 4))
            
        except ValueError as e:
            # Handle the case where there's not enough data
            print(f"Warning: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            print(f"Error in Ichimoku strategy: {str(e)}")
            return 0.0000
