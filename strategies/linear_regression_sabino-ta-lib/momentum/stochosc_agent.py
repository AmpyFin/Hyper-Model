"""
Stochastic Oscillator Agent
===========================

Classic slow-stoch: %K 14, %D 3 SMA.

Features
--------
* k
* d
* diff (k − d)
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _stoch(df, k_len=14, d_len=3):
    low_min = df["low"].rolling(k_len).min()
    high_max = df["high"].rolling(k_len).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    d = k.rolling(d_len).mean()
    return k, d


class StochOsc_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self, df):
        k,d=_stoch(df)
        dfrm=pd.DataFrame({"k":k/100.0,"d":d/100.0})
        dfrm["diff"]=dfrm["k"]-dfrm["d"]
        return pd.concat([df,dfrm],axis=1).dropna(subset=["k","d","diff"])
    def fit(self, df):
        d=self._feat(df)
        if len(d)<50: raise ValueError("Not enough rows for StochOsc_Agent")
        X=d[["k","d","diff"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["k","d","diff"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Stochastic Oscillator.
        
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
            print(f"Error in Stochastic Oscillator strategy: {str(e)}")
            return 0.0000
