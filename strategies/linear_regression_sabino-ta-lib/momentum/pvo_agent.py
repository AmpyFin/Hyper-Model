"""
PVO Agent
=========

Percentage Volume Oscillator (vol-PPO) – fast 12, slow 26, signal 9.

Features
--------
* pvo_norm
* pvo_sig
* pvo_hist
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _ema(s, n): return s.ewm(span=n, adjust=False).mean()


class PVO_Agent:
    def __init__(self, fast=12, slow=26, sig=9):
        self.f, self.s, self.g = fast, slow, sig
        self.m=LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        ema_f = _ema(df["volume"], self.f)
        ema_s = _ema(df["volume"], self.s)
        pvo = (ema_f - ema_s) / ema_s * 100
        sig = _ema(pvo, self.g)
        hist = pvo - sig
        d = df.copy()
        d["pvo_norm"] = pvo / 100.0
        d["sig"] = sig / 100.0
        d["hist"] = hist / 100.0
        return d.dropna(subset=["pvo_norm","sig","hist"])

    def fit(self, df):
        d=self._feat(df)
        if len(d)<self.s+10: raise ValueError("Not enough rows for PVO_Agent")
        X=d[["pvo_norm","sig","hist"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["pvo_norm","sig","hist"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Percentage Volume Oscillator.
        
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
            print(f"Error in PVO strategy: {str(e)}")
            return 0.0000
