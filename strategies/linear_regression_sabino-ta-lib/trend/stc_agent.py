"""
STC Agent
=========

Schaff Trend Cycle – MACD fed into Stochastic.

Parameters
----------
* MACD fast 23, slow 50, signal 10
* Stoch k 10, d 3

Features
--------
* stc      (0…100)
* stc_slope
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _ema(s, n): return s.ewm(span=n, adjust=False).mean()


def _stc(close):
    macd_fast, macd_slow, macd_sig = 23, 50, 10
    k_len, d_len = 10, 3

    macd = _ema(close, macd_fast) - _ema(close, macd_slow)
    macd_sig_line = _ema(macd, macd_sig)
    macd_hist = macd - macd_sig_line

    lowest = macd_hist.rolling(k_len).min()
    highest = macd_hist.rolling(k_len).max()
    stoch_k = 100 * (macd_hist - lowest) / (highest - lowest)
    stoch_d = stoch_k.rolling(d_len).mean()
    return stoch_d


class STC_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        stc = _stc(df["close"])
        d = df.copy()
        d["stc"] = stc / 100.0
        d["stc_slope"] = d["stc"].diff()
        return d.dropna(subset=["stc","stc_slope"])

    def fit(self, df):
        d=self._feat(df)
        if len(d)<100: raise ValueError("Not enough rows for STC_Agent")
        X=d[["stc","stc_slope"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["stc","stc_slope"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Schaff Trend Cycle.
        
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
            print(f"Error in STC strategy: {str(e)}")
            return 0.0000
