"""
Ultimate Oscillator Agent
=========================

Periods: 7,14,28  (classic Williams).

Features
--------
* ultosc   (0‒1 scaled)
* ult_slope
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _bp_tr(df):
    bp = df["close"] - pd.concat([df["low"], df["close"].shift()], axis=1).min(axis=1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    return bp, tr


def _ultimate(df, p1=7, p2=14, p3=28):
    bp, tr = _bp_tr(df)
    avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum()
    avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum()
    avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum()
    return 100 * (4*avg1 + 2*avg2 + avg3) / 7


class UltimateOsc_Agent:
    def __init__(self):
        self.m=LogisticRegression(max_iter=1000); self.fitted=False
    def _feat(self, df):
        u=_ultimate(df)
        d=df.copy(); d["ult"]=u/100.0; d["ult_slope"]=d["ult"].diff()
        return d.dropna(subset=["ult","ult_slope"])
    def fit(self, df):
        d=self._feat(df)
        if len(d)<60: raise ValueError("Not enough rows for UltimateOsc_Agent")
        X=d[["ult","ult_slope"]][:-1]; y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True
    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["ult","ult_slope"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Ultimate Oscillator.
        
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
            print(f"Error in Ultimate Oscillator strategy: {str(e)}")
            return 0.0000
