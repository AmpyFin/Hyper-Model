"""
KST Agent
=========

KST (Know Sure Thing) Oscillator - common parameters:

ROC-periods 10,15,20,30  → smoothed by SMA 10,10,10,15.  
KST  =  ROC1s + 2·ROC2s + 3·ROC3s + 4·ROC4s  
Signal = SMA(KST, 9)

Features
--------
* kst
* kst_signal
* kst_diff  (kst − signal)
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _roc(s: pd.Series, n: int):
    return s.pct_change(n) * 100


def _sma(s, n): return s.rolling(n).mean()


def _kst(close: pd.Series):
    r1 = _roc(close, 10).rolling(10).mean()
    r2 = _roc(close, 15).rolling(10).mean()
    r3 = _roc(close, 20).rolling(10).mean()
    r4 = _roc(close, 30).rolling(15).mean()
    kst = r1 + 2 * r2 + 3 * r3 + 4 * r4
    signal = _sma(kst, 9)
    return kst, signal


class KST_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000); self.fitted=False

    def _feat(self, df):
        kst, sig = _kst(df["close"])
        d = df.copy()
        d["kst"], d["kst_sig"] = kst, sig
        d["kst_diff"] = kst - sig
        return d.dropna(subset=["kst", "kst_sig", "kst_diff"])

    def fit(self, df):
        d=self._feat(df)
        if len(d)<100: raise ValueError("Not enough rows for KST_Agent")
        X=d[["kst","kst_sig","kst_diff"]][:-1]
        y=(d["close"].shift(-1)>d["close"]).astype(int)[:-1]
        self.m.fit(X,y); self.fitted=True

    def predict(self,*,current_price,historical_df):
        if not self.fitted: self.fit(historical_df)
        last=self._feat(historical_df).iloc[-1:][["kst","kst_sig","kst_diff"]]
        return 2*self.m.predict_proba(last)[0,1]-1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Know Sure Thing (KST).
        
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
            print(f"Error in KST strategy: {str(e)}")
            return 0.0000
