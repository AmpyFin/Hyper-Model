"""
ADOSC Agent
===========

Chaikin Accumulation / Distribution Oscillator (ADOSC).

The ADOSC is calculated as the difference between fast and slow EMAs 
of the Accumulation/Distribution Line:

    MFM  = ((close − low) − (high − close)) / (high − low)
    MFV  = MFM · volume
    AD   = cumulative Σ(MFV)
    ADOSC = EMA_fast(AD) - EMA_slow(AD)

Typically uses 3-day and 10-day periods for fast and slow EMAs.

Features
--------
* **ADOSC**     – The oscillator value
* **ADOSC_norm** – ADOSC normalized by price
* **ADOSC Slope**
* **Price ROC-3**
"""

from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def _ad(df):
    """Calculate Accumulation/Distribution Line"""
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        (df["high"] - df["low"]).replace(0, np.nan)
    )
    mfv = mfm * df["volume"]
    return mfv.cumsum()

def _adosc(df, fast_period=3, slow_period=10):
    """Calculate Accumulation/Distribution Oscillator"""
    ad = _ad(df)
    ema_fast = ad.ewm(span=fast_period).mean()
    ema_slow = ad.ewm(span=slow_period).mean()
    return ema_fast - ema_slow

class ADOSC_Agent:
    def __init__(self, fast_period=3, slow_period=10):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False
    
    def _feat(self, df):
        req = {"high", "low", "close", "volume"}
        if not req.issubset(df.columns): 
            raise ValueError("Need OHLCV")
        
        adosc = _adosc(df, self.fast_period, self.slow_period)
        d = df.copy()
        
        # Normalize ADOSC by price to make it scale-independent
        d["adosc"] = adosc
        d["adosc_norm"] = adosc / df["close"]
        d["adosc_slope"] = d["adosc"].diff()
        d["roc3"] = d["close"].pct_change(3)
        
        feats = ["adosc_norm", "adosc_slope", "roc3"]
        d[feats] = d[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return d.dropna(subset=feats)
    
    def fit(self, df):
        d = self._feat(df)
        if len(d) < 30: 
            raise ValueError("Not enough rows for ADOSC_Agent")
        
        X = d[["adosc_norm", "adosc_slope", "roc3"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y)
        self.fitted = True
    
    def predict(self, *, current_price, historical_df):
        if not self.fitted: 
            self.fit(historical_df)
        
        last = self._feat(historical_df).iloc[-1:][["adosc_norm", "adosc_slope", "roc3"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1
