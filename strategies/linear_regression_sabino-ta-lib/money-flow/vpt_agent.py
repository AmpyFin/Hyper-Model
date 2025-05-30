"""
VPT Agent
=========

Volume-Price Trend (cumulative):

    VPTₜ = VPTₜ₋₁ + volumeₜ × Δclose% ,
    where Δclose% = (closeₜ − closeₜ₋₁) / closeₜ₋₁.

Features
--------
* vpt_norm   (VPT divided by cumulative volume)
* vpt_slope  (first diff)
"""

from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _vpt(df: pd.DataFrame) -> pd.Series:
    pct = df["close"].pct_change()
    return (df["volume"] * pct).cumsum()


class VPT_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000); self.fitted = False

    def _feat(self, df):
        vpt = _vpt(df)
        d = df.copy()
        d["vpt_norm"] = vpt / df["volume"].cumsum().replace(0, pd.NA)
        d["vpt_slope"] = d["vpt_norm"].diff()
        return d.dropna(subset=["vpt_norm", "vpt_slope"])

    def fit(self, df):
        d = self._feat(df)
        if len(d) < 50:
            raise ValueError("Not enough rows for VPT_Agent")
        X, y = d[["vpt_norm", "vpt_slope"]][:-1], (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        last = self._feat(historical_df).iloc[-1:][["vpt_norm", "vpt_slope"]]
        return 2 * self.m.predict_proba(last)[0, 1] - 1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Volume-Price Trend.
        
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
            print(f"Error in VPT strategy: {str(e)}")
            return 0.0000
