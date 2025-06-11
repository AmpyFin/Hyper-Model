"""
MFI Agent
=========

Money Flow Index (volume-weighted RSI) – default window **14**.

Computation
-----------
1. Typical Price   TPₜ = (high + low + close) / 3  
2. Raw Flow        MFₜ = TPₜ × volumeₜ  
3. Positive / Negative flow buckets by price change  
4. MFIₜ = 100 − 100 / (1 + Σ⁺MF / Σ⁻MF)  over *n* periods

Features
--------
* **mfi**            (scaled 0‒1 by /100)
* **mfi_slope**      first difference of mfi

A tiny *LogisticRegression* turns the features into a score in **[−1, +1]**.
"""

from __future__ import annotations
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression


# ────────────────── helper: rolling MFI ──────────────────
def _mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"]

    up = tp > tp.shift(1)
    down = tp < tp.shift(1)

    pos_flow = mf.where(up, 0.0)
    neg_flow = mf.where(down, 0.0)

    pos_roll = pos_flow.rolling(n).sum()
    neg_roll = neg_flow.rolling(n).sum().replace(0, np.nan)  # avoid ÷0

    mfi = 100 - (100 / (1 + pos_roll / neg_roll))
    return mfi


# ───────────────────────── agent ─────────────────────────
class MFI_Agent:
    """Volume-momentum learner based on Money-Flow Index."""

    def __init__(self, period: int = 14):
        self.n = period
        self.model = LogisticRegression(max_iter=1000)
        self.fitted = False

    # --------------- feature engineering --------------- #
    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        needed = {"high", "low", "close", "volume"}
        if not needed.issubset(df.columns):
            raise ValueError(f"DataFrame must include {needed}")

        mfi = _mfi(df, self.n)
        out = df.copy()
        out["mfi"] = mfi / 100.0          # scale to 0-1
        out["mfi_slope"] = out["mfi"].diff()
        return out.dropna(subset=["mfi", "mfi_slope"])

    # --------------------- training -------------------- #
    def fit(self, ohlcv: pd.DataFrame) -> None:
        d = self._features(ohlcv)
        if len(d) < self.n + 10:
            raise ValueError("Not enough rows to train MFI_Agent")

        X = d[["mfi", "mfi_slope"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]

        self.model.fit(X, y)
        self.fitted = True

    # --------------------- predict --------------------- #
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        if not self.fitted:
            self.fit(historical_df)

        last = self._features(historical_df).iloc[-1:][["mfi", "mfi_slope"]]
        prob_up = self.model.predict_proba(last)[0, 1]
        return 2 * prob_up - 1        # map to −1…+1

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Money Flow Index.
        
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
            print(f"Error in MFI strategy: {str(e)}")
            return 0.0000


