"""
ADI Agent
=========

Accumulation / Distribution Index (a.k.a. Acc/Dist Line).

Formula
-------
MF Multiplier  = ((close − low) − (high − close)) / (high − low)  
Money Flow Vol = MF Multiplier × volume  
**ADIₜ**       = Σ Money-Flow-Vol (cumulative)

Features
--------
* adi_norm    (ADI divided by cumulative volume → 0-1-ish)
* adi_slope   (first diff)

Outputs a score in [-1, +1] via LogisticRegression.
"""

from __future__ import annotations
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression


def _adi(df: pd.DataFrame) -> pd.Series:
    
  
    
    # Check if df is actually a DataFrame
    if not isinstance(df, pd.DataFrame):
        print(f"ERROR: Expected DataFrame, got {type(df)}")
        if isinstance(df, list):
            print(f"ERROR: Received list with {len(df)} items")
            if df and isinstance(df[0], dict):
                print(f"ERROR: First item keys: {df[0].keys()}")
                # Try to convert list to DataFrame
                df = pd.DataFrame(df)
            else:
                raise ValueError("Cannot convert list to DataFrame - items are not dictionaries")
        else:
            raise ValueError(f"Expected DataFrame or list of dicts, got {type(df)}")
    
    # Check if required columns exist
    required_columns = ["high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    try:
        high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        mfv = mfm * vol
        return mfv.cumsum()
    except Exception as e:
       
        raise e


class ADI_Agent:
    def __init__(self):
        self.m = LogisticRegression(max_iter=1000)
        self.fitted = False

    def _feat(self, df):
       
        
        adi = _adi(df)
        
        d = df.copy()
        
        d["adi_norm"] = adi / df["volume"].cumsum().replace(0, np.nan)
        
        d["adi_slope"] = d["adi_norm"].diff()
        
        result = d.dropna(subset=["adi_norm", "adi_slope"])
        
        return result

    def fit(self, df):
        d = self._feat(df)
        if len(d) < 50:
            raise ValueError("Not enough rows for ADI_Agent")
        X = d[["adi_norm", "adi_slope"]][:-1]
        y = (d["close"].shift(-1) > d["close"]).astype(int)[:-1]
        self.m.fit(X, y); self.fitted = True

    def predict(self, *, current_price, historical_df):
        if not self.fitted:
            self.fit(historical_df)
        
        try:
            x = self._feat(historical_df).iloc[-1:][["adi_norm", "adi_slope"]]
            
            # Ensure we have valid features
            if len(x) == 0 or x.isnull().any().any():
                return 0.0
                
            # Get prediction probabilities with safe handling
            prob_result = self.m.predict_proba(x)
            
            # Handle different return types (array vs list)
            if hasattr(prob_result, 'shape') and len(prob_result.shape) == 2:
                # Standard numpy array format
                prob_up = float(prob_result[0, 1])
            elif isinstance(prob_result, (list, tuple)) and len(prob_result) > 0:
                # Handle potential list format
                if hasattr(prob_result[0], '__len__') and len(prob_result[0]) > 1:
                    prob_up = float(prob_result[0][1])
                else:
                    prob_up = 0.5
            else:
                # Fallback to neutral
                prob_up = 0.5
                
            # Convert to signal [-1, 1]
            signal = 2 * prob_up - 1
            return max(-1.0, min(1.0, signal))
            
        except Exception:
            # Return neutral signal if there's any error
            return 0.0

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Accumulation/Distribution Index.
        
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
            print(f"Error in ADI strategy: {str(e)}")
            return 0.0000
