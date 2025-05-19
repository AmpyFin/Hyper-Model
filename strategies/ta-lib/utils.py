"""
Utilities for TA-Lib Agents
===========================

Common utilities and base classes for all TA-Lib agents.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class BaseAgent:
    """Base class for all TA-Lib agents with robust machine learning setup."""
    
    def __init__(self, max_iter=2000):
        # Use liblinear solver which is more robust for small datasets
        self.m = LogisticRegression(
            max_iter=max_iter, 
            solver='liblinear',  # More robust than default 'lbfgs'
            tol=1e-4,  # Tolerance for stopping criteria
            C=1.0      # Regularization parameter
        )
        self.scaler = StandardScaler()  # For feature scaling
        self.fitted = False
    
    def _scale_features(self, X):
        """Scale features for training or prediction."""
        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler()
        
        # Check if X is a DataFrame or numpy array
        if isinstance(X, pd.DataFrame):
            # Handle empty dataframes
            if X.empty:
                return np.array([])
            # If fitting, learn the scaling parameters
            if not self.fitted:
                return self.scaler.fit_transform(X)
            # Otherwise use existing scaling parameters
            return self.scaler.transform(X)
        else:
            # For numpy arrays
            if len(X) == 0:
                return np.array([])
            
            if not self.fitted:
                return self.scaler.fit_transform(X)
            return self.scaler.transform(X)
    
    def replace_inf_with_nan(self, df, columns):
        """Replace infinities with NaN and handle missing values."""
        if columns and len(columns) > 0:
            df[columns] = df[columns].replace([np.inf, -np.inf], np.nan)
            df[columns] = df[columns].ffill().bfill()
        return df 