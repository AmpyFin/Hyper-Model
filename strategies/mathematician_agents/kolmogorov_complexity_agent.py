"""
Kolmogorov Agent
~~~~~~~~~~~~~~~
Agent implementing trading strategies based on Andrey Kolmogorov's complexity theory
and algorithmic information theory principles.

This agent analyzes price sequences in terms of their algorithmic complexity, trying
to identify patterns that are simpler (more predictable) versus those that are more 
random and complex.

Key concepts:
1. Kolmogorov Complexity: Measuring the randomness of price sequences
2. Minimum Description Length: Finding simplest models to explain price movements
3. Complexity-based trend detection: Identifying regime changes through complexity shifts
4. Approximate entropy analysis: Estimating the complexity of time series
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class KolmogorovAgent:
    """
    Trading agent based on Kolmogorov's complexity theory principles.
    
    Parameters
    ----------
    complexity_window : int, default=30
        Window size for complexity calculations
    comparison_window : int, default=10
        Window size for comparing current complexity to historical
    embedding_dim : int, default=2
        Embedding dimension for approximate entropy
    similarity_threshold : float, default=0.2
        Threshold for similarity in approximate entropy calculation
    smoothing_window : int, default=5
        Window size for smoothing signals
    """
    
    def __init__(
        self,
        complexity_window: int = 30,
        comparison_window: int = 10,
        embedding_dim: int = 2,
        similarity_threshold: float = 0.2,
        smoothing_window: int = 5
    ):
        self.complexity_window = complexity_window
        self.comparison_window = comparison_window
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.smoothing_window = smoothing_window
        self.latest_signal = 0.0
        self.is_fitted = False
        
    def _sample_entropy(self, time_series: np.ndarray) -> float:
        """
        Calculate sample entropy as an approximation of Kolmogorov complexity
        
        Parameters
        ----------
        time_series : numpy.ndarray
            Input time series
            
        Returns
        -------
        float
            Sample entropy value
        """
        if len(time_series) < self.embedding_dim + 2:
            return 0.0
        
        # Normalize the time series
        time_series = (time_series - np.mean(time_series)) / (np.std(time_series) + 1e-10)
        
        n = len(time_series)
        m = self.embedding_dim
        r = self.similarity_threshold
        
        # Create embedding vectors
        def create_vectors(m_dim):
            vectors = []
            for i in range(n - m_dim + 1):
                vectors.append(time_series[i:i+m_dim])
            return np.array(vectors)
        
        # Count similar patterns
        def count_matches(vectors, threshold):
            n_vectors = len(vectors)
            count = 0
            for i in range(n_vectors):
                # Calculate distances, excluding self-match
                distances = np.max(np.abs(vectors - vectors[i]), axis=1)
                # Count matches within threshold, excluding self
                count += np.sum(distances < threshold) - 1
            return count / (n_vectors * (n_vectors - 1))
        
        # Create embedding vectors for dimensions m and m+1
        emb_m = create_vectors(m)
        emb_m_plus = create_vectors(m + 1)
        
        # Count matches for each dimension
        b_m = count_matches(emb_m, r)
        b_m_plus = count_matches(emb_m_plus, r)
        
        # Calculate sample entropy
        if b_m == 0 or b_m_plus == 0:
            return 2.0  # High entropy value
        
        return -np.log(b_m_plus / b_m)
    
    def _compression_ratio(self, sequence: np.ndarray) -> float:
        """
        Estimate complexity by looking at the 'compressibility' of a sequence
        
        Parameters
        ----------
        sequence : numpy.ndarray
            Input sequence
            
        Returns
        -------
        float
            Compression ratio estimate
        """
        # Discretize the sequence into bins
        n_bins = min(20, len(sequence) // 4)
        if n_bins < 2:
            return 0.5
            
        bins = np.linspace(min(sequence), max(sequence), n_bins)
        digitized = np.digitize(sequence, bins)
        
        # Count run lengths as a simple compression method
        runs = []
        current_value = digitized[0]
        current_count = 1
        
        for i in range(1, len(digitized)):
            if digitized[i] == current_value:
                current_count += 1
            else:
                runs.append((current_value, current_count))
                current_value = digitized[i]
                current_count = 1
                
        runs.append((current_value, current_count))
        
        # Calculate compression ratio
        original_size = len(digitized)
        compressed_size = len(runs) * 2  # Each run needs (value, count)
        
        # Normalize to [0, 1] where 0 is incompressible (random) and 1 is fully compressible
        ratio = 1.0 - (compressed_size / (original_size + 1))
        return max(0.0, min(1.0, ratio))
    
    def _complexity_metrics(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate various complexity metrics for the price series
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data with at minimum 'close' column
            
        Returns
        -------
        dict
            Dictionary of complexity metrics arrays
        """
        # Make a copy to avoid warnings
        df_copy = df.copy()
        n = len(df_copy)
        
        # Initialize arrays
        sample_entropy_series = np.zeros(n)
        compression_ratio_series = np.zeros(n)
        trend_complexity_ratio = np.zeros(n)
        
        # Calculate returns
        df_copy['returns'] = df_copy['close'].pct_change()
        
        # Calculate rolling complexity metrics
        for i in range(self.complexity_window, n):
            window = slice(i - self.complexity_window, i)
            
            # Sample entropy on returns
            returns = df_copy['returns'].values[window]
            returns = returns[~np.isnan(returns)]
            if len(returns) >= self.embedding_dim + 2:
                sample_entropy_series[i] = self._sample_entropy(returns)
            
            # Compression ratio on prices
            prices = df_copy['close'].values[window]
            compression_ratio_series[i] = self._compression_ratio(prices)
            
            # Calculate trend complexity ratio
            # Lower ratio means trend is more predictable than noise
            if i >= self.complexity_window * 2:
                long_window = slice(i - self.complexity_window * 2, i)
                short_window = slice(i - self.complexity_window, i)
                
                trend = df_copy['close'].rolling(window=self.complexity_window//2).mean().values[long_window]
                trend_entropy = self._sample_entropy(trend[~np.isnan(trend)])
                
                residuals = df_copy['close'].values[short_window] - trend[-self.complexity_window:]
                residual_entropy = self._sample_entropy(residuals[~np.isnan(residuals)])
                
                # Ratio of trend complexity to noise complexity
                if residual_entropy > 0:
                    trend_complexity_ratio[i] = trend_entropy / residual_entropy
                else:
                    trend_complexity_ratio[i] = 1.0
        
        # Smooth the metrics
        sample_entropy_series = pd.Series(sample_entropy_series).rolling(self.smoothing_window).mean().values
        compression_ratio_series = pd.Series(compression_ratio_series).rolling(self.smoothing_window).mean().values
        trend_complexity_ratio = pd.Series(trend_complexity_ratio).rolling(self.smoothing_window).mean().values
        
        return {
            'sample_entropy': sample_entropy_series,
            'compression_ratio': compression_ratio_series,
            'trend_complexity_ratio': trend_complexity_ratio
        }
    
    def _detect_regime_changes(self, complexity_metrics: Dict[str, np.ndarray]) -> float:
        """
        Detect market regime changes based on complexity shifts
        
        Parameters
        ----------
        complexity_metrics : dict
            Dictionary of complexity metrics arrays
            
        Returns
        -------
        float
            Regime change indicator [-1, 1] where 0 is no change
        """
        n = len(complexity_metrics['sample_entropy'])
        if n < self.complexity_window + self.comparison_window:
            return 0.0
            
        # Get current and previous window values
        current_entropy = complexity_metrics['sample_entropy'][-self.comparison_window:]
        previous_entropy = complexity_metrics['sample_entropy'][-(self.comparison_window*2):-self.comparison_window]
        
        current_compression = complexity_metrics['compression_ratio'][-self.comparison_window:]
        previous_compression = complexity_metrics['compression_ratio'][-(self.comparison_window*2):-self.comparison_window]
        
        # Compare distributions
        entropy_change = np.mean(current_entropy) - np.mean(previous_entropy)
        compression_change = np.mean(current_compression) - np.mean(previous_compression)
        
        # If entropy decreases and compression increases, the market is becoming more predictable
        # If entropy increases and compression decreases, the market is becoming more random
        regime_indicator = -entropy_change + compression_change
        
        # Scale to [-1, 1]
        return np.clip(regime_indicator * 5, -1.0, 1.0)
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate complexity metrics
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < self.complexity_window * 2:
            self.is_fitted = False
            return
            
        try:
            # Calculate complexity metrics
            complexity_metrics = self._complexity_metrics(historical_df)
            
            # Detect regime changes
            regime_indicator = self._detect_regime_changes(complexity_metrics)
            
            # Get latest metrics
            latest_entropy = complexity_metrics['sample_entropy'][-1]
            latest_compression = complexity_metrics['compression_ratio'][-1]
            latest_trend_ratio = complexity_metrics['trend_complexity_ratio'][-1]
            
            # Calculate recent price trend
            recent_returns = historical_df['close'].pct_change().dropna().tail(self.comparison_window).values
            recent_trend = np.mean(recent_returns) if len(recent_returns) > 0 else 0
            
            # Generate signal based on complexity and trend
            # In more predictable markets (low entropy, high compression), follow the trend
            # In more random markets, fade the trend
            market_predictability = latest_compression * (1 - min(1.0, latest_entropy / 2))
            
            # Adjust trend confidence based on market predictability and trend complexity ratio
            if latest_trend_ratio < 0.5:  # Trend is more predictable than noise
                trend_confidence = recent_trend * (2.0 - latest_trend_ratio) * 5
            else:  # Noise dominates
                trend_confidence = recent_trend * (1.0 / (latest_trend_ratio + 0.5)) * 3
            
            # Combine signals: regime change indicator and trend confidence
            combined_signal = (regime_indicator * 0.7) + (trend_confidence * 0.3)
            
            # Scale final signal to [-1, 1]
            self.latest_signal = np.clip(combined_signal, -1.0, 1.0)
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Kolmogorov Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Kolmogorov complexity principles
        
        Parameters
        ----------
        current_price : float
            Current asset price
        historical_df : pandas.DataFrame
            Historical price data
            
        Returns
        -------
        float
            Signal value in range [-1, +1]
        """
        # Process the data
        self.fit(historical_df)
        
        if not self.is_fitted:
            return 0.0
            
        return self.latest_signal
    
    def __str__(self) -> str:
        return "Kolmogorov Agent" 