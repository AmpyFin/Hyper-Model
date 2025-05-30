"""
Hamming Agent
~~~~~~~~~~~
Agent implementing trading strategies based on Richard Hamming's work on error
detection and correction, digital filters, and numerical methods.

Richard Hamming is known for:
1. Hamming codes for error detection and correction
2. Hamming distance in information theory
3. Hamming window in digital signal processing
4. Numerical methods and error analysis
5. The famous quote "The purpose of computing is insight, not numbers"

This agent models market behavior using:
1. Error detection in price signals
2. Signal correction using Hamming-inspired techniques
3. Digital filtering with Hamming windows
4. Numerical analysis of market data
5. Pattern recognition with error tolerance

Input: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
Output: Signal ∈ [-1.0000, 1.0000] where:
  -1.0000 = Strong sell signal (strong downward trend detected)
  -0.5000 = Weak sell signal (weak downward trend detected)
   0.0000 = Neutral signal (no clear trend)
   0.5000 = Weak buy signal (weak upward trend detected)
   1.0000 = Strong buy signal (strong upward trend detected)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import math
from collections import defaultdict, deque
from scipy import signal as scipy_signal

from ..agent import Agent

logger = logging.getLogger(__name__)

class HammingAgent(Agent):
    """
    Trading agent based on Richard Hamming's error correction and numerical principles.
    
    Parameters
    ----------
    window_size : int, default=30
        Size of Hamming window for signal processing
    error_threshold : float, default=0.1
        Threshold for error detection
    distance_sensitivity : float, default=0.7
        Sensitivity parameter for Hamming distance calculations
    code_redundancy : int, default=3
        Redundancy factor for error correction
    numerical_precision : float, default=1e-6
        Precision for numerical methods
    """
    
    def __init__(
        self,
        window_size: int = 30,
        error_threshold: float = 0.1,
        distance_sensitivity: float = 0.7,
        code_redundancy: int = 3,
        numerical_precision: float = 1e-6
    ):
        self.window_size = window_size
        self.error_threshold = error_threshold
        self.distance_sensitivity = distance_sensitivity
        self.code_redundancy = code_redundancy
        self.numerical_precision = numerical_precision
        self.latest_signal = 0.0
        self.is_fitted = False
        
        # Internal state
        self.error_matrix = None
        self.pattern_library = {}
        self.correction_history = deque(maxlen=20)
        self.signal_quality = 0.0
        self.hamming_weights = None
        
    def _create_hamming_window(self, window_size: int) -> np.ndarray:
        """
        Create a Hamming window for signal processing
        
        Parameters
        ----------
        window_size : int
            Size of the window
            
        Returns
        -------
        numpy.ndarray
            Hamming window weights
        """
        # Use Hamming's window function: w(n) = 0.54 - 0.46 * cos(2πn/N)
        return np.hamming(window_size)
        
    def _hamming_distance(self, sequence1: np.ndarray, sequence2: np.ndarray) -> float:
        """
        Calculate normalized Hamming distance between two price sequences
        
        Parameters
        ----------
        sequence1 : numpy.ndarray
            First price sequence
        sequence2 : numpy.ndarray
            Second price sequence
            
        Returns
        -------
        float
            Normalized Hamming distance
        """
        if len(sequence1) != len(sequence2):
            # Pad the shorter sequence if needed
            max_len = max(len(sequence1), len(sequence2))
            sequence1 = np.pad(sequence1, (0, max_len - len(sequence1)), 'constant')
            sequence2 = np.pad(sequence2, (0, max_len - len(sequence2)), 'constant')
            
        # Normalize sequences to reduce amplitude differences
        if np.std(sequence1) > 0:
            seq1_norm = (sequence1 - np.mean(sequence1)) / np.std(sequence1)
        else:
            seq1_norm = np.zeros_like(sequence1)
            
        if np.std(sequence2) > 0:
            seq2_norm = (sequence2 - np.mean(sequence2)) / np.std(sequence2)
        else:
            seq2_norm = np.zeros_like(sequence2)
            
        # Quantize to simplify comparison (like digital signals)
        bins = 16  # 4-bit quantization
        seq1_digital = np.digitize(seq1_norm, np.linspace(-2, 2, bins))
        seq2_digital = np.digitize(seq2_norm, np.linspace(-2, 2, bins))
        
        # Calculate Hamming distance (count of differing elements)
        distance = np.sum(seq1_digital != seq2_digital) / len(seq1_digital)
        
        return distance
        
    def _error_detection(self, prices: np.ndarray) -> List[int]:
        """
        Detect potential errors in price data using Hamming principles
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        list
            Indices of detected errors
        """
        if len(prices) < 10:
            return []
            
        # Calculate returns as the basis for error detection
        returns = np.diff(prices) / prices[:-1]
        
        # Use rolling statistics to identify outliers
        window = min(len(returns), 20)
        errors = []
        
        for i in range(window, len(returns)):
            # Calculate mean and standard deviation of recent returns
            recent_returns = returns[i-window:i]
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            
            # Check if current return is an outlier (potential error)
            if std_return > 0 and abs(returns[i] - mean_return) > self.error_threshold * std_return:
                errors.append(i+1)  # +1 because returns are diff of prices
                
        return errors
        
    def _error_correction(self, prices: np.ndarray, error_indices: List[int]) -> np.ndarray:
        """
        Apply error correction to price data (like Hamming codes)
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        error_indices : list
            Indices of detected errors
            
        Returns
        -------
        numpy.ndarray
            Corrected price data
        """
        if not error_indices or len(prices) < 10:
            return prices
            
        corrected_prices = prices.copy()
        
        for idx in error_indices:
            if idx > 0 and idx < len(prices) - 1:
                # Apply simple correction: average of neighbors
                corrected_prices[idx] = (prices[idx-1] + prices[idx+1]) / 2
                
                # Record correction amount
                correction_amount = abs(corrected_prices[idx] - prices[idx]) / prices[idx]
                self.correction_history.append(correction_amount)
                
        return corrected_prices
        
    def _apply_hamming_window(self, prices: np.ndarray) -> np.ndarray:
        """
        Apply Hamming window to price data for spectral analysis
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        numpy.ndarray
            Windowed price data
        """
        if len(prices) < self.window_size:
            return prices
            
        # Ensure we have Hamming weights
        if self.hamming_weights is None or len(self.hamming_weights) != self.window_size:
            self.hamming_weights = self._create_hamming_window(self.window_size)
            
        # Apply window to most recent data
        recent_prices = prices[-self.window_size:]
        
        # Detrend the data before applying window
        x = np.arange(len(recent_prices))
        slope, intercept = np.polyfit(x, recent_prices, 1)
        trend = slope * x + intercept
        detrended = recent_prices - trend
        
        # Apply Hamming window
        windowed = detrended * self.hamming_weights
        
        # Return windowed data + trend to maintain original scale
        return windowed + trend
        
    def _spectral_analysis(self, windowed_prices: np.ndarray) -> Dict[str, Any]:
        """
        Perform spectral analysis on windowed price data
        
        Parameters
        ----------
        windowed_prices : numpy.ndarray
            Windowed price data
            
        Returns
        -------
        dict
            Spectral analysis results
        """
        if len(windowed_prices) < self.window_size // 2:
            return {}
            
        # Compute FFT
        fft_result = np.fft.rfft(windowed_prices)
        fft_freqs = np.fft.rfftfreq(len(windowed_prices))
        
        # Get magnitude spectrum
        magnitude = np.abs(fft_result)
        
        # Find dominant frequencies
        dominant_idx = np.argsort(magnitude)[-3:]  # Top 3 frequencies
        dominant_freqs = fft_freqs[dominant_idx]
        dominant_mags = magnitude[dominant_idx]
        
        # Calculate spectral entropy
        normalized_magnitude = magnitude / np.sum(magnitude)
        spectral_entropy = -np.sum(normalized_magnitude * np.log2(normalized_magnitude + 1e-10))
        
        # Calculate signal-to-noise ratio
        if len(dominant_idx) > 0 and np.sum(magnitude) > 0:
            signal_power = np.sum(magnitude[dominant_idx])
            noise_power = np.sum(magnitude) - signal_power
            snr = signal_power / noise_power if noise_power > 0 else 100.0
        else:
            snr = 0.0
            
        # Store signal quality
        self.signal_quality = min(1.0, snr / 10)
        
        return {
            'dominant_frequencies': dominant_freqs,
            'dominant_magnitudes': dominant_mags,
            'spectral_entropy': spectral_entropy,
            'snr': snr
        }
        
    def _find_similar_patterns(self, prices: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Find historical patterns similar to current pattern using Hamming distance
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
            
        Returns
        -------
        list
            List of (pattern, distance) tuples
        """
        if len(prices) < self.window_size:
            return []
            
        # Current pattern is the most recent window
        current_pattern = prices[-self.window_size:]
        
        # Search for similar patterns in history
        similar_patterns = []
        
        for i in range(self.window_size, len(prices) - self.window_size):
            historical_pattern = prices[i-self.window_size:i]
            
            # Calculate Hamming distance
            distance = self._hamming_distance(current_pattern, historical_pattern)
            
            # Store pattern if it's similar enough
            if distance < self.distance_sensitivity:
                # Include what happened after the pattern
                future_pattern = prices[i:i+self.window_size//2]
                similar_patterns.append((future_pattern, distance))
                
        # Sort by distance (most similar first)
        similar_patterns.sort(key=lambda x: x[1])
        
        # Return top matches
        return similar_patterns[:5]
        
    def _numerical_extrapolation(self, prices: np.ndarray, similar_patterns: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Use numerical methods to extrapolate future prices
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        similar_patterns : list
            List of similar historical patterns
            
        Returns
        -------
        numpy.ndarray
            Extrapolated future prices
        """
        if not similar_patterns or len(prices) < self.window_size:
            # Fallback to simple linear extrapolation
            horizon = self.window_size // 4
            x = np.arange(len(prices[-self.window_size:]))
            slope, intercept = np.polyfit(x, prices[-self.window_size:], 1)
            future_x = np.arange(len(x), len(x) + horizon)
            return slope * future_x + intercept
            
        # Weight patterns by inverse distance
        weights = [1 / (d + self.numerical_precision) for _, d in similar_patterns]
        normalized_weights = [w / sum(weights) for w in weights]
        
        # Calculate weighted average of future patterns
        horizon = min(len(similar_patterns[0][0]), self.window_size // 4)
        future_prices = np.zeros(horizon)
        
        for i, ((pattern, _), weight) in enumerate(zip(similar_patterns, normalized_weights)):
            pattern_horizon = min(len(pattern), horizon)
            future_prices[:pattern_horizon] += pattern[:pattern_horizon] * weight
            
        # Scale to match the current price level
        current_price = prices[-1]
        first_future_price = future_prices[0]
        scaling_factor = current_price / first_future_price if first_future_price != 0 else 1.0
        
        return future_prices * scaling_factor
        
    def _generate_signal_from_prediction(self, prices: np.ndarray, predicted_prices: np.ndarray, spectral_info: Dict[str, Any]) -> float:
        """
        Generate trading signal from price prediction and spectral analysis
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of price values
        predicted_prices : numpy.ndarray
            Predicted future prices
        spectral_info : dict
            Spectral analysis results
            
        Returns
        -------
        float
            Trading signal
        """
        if len(predicted_prices) < 3 or len(prices) < 3:
            return 0.0
            
        # Calculate predicted return
        current_price = prices[-1]
        future_price = predicted_prices[-1]  # End of prediction horizon
        predicted_return = future_price / current_price - 1
        
        # Generate base signal from prediction
        base_signal = np.clip(predicted_return * 10, -1.0, 1.0)  # Scale to [-1, 1]
        
        # Adjust signal based on spectral properties
        if 'snr' in spectral_info and 'spectral_entropy' in spectral_info:
            # Higher SNR = more confidence in signal
            snr_factor = min(1.0, spectral_info['snr'] / 10)
            
            # Lower entropy = more predictable pattern
            entropy_norm = spectral_info['spectral_entropy'] / math.log2(self.window_size // 2 + 1)
            entropy_factor = 1.0 - entropy_norm
            
            # Adjust signal by confidence factors
            confidence = 0.5 * snr_factor + 0.5 * entropy_factor
            adjusted_signal = base_signal * confidence
        else:
            adjusted_signal = base_signal * 0.5  # Reduced confidence without spectral info
            
        return adjusted_signal
        
    def _hamming_error_code(self, signal: float) -> Tuple[List[int], float]:
        """
        Apply Hamming-like error coding to trading signal for robustness
        
        Parameters
        ----------
        signal : float
            Raw trading signal
            
        Returns
        -------
        tuple
            (error_code, robust_signal)
        """
        # Convert signal to binary representation with redundancy
        # This is inspired by Hamming codes but highly simplified
        
        # Scale signal to [0, 1] for encoding
        scaled_signal = (signal + 1) / 2
        
        # Create code with redundancy
        code = []
        for i in range(self.code_redundancy):
            # Each copy has slight intentional variations to improve robustness
            variation = scaled_signal * (1 + (i - self.code_redundancy // 2) * 0.05)
            code_bit = 1 if variation > 0.5 else 0
            code.append(code_bit)
            
        # Decode with majority vote
        robust_bit = 1 if sum(code) > len(code) / 2 else 0
        
        # Convert back to signal scale
        robust_signal = robust_bit * 2 - 1  # Maps 0->-1, 1->+1
        
        # If signal is weak, dampen it further to reduce false positives
        if abs(signal) < 0.3:
            robust_signal *= abs(signal) / 0.3
            
        return code, robust_signal
        
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data using Hamming's error correction and signal processing principles
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        # Reduce minimum requirement to be more flexible
        min_required = max(self.window_size, 25)
        if len(historical_df) < min_required:
            self.is_fitted = False
            return
            
        try:
            # Extract price data
            prices = historical_df['close'].values
            
            # Initialize components
            corrected_prices = prices
            spectral_info = {}
            similar_patterns = []
            predicted_prices = np.array([])
            base_signal = 0.0
            
            # 1. Error detection and correction
            try:
                error_indices = self._error_detection(prices)
                corrected_prices = self._error_correction(prices, error_indices)
            except Exception as e:
                logger.warning(f"Error detection/correction failed: {e}")
                corrected_prices = prices
            
            # 2. Apply Hamming window for spectral analysis
            try:
                windowed_prices = self._apply_hamming_window(corrected_prices)
            except Exception as e:
                logger.warning(f"Hamming window application failed: {e}")
                windowed_prices = corrected_prices
            
            # 3. Perform spectral analysis
            try:
                spectral_info = self._spectral_analysis(windowed_prices)
            except Exception as e:
                logger.warning(f"Spectral analysis failed: {e}")
                spectral_info = {}
            
            # 4. Find similar historical patterns
            try:
                similar_patterns = self._find_similar_patterns(corrected_prices)
            except Exception as e:
                logger.warning(f"Pattern finding failed: {e}")
                similar_patterns = []
            
            # 5. Numerical extrapolation
            try:
                predicted_prices = self._numerical_extrapolation(corrected_prices, similar_patterns)
            except Exception as e:
                logger.warning(f"Numerical extrapolation failed: {e}")
                predicted_prices = np.array([])
            
            # 6. Generate base signal
            try:
                if len(predicted_prices) > 0:
                    base_signal = self._generate_signal_from_prediction(
                        corrected_prices, predicted_prices, spectral_info
                    )
                else:
                    # Fallback: simple momentum signal
                    if len(corrected_prices) >= 10:
                        recent_returns = np.diff(corrected_prices[-10:]) / corrected_prices[-11:-1]
                        clean_returns = recent_returns[np.isfinite(recent_returns)]
                        if len(clean_returns) > 0:
                            momentum = np.mean(clean_returns)
                            if np.isfinite(momentum):
                                base_signal = np.sign(momentum) * min(0.3, abs(momentum) * 10)
                
                if not np.isfinite(base_signal):
                    base_signal = 0.0
                    
            except Exception as e:
                logger.warning(f"Signal generation failed: {e}")
                base_signal = 0.0
            
            # 7. Apply error coding for robustness
            try:
                code, robust_signal = self._hamming_error_code(base_signal)
                if np.isfinite(robust_signal):
                    self.latest_signal = robust_signal
                else:
                    self.latest_signal = base_signal
            except Exception as e:
                logger.warning(f"Error coding failed: {e}")
                self.latest_signal = base_signal
            
            # Ensure signal is finite and in range
            if not np.isfinite(self.latest_signal):
                self.latest_signal = 0.0
            else:
                self.latest_signal = np.clip(self.latest_signal, -1.0, 1.0)
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Hamming Agent fit: {e}")
            self.latest_signal = 0.0
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Hamming's error correction principles
        
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
        return "Hamming Agent"

    def strategy(self, historical_df: pd.DataFrame) -> float:
        """
        Generate a trading signal using Hamming's error correction principles.
        
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
            # Process the data using existing workflow
            self.fit(historical_df)
            
            if not self.is_fitted:
                return 0.0000
                
            # Get current price for prediction
            current_price = historical_df['close'].iloc[-1]
            
            # Generate signal using existing predict method
            signal = self.predict(current_price=current_price, historical_df=historical_df)
            
            # Ensure signal is properly formatted to 4 decimal places
            return float(round(signal, 4))
            
        except ValueError as e:
            # Handle the case where there's not enough data
            logger.error(f"ValueError in Hamming strategy: {str(e)}")
            return 0.0000
        except Exception as e:
            # Handle any other unexpected errors
            logger.error(f"Error in Hamming strategy: {str(e)}")
            return 0.0000 