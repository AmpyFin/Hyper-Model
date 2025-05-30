"""
Fourier Agent
~~~~~~~~~~~~
Agent implementing trading strategies based on Joseph Fourier's principles of
harmonic analysis and Fourier transforms.

This agent decomposes price movements into component frequencies using Fourier
analysis, identifying dominant cycles and harmonics that can predict future price
movements based on their phase and amplitude.

Concepts employed:
1. Fast Fourier Transform (FFT) for decomposing price into frequency components
2. Power spectral density analysis to identify dominant cycles
3. Phase analysis to determine cycle positioning
4. Reconstruction of filtered signals for trend/cycle isolation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import logging

logger = logging.getLogger(__name__)

class FourierAgent:
    """
    Trading agent based on Fourier analysis principles.
    
    Parameters
    ----------
    lookback_window : int, default=256
        Window size for Fourier analysis (preferably power of 2)
    top_components : int, default=5
        Number of top frequency components to consider
    noise_threshold : float, default=0.1
        Power threshold for filtering noise (0 to 1)
    forecast_horizon : int, default=20
        Steps ahead to forecast
    detrend : bool, default=True
        Whether to detrend data before analysis
    """
    
    def __init__(
        self,
        lookback_window: int = 256,
        top_components: int = 5,
        noise_threshold: float = 0.1,
        forecast_horizon: int = 20,
        detrend: bool = True
    ):
        self.lookback_window = lookback_window
        self.top_components = top_components
        self.noise_threshold = noise_threshold
        self.forecast_horizon = forecast_horizon
        self.detrend = detrend
        self.latest_signal = 0.0
        self.is_fitted = False
        self.dominant_periods = []
        self.dominant_amplitudes = []
        self.dominant_phases = []
        
    def _prepare_price_series(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare price series for Fourier analysis
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data with 'close' column
            
        Returns
        -------
        numpy.ndarray
            Prepared price series (detrended if specified)
        """
        # Extract close prices
        prices = df['close'].values
        
        # Ensure we have the right amount of data
        if len(prices) > self.lookback_window:
            prices = prices[-self.lookback_window:]
        
        # Pad to power of 2 if needed
        target_len = 2 ** int(np.ceil(np.log2(len(prices))))
        if len(prices) < target_len:
            # Pad with reflection of the data
            pad_len = target_len - len(prices)
            prices = np.pad(prices, (0, pad_len), mode='reflect')
            
        # Apply detrending if specified
        if self.detrend:
            prices = signal.detrend(prices)
            
        return prices
    
    def _perform_fft(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Fast Fourier Transform on price series
        
        Parameters
        ----------
        prices : numpy.ndarray
            Prepared price series
            
        Returns
        -------
        tuple
            (frequencies, amplitudes, phases)
        """
        n = len(prices)
        
        # Perform FFT
        fft_result = fft(prices)
        
        # Calculate frequency bins
        freq = fftfreq(n)
        
        # Calculate amplitude and phase
        amplitude = np.abs(fft_result) / n  # Normalize by signal length
        phase = np.angle(fft_result)
        
        return freq, amplitude, phase
    
    def _identify_dominant_cycles(
        self, 
        freq: np.ndarray, 
        amplitude: np.ndarray, 
        phase: np.ndarray
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Identify dominant cycles from FFT results
        
        Parameters
        ----------
        freq : numpy.ndarray
            Frequency bins
        amplitude : numpy.ndarray
            Amplitude values
        phase : numpy.ndarray
            Phase values
            
        Returns
        -------
        tuple
            (dominant_periods, dominant_amplitudes, dominant_phases)
        """
        n = len(freq)
        
        # Only consider positive frequencies (up to Nyquist frequency)
        pos_range = np.arange(1, n // 2)
        
        # Filter out low amplitude components (noise)
        max_amp = np.max(amplitude[pos_range])
        threshold = max_amp * self.noise_threshold
        valid_indices = pos_range[amplitude[pos_range] > threshold]
        
        # Sort by amplitude
        sorted_indices = valid_indices[np.argsort(-amplitude[valid_indices])]
        
        # Take top components
        top_indices = sorted_indices[:min(self.top_components, len(sorted_indices))]
        
        # Calculate periods, amplitudes, and phases
        periods = 1.0 / np.abs(freq[top_indices])  # Convert frequency to period
        amplitudes = amplitude[top_indices]
        phases = phase[top_indices]
        
        return periods.tolist(), amplitudes.tolist(), phases.tolist()
    
    def _reconstruct_signal(
        self, 
        periods: List[float], 
        amplitudes: List[float], 
        phases: List[float], 
        n_points: int
    ) -> np.ndarray:
        """
        Reconstruct time domain signal from dominant components
        
        Parameters
        ----------
        periods : list
            Periods of dominant cycles
        amplitudes : list
            Amplitudes of dominant cycles
        phases : list
            Phases of dominant cycles
        n_points : int
            Number of points to reconstruct
            
        Returns
        -------
        numpy.ndarray
            Reconstructed signal
        """
        t = np.arange(n_points)
        reconstructed = np.zeros(n_points)
        
        for period, amplitude, phase in zip(periods, amplitudes, phases):
            # Convert period to frequency (cycles per point)
            freq = 1.0 / period
            
            # Add this component's contribution
            reconstructed += amplitude * np.cos(2 * np.pi * freq * t + phase)
            
        return reconstructed
    
    def _forecast_next_points(
        self, 
        periods: List[float], 
        amplitudes: List[float], 
        phases: List[float], 
        current_len: int
    ) -> np.ndarray:
        """
        Forecast future points based on dominant cycles
        
        Parameters
        ----------
        periods : list
            Periods of dominant cycles
        amplitudes : list
            Amplitudes of dominant cycles
        phases : list
            Phases of dominant cycles
        current_len : int
            Current signal length
            
        Returns
        -------
        numpy.ndarray
            Forecasted points
        """
        t = np.arange(current_len, current_len + self.forecast_horizon)
        forecast = np.zeros(self.forecast_horizon)
        
        for period, amplitude, phase in zip(periods, amplitudes, phases):
            # Convert period to frequency (cycles per point)
            freq = 1.0 / period
            
            # Add this component's contribution
            forecast += amplitude * np.cos(2 * np.pi * freq * t + phase)
            
        return forecast
    
    def _calculate_slope_indicators(self, forecast: np.ndarray) -> Dict[str, float]:
        """
        Calculate slope-based indicators from forecast
        
        Parameters
        ----------
        forecast : numpy.ndarray
            Forecasted price points
            
        Returns
        -------
        dict
            Dictionary with slope indicators
        """
        if len(forecast) < 2:
            return {'slope': 0.0, 'acceleration': 0.0, 'r_squared': 0.0}
        
        # Calculate linear trend
        x = np.arange(len(forecast))
        coeffs = np.polyfit(x, forecast, 1)
        slope = coeffs[0]
        
        # Calculate R-squared
        trend_line = np.polyval(coeffs, x)
        ss_total = np.sum((forecast - forecast.mean()) ** 2)
        ss_residual = np.sum((forecast - trend_line) ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        # Calculate acceleration (second derivative approximation)
        if len(forecast) >= 3:
            second_diff = forecast[2:] - 2 * forecast[1:-1] + forecast[:-2]
            acceleration = second_diff.mean()
        else:
            acceleration = 0.0
            
        return {
            'slope': slope,
            'acceleration': acceleration,
            'r_squared': r_squared
        }
    
    def _calculate_cycle_position(self, periods: List[float], phases: List[float]) -> List[float]:
        """
        Calculate current position within each cycle
        
        Parameters
        ----------
        periods : list
            Periods of dominant cycles
        phases : list
            Phases of dominant cycles
            
        Returns
        -------
        list
            Position within cycle (0 to 1) for each component
        """
        positions = []
        
        for period, phase in zip(periods, phases):
            # Convert phase from radians to position in cycle (0 to 1)
            # Adjust for negative phase
            position = ((-phase / (2 * np.pi)) % 1)
            positions.append(position)
            
        return positions
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and perform Fourier analysis
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        if len(historical_df) < min(32, self.lookback_window // 2):
            self.is_fitted = False
            return
            
        try:
            # Prepare price series
            prices = self._prepare_price_series(historical_df)
            
            # Perform FFT
            freq, amplitude, phase = self._perform_fft(prices)
            
            # Identify dominant cycles
            periods, amplitudes, phases = self._identify_dominant_cycles(freq, amplitude, phase)
            
            # Store for later use
            self.dominant_periods = periods
            self.dominant_amplitudes = amplitudes
            self.dominant_phases = phases
            
            # Forecast future points
            forecast = self._forecast_next_points(periods, amplitudes, phases, len(prices))
            
            # Calculate cycle positions
            cycle_positions = self._calculate_cycle_position(periods, phases)
            
            # Calculate slope indicators
            slope_indicators = self._calculate_slope_indicators(forecast)
            
            # Generate trading signal
            
            # 1. Forecast slope component
            slope_signal = np.clip(slope_indicators['slope'] * 100, -1.0, 1.0)
            
            # 2. Cycle position component
            # Weight cycles by amplitude
            total_weight = sum(amplitudes) if amplitudes else 1.0
            weighted_position = 0.0
            
            for pos, amp, period in zip(cycle_positions, amplitudes, periods):
                # Convert position to signal: +1 at beginning of cycle, -1 at middle
                cycle_signal = np.cos(2 * np.pi * pos)
                
                # Weight by relative amplitude and period (shorter periods get less weight)
                period_factor = min(1.0, period / 20)  # Cap weight for very long periods
                weight = (amp / total_weight) * period_factor
                
                weighted_position += cycle_signal * weight
                
            # 3. Combine signals
            r_squared = slope_indicators['r_squared']
            
            # If forecast fit is good, trust the slope more, otherwise trust cycle positions
            final_signal = (slope_signal * r_squared + 
                           weighted_position * (1 - r_squared))
            
            # Clip to valid range
            self.latest_signal = np.clip(final_signal, -1.0, 1.0)
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error in Fourier Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Fourier analysis
        
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
        return "Fourier Agent" 