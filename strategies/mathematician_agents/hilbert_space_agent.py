"""
Hilbert Agent
~~~~~~~~~~~~
Agent implementing trading strategies based on David Hilbert's principles of
functional analysis, Hilbert spaces, and infinite-dimensional mathematics.

The agent uses concepts from Hilbert's work to:
1. Project price movements into higher-dimensional Hilbert spaces
2. Apply Hilbert transforms for phase and amplitude decomposition
3. Identify orthogonal market factors using Gram-Schmidt process
4. Detect regime changes through spectral analysis of market states

This enables the agent to separate signal from noise and identify underlying
market structure across multiple timescales.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import signal, linalg, stats
import logging
from scipy.signal import hilbert as hilbert_transform

logger = logging.getLogger(__name__)

class HilbertAgent:
    """
    Trading agent based on Hilbert's mathematical principles.
    
    Parameters
    ----------
    dimension_count : int, default=5
        Number of dimensions to use in Hilbert space projection
    analytic_window : int, default=30
        Window size for Hilbert transform analysis
    orthogonalization_period : int, default=50
        Period for orthogonalizing market factors
    phase_significance : float, default=0.7
        Significance threshold for phase-based signals
    spectrum_bins : int, default=8
        Number of frequency bins in spectral analysis
    """
    
    def __init__(
        self,
        dimension_count: int = 5,
        analytic_window: int = 30,
        orthogonalization_period: int = 50,
        phase_significance: float = 0.7,
        spectrum_bins: int = 8
    ):
        self.dimension_count = dimension_count
        self.analytic_window = analytic_window
        self.orthogonalization_period = orthogonalization_period
        self.phase_significance = phase_significance
        self.spectrum_bins = spectrum_bins
        self.latest_signal = 0.0
        self.is_fitted = False
        self.space_components = {}
        
    def _create_feature_space(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create a multi-dimensional feature space from price data
        
        Parameters
        ----------
        df : pandas.DataFrame
            Price data with OHLCV columns
            
        Returns
        -------
        numpy.ndarray
            Matrix of features (dimensions × time)
        """
        n = len(df)
        if n < self.orthogonalization_period:
            return np.zeros((1, n))
            
        # Create feature matrix with different aspects of price action
        features = []
        
        # 1. Price levels (normalized)
        if 'close' in df.columns:
            close_norm = (df['close'] - df['close'].mean()) / df['close'].std()
            features.append(close_norm.values)
        
        # 2. Returns at different timescales
        for period in [1, 5, 10, 20]:
            if n > period:
                returns = df['close'].pct_change(periods=period).values
                returns[np.isnan(returns)] = 0
                features.append(returns)
        
        # 3. Volatility (rolling standard deviation)
        if n >= 10:
            volatility = df['close'].rolling(window=10).std().values
            volatility[np.isnan(volatility)] = 0
            volatility = (volatility - np.mean(volatility)) / np.std(volatility) if np.std(volatility) > 0 else volatility
            features.append(volatility)
        
        # 4. Volume (if available)
        if 'volume' in df.columns:
            volume_norm = (df['volume'] - df['volume'].mean()) / df['volume'].std() if df['volume'].std() > 0 else 0
            features.append(volume_norm)
        
        # 5. Price range as percentage
        if all(col in df.columns for col in ['high', 'low', 'close']):
            price_range = (df['high'] - df['low']) / df['close']
            price_range = (price_range - price_range.mean()) / price_range.std() if price_range.std() > 0 else 0
            features.append(price_range.values)
            
        # Convert to array and limit to specified dimension count
        feature_array = np.array(features)
        if feature_array.shape[0] > self.dimension_count:
            feature_array = feature_array[:self.dimension_count, :]
        
        return feature_array
    
    def _orthogonalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply Gram-Schmidt orthogonalization process to feature space
        
        Parameters
        ----------
        features : numpy.ndarray
            Matrix of features (dimensions × time)
            
        Returns
        -------
        numpy.ndarray
            Orthogonalized feature space
        """
        if features.shape[0] <= 1:
            return features
            
        # Extract recent data for orthogonalization
        recent_features = features[:, -self.orthogonalization_period:]
        
        # Transpose to get shape (time × dimensions) for linear algebra operations
        X = recent_features.T
        
        # Apply QR decomposition (based on Gram-Schmidt)
        Q, R = np.linalg.qr(X)
        
        # Q contains orthogonalized vectors
        # Transpose back to (dimensions × time)
        ortho_recent = Q.T
        
        # Apply transformation to entire feature set
        # This involves solving for the transformation matrix and applying it
        
        # Calculate transformation matrix from original to orthogonal space
        # (using recent data only)
        # Solve X = QA for A
        A, residuals, rank, s = np.linalg.lstsq(recent_features.T, ortho_recent.T, rcond=None)
        
        # Apply transformation to all data
        ortho_all = np.dot(features.T, A).T
        
        return ortho_all
    
    def _apply_hilbert_transform(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Hilbert transform to decompose signal into amplitude and phase
        
        Parameters
        ----------
        series : numpy.ndarray
            Time series to analyze
            
        Returns
        -------
        tuple
            (analytic_signal, amplitude_envelope, instantaneous_phase)
        """
        # Remove NaN values
        clean_series = series.copy()
        clean_series[np.isnan(clean_series)] = 0
        
        # Apply Hilbert transform to get analytic signal
        analytic_signal = hilbert_transform(clean_series)
        
        # Extract amplitude envelope
        amplitude_envelope = np.abs(analytic_signal)
        
        # Extract instantaneous phase
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
        return analytic_signal, amplitude_envelope, instantaneous_phase
    
    def _detect_phase_shifts(self, phases: np.ndarray) -> List[Tuple[int, float]]:
        """
        Detect significant shifts in signal phase
        
        Parameters
        ----------
        phases : numpy.ndarray
            Array of unwrapped phase values
            
        Returns
        -------
        list
            List of (position, significance) tuples for phase shifts
        """
        n = len(phases)
        if n < 10:
            return []
            
        # Calculate first derivative of phase (instantaneous frequency)
        phase_velocity = np.diff(phases)
        
        # Calculate mean and standard deviation of phase velocity
        mean_velocity = np.mean(phase_velocity)
        std_velocity = np.std(phase_velocity)
        
        # Detect significant deviations in phase velocity
        shifts = []
        for i in range(1, len(phase_velocity)):
            # Calculate Z-score
            z_score = abs(phase_velocity[i] - mean_velocity) / std_velocity if std_velocity > 0 else 0
            
            # If significant deviation and local maximum/minimum
            if z_score > 2.0:
                if i > 0 and i < len(phase_velocity) - 1:
                    if (phase_velocity[i] > phase_velocity[i-1] and 
                        phase_velocity[i] > phase_velocity[i+1]):
                        # Local maximum
                        shifts.append((i, min(1.0, z_score / 5.0)))
                    elif (phase_velocity[i] < phase_velocity[i-1] and 
                          phase_velocity[i] < phase_velocity[i+1]):
                        # Local minimum
                        shifts.append((i, min(1.0, z_score / 5.0)))
        
        return shifts
    
    def _spectral_analysis(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform spectral analysis of market state
        
        Parameters
        ----------
        features : numpy.ndarray
            Orthogonalized feature space
            
        Returns
        -------
        dict
            Dictionary of spectral properties
        """
        if features.shape[1] < self.analytic_window:
            return {}
            
        result = {}
        
        # Extract recent window for analysis
        recent_features = features[:, -self.analytic_window:]
        
        # Calculate power spectral density for each dimension
        psd_matrix = np.zeros((features.shape[0], self.spectrum_bins))
        
        for i in range(features.shape[0]):
            # Calculate FFT
            fft_vals = np.fft.rfft(recent_features[i, :])
            
            # Get power spectrum
            power = np.abs(fft_vals) ** 2
            
            # Bin the power spectrum
            if len(power) > self.spectrum_bins:
                # Reshape to desired number of bins by averaging
                bin_size = len(power) // self.spectrum_bins
                binned_power = np.array([np.mean(power[j*bin_size:(j+1)*bin_size]) 
                                    for j in range(self.spectrum_bins)])
                psd_matrix[i, :] = binned_power
            elif len(power) > 0:
                # If not enough points, pad with zeros
                psd_matrix[i, :len(power)] = power
        
        # Normalize PSD
        for i in range(features.shape[0]):
            if np.sum(psd_matrix[i, :]) > 0:
                psd_matrix[i, :] = psd_matrix[i, :] / np.sum(psd_matrix[i, :])
        
        result['psd_matrix'] = psd_matrix
        
        # Calculate dominant frequencies for each dimension
        dominant_freqs = np.argmax(psd_matrix, axis=1)
        result['dominant_freqs'] = dominant_freqs
        
        # Calculate spectral entropy (higher = more random, lower = more predictable)
        spectral_entropy = np.zeros(features.shape[0])
        for i in range(features.shape[0]):
            # Normalize PSD for entropy calculation
            p = psd_matrix[i, :]
            p_norm = p / np.sum(p) if np.sum(p) > 0 else p
            
            # Calculate entropy
            entropy = -np.sum(p_norm * np.log2(p_norm + 1e-10))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(self.spectrum_bins)
            spectral_entropy[i] = entropy / max_entropy if max_entropy > 0 else 0
            
        result['spectral_entropy'] = spectral_entropy
        
        return result
    
    def _generate_signal_from_hilbert_space(
        self,
        features: np.ndarray,
        ortho_features: np.ndarray,
        phases: List[np.ndarray],
        amplitudes: List[np.ndarray],
        spectral_data: Dict[str, np.ndarray]
    ) -> float:
        """
        Generate trading signal based on Hilbert space analysis
        
        Parameters
        ----------
        features : numpy.ndarray
            Original feature space
        ortho_features : numpy.ndarray
            Orthogonalized feature space
        phases : list
            List of phase arrays for each dimension
        amplitudes : list
            List of amplitude arrays for each dimension
        spectral_data : dict
            Spectral analysis results
            
        Returns
        -------
        float
            Signal value in range [-1, +1]
        """
        if not phases or not amplitudes:
            return 0.0
            
        signal_components = []
        weights = []
        
        # 1. Phase-based signals (40%)
        phase_signals = []
        for dim in range(min(len(phases), ortho_features.shape[0])):
            if len(phases[dim]) < 2:
                continue
                
            # Recent phase velocity (first derivative of phase)
            recent_velocity = np.diff(phases[dim])[-5:]
            
            # Acceleration (second derivative)
            phase_accel = np.diff(np.concatenate([[0], recent_velocity]))
            
            # Generate signal based on phase acceleration
            # Positive acceleration suggests increasing momentum
            phase_signal = np.sign(np.mean(phase_accel)) * min(1.0, abs(np.mean(phase_accel)) * 5)
            
            # Weight by amplitude (higher amplitude = stronger signal)
            if len(amplitudes[dim]) > 0:
                recent_amp = amplitudes[dim][-1] if len(amplitudes[dim]) > 0 else 0
                norm_amp = recent_amp / np.mean(amplitudes[dim]) if np.mean(amplitudes[dim]) > 0 else 0
                weight = min(1.0, norm_amp)
            else:
                weight = 0.5
                
            phase_signals.append(phase_signal * weight)
            
        # Average phase signals
        phase_component = np.mean(phase_signals) if phase_signals else 0.0
        signal_components.append(phase_component)
        weights.append(0.4)
        
        # 2. Spectral-based signal (30%)
        spectral_signal = 0.0
        if 'spectral_entropy' in spectral_data and len(spectral_data['spectral_entropy']) > 0:
            # Average spectral entropy across dimensions
            avg_entropy = np.mean(spectral_data['spectral_entropy'])
            
            # Lower entropy (more predictable) strengthens existing trends
            # Higher entropy (more random) suggests potential reversals
            
            # Get recent trend
            if features.shape[1] > 5:
                recent_trend = np.mean(np.diff(features[0, -5:]))
                
                if avg_entropy < 0.5:
                    # Low entropy - trend continuation
                    spectral_signal = np.sign(recent_trend) * (1 - avg_entropy)
                else:
                    # High entropy - potential reversal
                    spectral_signal = -np.sign(recent_trend) * (avg_entropy - 0.5) * 2
        
        signal_components.append(spectral_signal)
        weights.append(0.3)
        
        # 3. Projection-based signal (30%)
        projection_signal = 0.0
        if ortho_features.shape[1] > 5:
            # Project recent movement onto dominant principal component
            recent_ortho = ortho_features[:, -5:]
            
            # Calculate direction of movement in first component
            comp1_direction = np.sign(recent_ortho[0, -1] - recent_ortho[0, 0])
            
            # Calculate momentum in this direction
            diffs = np.diff(np.concatenate([[recent_ortho[0, 0]], recent_ortho[0, :]]))
            momentum = np.mean(diffs)
            
            # Signal strength based on momentum
            projection_signal = np.sign(momentum) * min(1.0, abs(momentum) * 10)
        
        signal_components.append(projection_signal)
        weights.append(0.3)
        
        # Combine signals with weights
        weighted_sum = sum(s * w for s, w in zip(signal_components, weights))
        weight_sum = sum(weights)
        
        if weight_sum > 0:
            combined_signal = weighted_sum / weight_sum
        else:
            combined_signal = 0.0
            
        # Ensure signal is in [-1, 1] range
        return np.clip(combined_signal, -1.0, 1.0)
    
    def fit(self, historical_df: pd.DataFrame) -> None:
        """Process historical data and calculate indicators
        
        Parameters
        ----------
        historical_df : pandas.DataFrame
            Historical price data with at minimum a 'close' column
        """
        min_required_bars = max(self.analytic_window, self.orthogonalization_period)
        if len(historical_df) < min_required_bars:
            self.is_fitted = False
            return
            
        try:
            # Make a copy of the dataframe
            df = historical_df.copy()
            
            # 1. Create feature space
            features = self._create_feature_space(df)
            
            # 2. Orthogonalize feature space
            ortho_features = self._orthogonalize_features(features)
            
            # 3. Apply Hilbert transform to each dimension
            phases = []
            amplitudes = []
            for dim in range(ortho_features.shape[0]):
                _, amp, phase = self._apply_hilbert_transform(ortho_features[dim, :])
                phases.append(phase)
                amplitudes.append(amp)
                
            # 4. Perform spectral analysis
            spectral_data = self._spectral_analysis(ortho_features)
            
            # Store components for analysis
            self.space_components = {
                'features': features,
                'ortho_features': ortho_features,
                'phases': phases,
                'amplitudes': amplitudes,
                'spectral_data': spectral_data
            }
            
            # 5. Generate signal
            self.latest_signal = self._generate_signal_from_hilbert_space(
                features, ortho_features, phases, amplitudes, spectral_data
            )
            
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error in Hilbert Agent fit: {e}")
            self.is_fitted = False
    
    def predict(self, *, current_price: float, historical_df: pd.DataFrame) -> float:
        """
        Generate trading signal based on Hilbert's mathematical principles
        
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
        return "Hilbert Agent" 