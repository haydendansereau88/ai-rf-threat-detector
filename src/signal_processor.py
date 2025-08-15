"""
RF Signal Processing Module
Handles signal generation, spectrum analysis, and feature extraction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple, List, Dict, Optional


class RFSignalGenerator:
    """Generate various RF signals for testing and training"""
    
    def __init__(self, sample_rate: float = 1e6):
        """
        Initialize RF Signal Generator
        
        Args:
            sample_rate: Sampling frequency in Hz (default 1 MHz)
        """
        self.fs = sample_rate
        self.nyquist = sample_rate / 2
        
    def generate_friendly_signal(self, duration: float = 1.0, 
                                signal_type: str = 'fm') -> np.ndarray:
        """
        Generate civilian/friendly communication signals
        
        Args:
            duration: Signal duration in seconds
            signal_type: Type of signal ('fm', 'wifi', 'cellular')
            
        Returns:
            Generated signal array
        """
        t = np.linspace(0, duration, int(self.fs * duration))
        
        if signal_type == 'fm':
            # FM radio signal simulation
            carrier_freq = 100e3  # 100 kHz carrier
            mod_freq = 1e3  # 1 kHz modulation
            mod_index = 5
            
            modulator = np.sin(2 * np.pi * mod_freq * t)
            phase = 2 * np.pi * carrier_freq * t + mod_index * modulator
            signal_out = np.cos(phase)
            
        elif signal_type == 'wifi':
            # Simplified WiFi-like OFDM signal
            n_subcarriers = 64
            frequencies = np.linspace(20e3, 100e3, n_subcarriers)
            signal_out = np.zeros_like(t)
            
            for freq in frequencies:
                signal_out += np.random.randn() * np.sin(2 * np.pi * freq * t)
            signal_out /= np.sqrt(n_subcarriers)
            
        else:  # cellular
            # Simplified cellular signal
            carrier_freq = 200e3
            signal_out = np.sin(2 * np.pi * carrier_freq * t)
            # Add some amplitude modulation
            signal_out *= (1 + 0.3 * np.sin(2 * np.pi * 100 * t))
            
        return signal_out
    
    def generate_threat_signal(self, duration: float = 1.0,
                              threat_type: str = 'jammer') -> np.ndarray:
        """
        Generate military/threat radar and jamming signals
        
        Args:
            duration: Signal duration in seconds
            threat_type: Type of threat ('jammer', 'radar', 'sweep')
            
        Returns:
            Generated threat signal array
        """
        t = np.linspace(0, duration, int(self.fs * duration))
        
        if threat_type == 'jammer':
            # Broadband noise jammer
            signal_out = np.random.randn(len(t)) * 2
            # Add some structure to make it detectable
            signal_out += np.sin(2 * np.pi * 50e3 * t) * 0.5
            
        elif threat_type == 'radar':
            # Pulsed radar signal
            pulse_width = 0.001  # 1ms pulse
            pulse_period = 0.01  # 10ms period
            carrier_freq = 300e3
            
            signal_out = np.zeros_like(t)
            pulse_indices = np.arange(0, len(t), int(pulse_period * self.fs))
            
            for idx in pulse_indices:
                end_idx = min(idx + int(pulse_width * self.fs), len(t))
                signal_out[idx:end_idx] = np.sin(2 * np.pi * carrier_freq * t[idx:end_idx])
                
        else:  # sweep
            # Frequency sweep jammer
            f_start = 50e3
            f_end = 400e3
            chirp_rate = (f_end - f_start) / duration
            frequency = f_start + chirp_rate * t
            signal_out = np.sin(2 * np.pi * frequency * t)
            
        return signal_out
    
    def add_noise(self, signal_in: np.ndarray, snr_db: float = 10) -> np.ndarray:
        """
        Add realistic noise to signals
        
        Args:
            signal_in: Input signal
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Noisy signal
        """
        signal_power = np.mean(signal_in ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(len(signal_in))
        return signal_in + noise
    
    def create_mixed_environment(self, duration: float = 1.0) -> Dict:
        """
        Create complex RF environment with multiple signals
        
        Args:
            duration: Environment duration in seconds
            
        Returns:
            Dictionary containing mixed signal and component info
        """
        # Generate multiple signals
        fm_signal = self.generate_friendly_signal(duration, 'fm') * 0.5
        wifi_signal = self.generate_friendly_signal(duration, 'wifi') * 0.3
        
        # Randomly add threat
        has_threat = np.random.random() > 0.3
        if has_threat:
            threat_type = np.random.choice(['jammer', 'radar', 'sweep'])
            threat_signal = self.generate_threat_signal(duration, threat_type) * 0.7
            mixed = fm_signal + wifi_signal + threat_signal
            components = ['fm', 'wifi', threat_type]
        else:
            mixed = fm_signal + wifi_signal
            components = ['fm', 'wifi']
            
        # Add noise
        mixed = self.add_noise(mixed, snr_db=15)
        
        return {
            'signal': mixed,
            'components': components,
            'has_threat': has_threat,
            'sample_rate': self.fs
        }


class SpectrumAnalyzer:
    """Real-time spectrum analysis and feature extraction"""
    
    def __init__(self, fft_size: int = 1024, overlap: float = 0.5,
                 window: str = 'hann'):
        """
        Initialize Spectrum Analyzer
        
        Args:
            fft_size: FFT size for spectral analysis
            overlap: Overlap ratio between consecutive FFT windows
            window: Window function type
        """
        self.fft_size = fft_size
        self.overlap = overlap
        self.window = window
        self.hop_size = int(fft_size * (1 - overlap))
        
    def compute_spectrogram(self, signal_in: np.ndarray, 
                           fs: float = 1e6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate real-time spectrogram using STFT
        
        Args:
            signal_in: Input signal
            fs: Sampling frequency
            
        Returns:
            Tuple of (frequencies, times, spectrogram)
        """
        f, t, Sxx = signal.spectrogram(
            signal_in, fs=fs,
            window=self.window,
            nperseg=self.fft_size,
            noverlap=int(self.fft_size * self.overlap)
        )
        
        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        return f, t, Sxx_db
    
    def compute_spectrum(self, signal_in: np.ndarray, 
                        fs: float = 1e6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency spectrum of signal
        
        Args:
            signal_in: Input signal
            fs: Sampling frequency
            
        Returns:
            Tuple of (frequencies, spectrum magnitude)
        """
        # Apply window
        window = signal.get_window(self.window, len(signal_in))
        windowed_signal = signal_in * window
        
        # Compute FFT
        spectrum = fft(windowed_signal)
        frequencies = fftfreq(len(signal_in), 1/fs)
        
        # Get positive frequencies only
        pos_freq_idx = frequencies >= 0
        frequencies = frequencies[pos_freq_idx]
        spectrum = np.abs(spectrum[pos_freq_idx])
        
        # Convert to dB
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        
        return frequencies, spectrum_db
    
    def extract_features(self, signal_in: np.ndarray, fs: float = 1e6) -> Dict:
        """
        Extract ML features from RF signal
        
        Args:
            signal_in: Input signal
            fs: Sampling frequency
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Time domain features
        features['mean'] = np.mean(signal_in)
        features['std'] = np.std(signal_in)
        features['max'] = np.max(np.abs(signal_in))
        features['rms'] = np.sqrt(np.mean(signal_in**2))
        
        # Frequency domain features
        freqs, spectrum = self.compute_spectrum(signal_in, fs)
        
        # Peak frequency
        peak_idx = np.argmax(spectrum)
        features['peak_freq'] = freqs[peak_idx]
        features['peak_power'] = spectrum[peak_idx]
        
        # Bandwidth (3dB)
        peak_power = spectrum[peak_idx]
        bandwidth_threshold = peak_power - 3
        bandwidth_indices = np.where(spectrum > bandwidth_threshold)[0]
        if len(bandwidth_indices) > 0:
            features['bandwidth'] = freqs[bandwidth_indices[-1]] - freqs[bandwidth_indices[0]]
        else:
            features['bandwidth'] = 0
            
        # Spectral centroid
        spectrum_linear = 10 ** (spectrum / 20)  # Convert from dB
        features['spectral_centroid'] = np.sum(freqs * spectrum_linear) / np.sum(spectrum_linear)
        
        # Spectral rolloff
        cumsum = np.cumsum(spectrum_linear)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        if len(rolloff_idx) > 0:
            features['spectral_rolloff'] = freqs[rolloff_idx[0]]
        else:
            features['spectral_rolloff'] = freqs[-1]
            
        return features
    
    def detect_signals(self, spectrum: np.ndarray, 
                      noise_floor_percentile: float = 25) -> List[Dict]:
        """
        Identify signal presence above noise floor
        
        Args:
            spectrum: Frequency spectrum in dB
            noise_floor_percentile: Percentile for noise floor estimation
            
        Returns:
            List of detected signals with their properties
        """
        # Estimate noise floor
        noise_floor = np.percentile(spectrum, noise_floor_percentile)
        
        # Detection threshold (10 dB above noise floor)
        threshold = noise_floor + 10
        
        # Find peaks
        peaks, properties = signal.find_peaks(
            spectrum, 
            height=threshold,
            distance=20,  # Minimum distance between peaks
            prominence=6   # Minimum prominence
        )
        
        detected_signals = []
        for i, peak in enumerate(peaks):
            detected_signals.append({
                'frequency_bin': peak,
                'power': spectrum[peak],
                'snr': spectrum[peak] - noise_floor,
                'prominence': properties['prominences'][i]
            })
            
        return detected_signals


# Example usage and testing
if __name__ == "__main__":
    print("Testing RF Signal Processor...")
    print("="*50)
    
    # Create instances
    generator = RFSignalGenerator(sample_rate=1e6)
    analyzer = SpectrumAnalyzer(fft_size=2048, overlap=0.75)
    
    # Generate test environment
    print("\nGenerating mixed RF environment...")
    env = generator.create_mixed_environment(duration=2.0)
    
    print(f"Environment contains: {env['components']}")
    print(f"Has threat: {env['has_threat']}")
    
    # Analyze signal
    print("\nExtracting features...")
    features = analyzer.extract_features(env['signal'], env['sample_rate'])
    print("Extracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    # Compute and plot spectrogram
    print("\nComputing spectrogram...")
    f, t, Sxx = analyzer.compute_spectrogram(env['signal'], env['sample_rate'])
    
    # Create visualization
    print("Creating visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot time domain
    plt.subplot(2, 1, 1)
    time_axis = np.arange(len(env['signal'])) / env['sample_rate']
    plt.plot(time_axis[:1000], env['signal'][:1000], 'b-', linewidth=0.5)
    plt.title('RF Signal (Time Domain)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    plt.pcolormesh(t, f/1000, Sxx, shading='gouraud', cmap='viridis')
    plt.colorbar(label='Power (dB)')
    plt.title('Spectrogram', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (kHz)')
    plt.ylim([0, 500])
    
    plt.suptitle('RF Signal Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    print("\nShowing plot (close window to exit)...")
    plt.show()
    
    print("\nTest complete!")