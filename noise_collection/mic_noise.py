"""
Microphone Background Noise Collection
Captures ambient noise and microphone self-noise
"""

import numpy as np
import sounddevice as sd
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicrophoneNoiseCollector:
    """Collects ambient and self-noise from microphone"""
    
    def __init__(
        self, 
        sample_rate: int = 44100,
        device: Optional[int] = None
    ):
        """
        Initialize microphone noise collector
        
        Args:
            sample_rate: Audio sample rate in Hz
            device: Audio device index (None for default)
        """
        self.sample_rate = sample_rate
        self.device = device
        
    def list_devices(self):
        """List available audio devices"""
        print("\n=== Available Audio Devices ===")
        print(sd.query_devices())
    
    def capture_ambient_noise(
        self, 
        duration: float = 1.0,
        channels: int = 1
    ) -> Optional[np.ndarray]:
        """
        Capture ambient noise from microphone
        
        Args:
            duration: Recording duration in seconds
            channels: Number of audio channels (1 for mono, 2 for stereo)
            
        Returns:
            Numpy array of audio samples or None if failed
        """
        try:
            logger.info(f"Recording {duration}s of ambient noise...")
            
            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=channels,
                device=self.device,
                dtype='float32'
            )
            
            # Wait for recording to complete
            sd.wait()
            
            # Flatten if mono or convert stereo to mono
            if channels > 1:
                recording = np.mean(recording, axis=1)
            else:
                recording = recording.flatten()
            
            logger.info(f"Captured {len(recording)} audio samples")
            return recording
            
        except Exception as e:
            logger.error(f"Failed to capture audio: {e}")
            return None
    
    def capture_multiple_samples(
        self, 
        num_samples: int = 5,
        duration: float = 0.5
    ) -> list:
        """
        Capture multiple noise samples
        
        Args:
            num_samples: Number of samples to capture
            duration: Duration of each sample
            
        Returns:
            List of audio sample arrays
        """
        samples = []
        
        for i in range(num_samples):
            sample = self.capture_ambient_noise(duration=duration)
            
            if sample is not None:
                samples.append(sample)
                logger.info(f"Captured sample {i+1}/{num_samples}")
            else:
                logger.warning(f"Failed to capture sample {i+1}")
        
        return samples
    
    def extract_noise_features(self, audio: np.ndarray) -> dict:
        """
        Extract statistical features from audio noise
        
        Args:
            audio: Audio sample array
            
        Returns:
            Dictionary of noise features
        """
        features = {
            'mean': float(np.mean(audio)),
            'std': float(np.std(audio)),
            'variance': float(np.var(audio)),
            'min': float(np.min(audio)),
            'max': float(np.max(audio)),
            'rms': float(np.sqrt(np.mean(audio**2))),
            'zero_crossings': int(np.sum(np.diff(np.sign(audio)) != 0)),
            'peak_to_peak': float(np.ptp(audio))
        }
        
        return features
    
    def get_frequency_spectrum(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency spectrum of audio noise
        
        Args:
            audio: Audio sample array
            
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        # Apply FFT
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        
        # Frequency bins
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        return freqs, magnitude
    
    def compute_spectral_entropy(self, audio: np.ndarray) -> float:
        """
        Compute spectral entropy of audio signal
        
        Args:
            audio: Audio sample array
            
        Returns:
            Spectral entropy value
        """
        # Get magnitude spectrum
        _, magnitude = self.get_frequency_spectrum(audio)
        
        # Normalize to probability distribution
        power = magnitude ** 2
        power_norm = power / np.sum(power)
        
        # Remove zeros to avoid log(0)
        power_norm = power_norm[power_norm > 0]
        
        # Calculate entropy
        entropy = -np.sum(power_norm * np.log2(power_norm))
        
        return float(entropy)
    
    def get_high_frequency_noise(
        self, 
        audio: np.ndarray,
        cutoff_freq: float = 10000
    ) -> np.ndarray:
        """
        Extract high-frequency noise component
        
        Args:
            audio: Audio sample array
            cutoff_freq: High-pass filter cutoff frequency
            
        Returns:
            High-frequency noise array
        """
        # Apply FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # High-pass filter
        fft[freqs < cutoff_freq] = 0
        
        # Inverse FFT
        filtered = np.fft.irfft(fft, len(audio))
        
        return filtered


def main():
    """Test microphone noise collection"""
    collector = MicrophoneNoiseCollector(sample_rate=44100)
    
    print("\n=== Microphone Noise Collection Test ===")
    
    # List available devices
    collector.list_devices()
    
    # Capture single sample
    print("\n=== Capturing Ambient Noise ===")
    print("Keep quiet for best noise isolation...")
    
    audio = collector.capture_ambient_noise(duration=1.0)
    
    if audio is not None:
        print(f"Audio samples: {len(audio)}")
        print(f"Duration: {len(audio)/collector.sample_rate:.2f}s")
        print(f"Sample rate: {collector.sample_rate} Hz")
        
        # Extract features
        print("\n=== Noise Features ===")
        features = collector.extract_noise_features(audio)
        for key, value in features.items():
            print(f"{key}: {value:.6f}")
        
        # Spectral analysis
        print("\n=== Spectral Analysis ===")
        freqs, magnitude = collector.get_frequency_spectrum(audio)
        print(f"Frequency bins: {len(freqs)}")
        print(f"Max magnitude: {magnitude.max():.2f}")
        print(f"Max frequency: {freqs[np.argmax(magnitude)]:.2f} Hz")
        
        entropy = collector.compute_spectral_entropy(audio)
        print(f"Spectral entropy: {entropy:.4f} bits")
        
        # High-frequency noise
        print("\n=== High-Frequency Noise ===")
        hf_noise = collector.get_high_frequency_noise(audio, cutoff_freq=8000)
        print(f"HF noise samples: {len(hf_noise)}")
        print(f"HF noise RMS: {np.sqrt(np.mean(hf_noise**2)):.6f}")
    
    # Capture multiple samples
    print("\n=== Capturing Multiple Samples ===")
    samples = collector.capture_multiple_samples(num_samples=3, duration=0.5)
    print(f"Collected {len(samples)} samples")
    
    # Compare entropy across samples
    print("\n=== Entropy Comparison ===")
    for i, sample in enumerate(samples):
        ent = collector.compute_spectral_entropy(sample)
        print(f"Sample {i+1} entropy: {ent:.4f} bits")


if __name__ == "__main__":
    main()
