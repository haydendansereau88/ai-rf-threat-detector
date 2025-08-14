"""
Training Data Generator Module
Creates labeled datasets for training RF threat detection models
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_processor import RFSignalGenerator, SpectrumAnalyzer
from threat_classifier import RFFeatureExtractor


class TrainingDataGenerator:
    """Generate and augment training data for RF signal classification"""
    
    # Signal categories and their labels
    SIGNAL_CATEGORIES = {
        'friendly_fm': 0,
        'friendly_wifi': 0,
        'friendly_cellular': 0,
        'military_radar': 1,
        'threat_jammer': 2,
        'threat_sweep': 2
    }
    
    def __init__(self, sample_rate: float = 1e6):
        """
        Initialize training data generator
        
        Args:
            sample_rate: Sampling frequency in Hz
        """
        self.fs = sample_rate
        self.signal_gen = RFSignalGenerator(sample_rate)
        self.feature_extractor = RFFeatureExtractor()
        
    def create_labeled_dataset(self, n_samples: int = 1000,
                              signal_duration: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate labeled training dataset
        
        Args:
            n_samples: Total number of samples to generate
            signal_duration: Duration of each signal in seconds
            
        Returns:
            Tuple of (features, labels) arrays
        """
        print(f"Generating {n_samples} training samples...")
        
        # Calculate samples per category
        categories = list(self.SIGNAL_CATEGORIES.keys())
        samples_per_category = n_samples // len(categories)
        
        all_features = []
        all_labels = []
        
        for category in categories:
            print(f"  Generating {category} signals...")
            
            for i in range(samples_per_category):
                # Generate signal based on category
                signal = self._generate_signal_by_category(category, signal_duration)
                
                # Add varying levels of noise
                snr = np.random.uniform(5, 30)  # Random SNR between 5-30 dB
                signal = self.signal_gen.add_noise(signal, snr_db=snr)
                
                # Extract features
                features = self.feature_extractor.extract_all_features(signal, self.fs)
                
                # Convert to array and add to dataset
                feature_array = np.array(list(features.values()))
                all_features.append(feature_array)
                all_labels.append(self.SIGNAL_CATEGORIES[category])
                
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def _generate_signal_by_category(self, category: str, 
                                     duration: float) -> np.ndarray:
        """
        Generate specific signal type
        
        Args:
            category: Signal category name
            duration: Signal duration
            
        Returns:
            Generated signal
        """
        if category == 'friendly_fm':
            return self.signal_gen.generate_friendly_signal(duration, 'fm')
        elif category == 'friendly_wifi':
            return self.signal_gen.generate_friendly_signal(duration, 'wifi')
        elif category == 'friendly_cellular':
            return self.signal_gen.generate_friendly_signal(duration, 'cellular')
        elif category == 'military_radar':
            # Military radar (not necessarily threat)
            signal = self.signal_gen.generate_threat_signal(duration, 'radar')
            # Make it slightly different from threat radar
            signal *= 0.7
            return signal
        elif category == 'threat_jammer':
            return self.signal_gen.generate_threat_signal(duration, 'jammer')
        elif category == 'threat_sweep':
            return self.signal_gen.generate_threat_signal(duration, 'sweep')
        else:
            raise ValueError(f"Unknown category: {category}")
    
    def augment_data(self, signals: np.ndarray, labels: np.ndarray,
                    augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment data with noise and distortion
        
        Args:
            signals: Original signals array
            labels: Original labels array
            augmentation_factor: How many augmented versions per original
            
        Returns:
            Tuple of (augmented_features, augmented_labels)
        """
        print(f"Augmenting data by factor of {augmentation_factor}...")
        
        augmented_features = []
        augmented_labels = []
        
        for i, (signal_features, label) in enumerate(zip(signals, labels)):
            # Add original
            augmented_features.append(signal_features)
            augmented_labels.append(label)
            
            # Create augmented versions
            for _ in range(augmentation_factor - 1):
                # Add random noise to features
                noise_scale = np.random.uniform(0.05, 0.15)
                augmented = signal_features + np.random.randn(len(signal_features)) * noise_scale
                
                # Random scaling
                scale_factor = np.random.uniform(0.8, 1.2)
                augmented *= scale_factor
                
                augmented_features.append(augmented)
                augmented_labels.append(label)
        
        X_aug = np.array(augmented_features)
        y_aug = np.array(augmented_labels)
        
        print(f"Augmented dataset: {X_aug.shape[0]} samples")
        
        return X_aug, y_aug
    
    def create_mixed_scenarios(self, n_scenarios: int = 100) -> List[Dict]:
        """
        Create complex RF environments with multiple signals
        
        Args:
            n_scenarios: Number of scenarios to generate
            
        Returns:
            List of scenario dictionaries
        """
        scenarios = []
        
        for i in range(n_scenarios):
            # Use the signal generator's mixed environment
            env = self.signal_gen.create_mixed_environment(duration=1.0)
            
            # Extract features
            features = self.feature_extractor.extract_all_features(
                env['signal'], env['sample_rate']
            )
            
            scenario = {
                'features': features,
                'has_threat': env['has_threat'],
                'components': env['components'],
                'signal': env['signal']
            }
            
            scenarios.append(scenario)
            
        return scenarios
    
    def save_dataset(self, X: np.ndarray, y: np.ndarray, 
                    filepath: str = 'data/processed/training_data.npz'):
        """
        Save dataset to file
        
        Args:
            X: Features array
            y: Labels array
            filepath: Save location
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, X=X, y=y)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str = 'data/processed/training_data.npz') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from file
        
        Args:
            filepath: File location
            
        Returns:
            Tuple of (features, labels)
        """
        data = np.load(filepath)
        return data['X'], data['y']
    
    def generate_test_signals(self) -> Dict[str, np.ndarray]:
        """
        Generate individual test signals for demonstration
        
        Returns:
            Dictionary of test signals by type
        """
        test_signals = {}
        duration = 1.0
        
        # Generate clean examples of each type
        test_signals['FM Radio'] = self.signal_gen.generate_friendly_signal(duration, 'fm')
        test_signals['WiFi'] = self.signal_gen.generate_friendly_signal(duration, 'wifi')
        test_signals['Cellular'] = self.signal_gen.generate_friendly_signal(duration, 'cellular')
        test_signals['Radar'] = self.signal_gen.generate_threat_signal(duration, 'radar')
        test_signals['Jammer'] = self.signal_gen.generate_threat_signal(duration, 'jammer')
        test_signals['Freq Sweep'] = self.signal_gen.generate_threat_signal(duration, 'sweep')
        
        # Add noise to make more realistic
        for key in test_signals:
            test_signals[key] = self.signal_gen.add_noise(test_signals[key], snr_db=20)
            
        return test_signals


# Test the module
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create generator
    generator = TrainingDataGenerator()
    
    # Generate small dataset for testing
    X, y = generator.create_labeled_dataset(n_samples=60)
    
    # Show class distribution
    print("\nClass distribution:")
    print("  Class 0 (Friendly):", np.sum(y == 0))
    print("  Class 1 (Military):", np.sum(y == 1))
    print("  Class 2 (Threat):", np.sum(y == 2))
    
    # Generate test signals
    test_signals = generator.generate_test_signals()
    
    # Plot examples
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, (name, signal) in enumerate(test_signals.items()):
        # Plot time domain (first 1000 samples)
        axes[idx].plot(signal[:1000], linewidth=0.5)
        axes[idx].set_title(name)
        axes[idx].set_xlabel('Sample')
        axes[idx].set_ylabel('Amplitude')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Example RF Signals', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Save the dataset
    generator.save_dataset(X, y)
    print("\nDataset saved successfully!")