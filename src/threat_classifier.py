"""
Threat Classification Module
Machine learning models for RF signal classification and anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import joblib
import os

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class RFFeatureExtractor:
    """Extract features from RF signals for machine learning"""
    
    def __init__(self):
        """Initialize feature extractor"""
        self.feature_names = []
        
    def extract_spectral_features(self, signal: np.ndarray, 
                                 fs: float = 1e6) -> Dict[str, float]:
        """
        Extract frequency domain features
        
        Args:
            signal: Input signal
            fs: Sampling frequency
            
        Returns:
            Dictionary of spectral features
        """
        # Compute FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/fs)
        
        # Get magnitude spectrum (positive frequencies only)
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        magnitude = np.abs(fft[pos_mask])
        
        # Normalize magnitude
        if np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)
        
        features = {}
        
        # Spectral centroid
        if np.sum(magnitude) > 0:
            features['spectral_centroid'] = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            features['spectral_centroid'] = 0
            
        # Spectral bandwidth
        if np.sum(magnitude) > 0:
            centroid = features['spectral_centroid']
            features['spectral_bandwidth'] = np.sqrt(
                np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude)
            )
        else:
            features['spectral_bandwidth'] = 0
            
        # Spectral rolloff (85% of energy)
        cumsum = np.cumsum(magnitude)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            if len(rolloff_idx) > 0:
                features['spectral_rolloff'] = freqs[rolloff_idx[0]]
            else:
                features['spectral_rolloff'] = freqs[-1]
        else:
            features['spectral_rolloff'] = 0
            
        # Spectral flux (rate of change)
        magnitude_diff = np.diff(magnitude)
        features['spectral_flux'] = np.sum(magnitude_diff ** 2)
        
        # Peak frequency
        peak_idx = np.argmax(magnitude)
        features['peak_frequency'] = freqs[peak_idx]
        features['peak_magnitude'] = magnitude[peak_idx]
        
        # Number of peaks
        peaks, _ = find_peaks(magnitude, height=0.3, distance=20)
        features['num_peaks'] = len(peaks)
        
        return features
    
    def extract_temporal_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract time domain features
        
        Args:
            signal: Input signal
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['max'] = np.max(np.abs(signal))
        features['min'] = np.min(np.abs(signal))
        features['rms'] = np.sqrt(np.mean(signal ** 2))
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(signal)
        
        # Peak-to-average ratio
        if features['rms'] > 0:
            features['peak_to_avg'] = features['max'] / features['rms']
        else:
            features['peak_to_avg'] = 0
            
        # Crest factor
        if features['rms'] > 0:
            features['crest_factor'] = features['max'] / features['rms']
        else:
            features['crest_factor'] = 0
            
        return features
    
    def extract_statistical_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features
        
        Args:
            signal: Input signal
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        
        # Higher order statistics
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)
        
        # Energy
        features['energy'] = np.sum(signal ** 2)
        
        # Entropy
        hist, _ = np.histogram(signal, bins=50)
        hist = hist / np.sum(hist)  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        features['entropy'] = -np.sum(hist * np.log2(hist))
        
        # Percentiles
        features['percentile_25'] = np.percentile(np.abs(signal), 25)
        features['percentile_75'] = np.percentile(np.abs(signal), 75)
        features['iqr'] = features['percentile_75'] - features['percentile_25']
        
        return features
    
    def extract_all_features(self, signal: np.ndarray, 
                            fs: float = 1e6) -> Dict[str, float]:
        """
        Extract all features from signal
        
        Args:
            signal: Input signal
            fs: Sampling frequency
            
        Returns:
            Dictionary of all features
        """
        all_features = {}
        
        # Extract all feature types
        spectral = self.extract_spectral_features(signal, fs)
        temporal = self.extract_temporal_features(signal)
        statistical = self.extract_statistical_features(signal)
        
        # Combine all features
        all_features.update(spectral)
        all_features.update(temporal)
        all_features.update(statistical)
        
        # Store feature names
        self.feature_names = list(all_features.keys())
        
        return all_features


class ThreatClassifier:
    """ML classifier for RF threat detection"""
    
    def __init__(self, classifier_type: str = 'random_forest'):
        """
        Initialize threat classifier
        
        Args:
            classifier_type: Type of classifier ('random_forest', 'svm')
        """
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()
        self.feature_extractor = RFFeatureExtractor()
        
        # Initialize classifier
        if classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif classifier_type == 'svm':
            self.classifier = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
            
        # Anomaly detector for unknown signals
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        self.is_trained = False
        self.classes_ = ['Friendly', 'Military', 'Threat']
        
    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                        validation_split: float = 0.2) -> Dict:
        """
        Train the threat classification model
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_split: Fraction for validation
            
        Returns:
            Training metrics dictionary
        """
        print(f"Training {self.classifier_type} classifier...")
        
        # Split data for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_split)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train classifier
        self.classifier.fit(X_train_scaled, y_train_split)
        
        # Train anomaly detector on normal data
        normal_data = X_train_scaled[y_train_split != 2]  # Non-threat data
        self.anomaly_detector.fit(normal_data)
        
        # Evaluate
        train_score = self.classifier.score(X_train_scaled, y_train_split)
        val_score = self.classifier.score(X_val_scaled, y_val)
        
        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X_train_scaled, 
                                   y_train_split, cv=5)
        
        # Predictions for detailed metrics
        y_pred = self.classifier.predict(X_val_scaled)
        
        metrics = {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_val, y_pred),
            'confusion_matrix': confusion_matrix(y_val, y_pred)
        }
        
        self.is_trained = True
        
        print(f"Training complete!")
        print(f"  Train accuracy: {train_score:.3f}")
        print(f"  Validation accuracy: {val_score:.3f}")
        print(f"  Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        return metrics
    
    def predict_threat(self, features: Union[np.ndarray, Dict]) -> Dict:
        """
        Classify signal as threat/friendly with confidence
        
        Args:
            features: Feature vector or dictionary
            
        Returns:
            Prediction results with confidence
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained yet!")
            
        # Convert dict to array if needed
        if isinstance(features, dict):
            features = np.array(list(features.values())).reshape(1, -1)
        elif len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get prediction and probabilities
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        # Check for anomaly
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
        
        result = {
            'prediction': int(prediction),
            'class_name': self.classes_[prediction],
            'confidence': float(max(probabilities)),
            'probabilities': {
                'Friendly': float(probabilities[0]),
                'Military': float(probabilities[1]) if len(probabilities) > 1 else 0,
                'Threat': float(probabilities[2]) if len(probabilities) > 2 else 0
            },
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(anomaly_score)
        }
        
        return result
    
    def detect_anomaly(self, features: np.ndarray) -> bool:
        """
        Detect unknown/suspicious signal patterns
        
        Args:
            features: Feature vector
            
        Returns:
            True if anomaly detected
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained yet!")
            
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.anomaly_detector.predict(features_scaled)
        
        return prediction[0] == -1
    
    def save_model(self, filepath: str = 'models/trained/threat_classifier.pkl'):
        """
        Save trained model to file
        
        Args:
            filepath: Save location
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model!")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'anomaly_detector': self.anomaly_detector,
            'classifier_type': self.classifier_type,
            'feature_names': self.feature_extractor.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'models/trained/threat_classifier.pkl'):
        """
        Load trained model from file
        
        Args:
            filepath: Model location
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        model_data = joblib.load(filepath)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.anomaly_detector = model_data['anomaly_detector']
        self.classifier_type = model_data['classifier_type']
        self.feature_extractor.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


# Test the module
if __name__ == "__main__":
    from data_generator import TrainingDataGenerator
    import matplotlib.pyplot as plt
    
    # Generate training data
    print("Generating training data...")
    generator = TrainingDataGenerator()
    X, y = generator.create_labeled_dataset(n_samples=600)
    
    # Train classifier
    classifier = ThreatClassifier(classifier_type='random_forest')
    metrics = classifier.train_classifier(X, y)
    
    # Print detailed results
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save the model
    classifier.save_model()
    
    # Test on a new signal
    print("\nTesting on new signal...")
    test_env = generator.signal_gen.create_mixed_environment()
    test_features = generator.feature_extractor.extract_all_features(
        test_env['signal'], test_env['sample_rate']
    )
    
    result = classifier.predict_threat(test_features)
    print(f"\nPrediction: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Is Anomaly: {result['is_anomaly']}")
    print(f"Actual has threat: {test_env['has_threat']}")