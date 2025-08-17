"""
Training Script for RF Threat Detection Models
Generates synthetic data and trains both Random Forest and Isolation Forest
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import joblib
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from signal_processor import RFSignalGenerator, SpectrumAnalyzer
from threat_classifier import RFFeatureExtractor, ThreatClassifier
from data_generator import TrainingDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


def create_training_data(n_samples=1200, visualize=True):
    """
    Generate synthetic RF signal training data
    
    Args:
        n_samples: Total number of samples to generate
        visualize: Whether to show sample signals
    
    Returns:
        X: Feature matrix
        y: Labels (0=Friendly, 1=Military, 2=Threat)
    """
    print("="*60)
    print("GENERATING TRAINING DATA")
    print("="*60)
    
    # Initialize generator
    generator = TrainingDataGenerator(sample_rate=1e6)
    
    # Generate labeled dataset
    print(f"\nGenerating {n_samples} synthetic RF signals...")
    X, y = generator.create_labeled_dataset(n_samples=n_samples, signal_duration=0.5)
    
    # Print class distribution
    print("\nClass Distribution:")
    print("-"*30)
    classes = ['Friendly', 'Military', 'Threat']
    for i in range(3):
        count = np.sum(y == i)
        percentage = count / len(y) * 100
        print(f"  {classes[i]:10s}: {count:4d} samples ({percentage:.1f}%)")
    
    # Visualize sample signals if requested
    if visualize:
        print("\nGenerating visualization samples...")
        visualize_signal_types(generator)
    
    # Data augmentation to increase dataset size
    print("\nApplying data augmentation...")
    X_augmented, y_augmented = generator.augment_data(X, y, augmentation_factor=2)
    print(f"Dataset size after augmentation: {len(X_augmented)} samples")
    
    # Save the dataset
    save_path = os.path.join('data', 'processed', 'training_data.npz')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, X=X_augmented, y=y_augmented)
    print(f"Dataset saved to: {save_path}")
    
    return X_augmented, y_augmented


def visualize_signal_types(generator):
    """
    Visualize different signal types for verification
    """
    print("\nCreating signal visualization...")
    
    # Generate test signals
    test_signals = generator.generate_test_signals()
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # Colors for different signal types
    colors = ['green', 'green', 'green', 'orange', 'red', 'red']
    
    for idx, (name, signal) in enumerate(test_signals.items()):
        ax = axes[idx]
        
        # Plot time domain (first 1000 samples)
        time = np.arange(1000) / 1e6 * 1000  # Convert to milliseconds
        ax.plot(time, signal[:1000], linewidth=0.5, color=colors[idx], alpha=0.7)
        ax.set_title(f'{name}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (ms)', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        
        # Add classification label
        if idx < 3:
            ax.text(0.02, 0.98, 'FRIENDLY', transform=ax.transAxes,
                   color='green', fontsize=8, fontweight='bold',
                   verticalalignment='top')
        else:
            ax.text(0.02, 0.98, 'THREAT', transform=ax.transAxes,
                   color='red', fontsize=8, fontweight='bold',
                   verticalalignment='top')
    
    plt.suptitle('RF Signal Types - Training Data Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join('docs', 'images', 'signal_types.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    plt.show()


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest classifier for threat detection
    
    Returns:
        classifier: Trained ThreatClassifier object
        metrics: Performance metrics dictionary
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    # Initialize classifier
    classifier = ThreatClassifier(classifier_type='random_forest')
    
    # Train the model
    print("\nTraining Random Forest with 100 trees...")
    metrics = classifier.train_classifier(X_train, y_train, validation_split=0.2)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_pred = classifier.classifier.predict(classifier.scaler.transform(X_val))
    
    # Print detailed classification report
    print("\nClassification Report:")
    print("-"*50)
    class_names = ['Friendly', 'Military', 'Threat']
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("-"*30)
    cm = confusion_matrix(y_val, y_pred)
    print("             Predicted")
    print("           Fri  Mil  Thr")
    for i, actual in enumerate(['Friendly', 'Military', 'Threat']):
        print(f"{actual:8s}  {cm[i,0]:3d}  {cm[i,1]:3d}  {cm[i,2]:3d}")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    print("-"*40)
    feature_importance = classifier.classifier.feature_importances_
    feature_names = classifier.feature_extractor.feature_names
    
    if feature_names:
        importance_pairs = list(zip(feature_names, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (fname, importance) in enumerate(importance_pairs[:10]):
            print(f"{i+1:2d}. {fname:25s}: {importance:.4f}")
    
    return classifier, metrics


def train_isolation_forest(X_normal):
    """
    Train Isolation Forest for anomaly detection
    
    Args:
        X_normal: Features from normal (non-threat) signals
    
    Returns:
        anomaly_detector: Trained Isolation Forest
    """
    print("\n" + "="*60)
    print("TRAINING ISOLATION FOREST ANOMALY DETECTOR")
    print("="*60)
    
    print(f"\nTraining on {len(X_normal)} normal signal samples...")
    
    # Initialize Isolation Forest
    anomaly_detector = IsolationForest(
        n_estimators=100,
        contamination=0.1,  # Expect 10% anomalies
        random_state=42
    )
    
    # Train the model
    anomaly_detector.fit(X_normal)
    
    # Test anomaly detection
    print("\nTesting anomaly detection...")
    
    # Predict on training data (should be mostly normal)
    predictions = anomaly_detector.predict(X_normal[:100])
    n_anomalies = np.sum(predictions == -1)
    print(f"Detected {n_anomalies}/100 as anomalies in normal data (expect ~10)")
    
    return anomaly_detector


def save_models(rf_classifier, anomaly_detector, metadata):
    """
    Save trained models and metadata
    """
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    # Create models directory
    os.makedirs('models/trained', exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Random Forest classifier
    rf_path = f'models/trained/rf_classifier_{timestamp}.pkl'
    rf_classifier.save_model(rf_path)
    
    # Save Isolation Forest
    iso_path = f'models/trained/isolation_forest_{timestamp}.pkl'
    joblib.dump(anomaly_detector, iso_path)
    print(f"Isolation Forest saved to: {iso_path}")
    
    # Save metadata
    metadata['timestamp'] = timestamp
    metadata['rf_model_path'] = rf_path
    metadata['isolation_forest_path'] = iso_path
    
    meta_path = f'models/trained/training_metadata_{timestamp}.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {meta_path}")
    
    # Create a "latest" symlink/copy for easy access
    latest_rf = 'models/trained/rf_classifier_latest.pkl'
    latest_iso = 'models/trained/isolation_forest_latest.pkl'
    latest_meta = 'models/trained/training_metadata_latest.json'
    
    # Copy files (Windows doesn't support symlinks easily)
    import shutil
    shutil.copy2(rf_path, latest_rf)
    shutil.copy2(iso_path, latest_iso)
    shutil.copy2(meta_path, latest_meta)
    
    print(f"\nLatest models linked for easy access:")
    print(f"  - {latest_rf}")
    print(f"  - {latest_iso}")
    print(f"  - {latest_meta}")
    
    return {
        'rf_path': rf_path,
        'iso_path': iso_path,
        'meta_path': meta_path
    }


def test_on_new_signals(rf_classifier, anomaly_detector):
    """
    Test models on completely new signals
    """
    print("\n" + "="*60)
    print("TESTING ON NEW SIGNALS")
    print("="*60)
    
    # Generate new test signals
    generator = TrainingDataGenerator()
    
    print("\nGenerating 5 random test scenarios...")
    for i in range(5):
        # Create a mixed environment
        env = generator.signal_gen.create_mixed_environment(duration=0.5)
        
        # Extract features
        features = generator.feature_extractor.extract_all_features(
            env['signal'], env['sample_rate']
        )
        
        # Get prediction
        result = rf_classifier.predict_threat(features)
        
        # Check for anomaly
        features_array = np.array(list(features.values())).reshape(1, -1)
        features_scaled = rf_classifier.scaler.transform(features_array)
        is_anomaly = anomaly_detector.predict(features_scaled)[0] == -1
        
        print(f"\nTest {i+1}:")
        print(f"  Actual components: {env['components']}")
        print(f"  Has threat: {env['has_threat']}")
        print(f"  Prediction: {result['class_name']} (confidence: {result['confidence']:.2%})")
        print(f"  Anomaly detected: {is_anomaly}")
        
        # Check if prediction matches reality
        predicted_threat = result['class_name'] == 'Threat'
        if predicted_threat == env['has_threat']:
            print(f"  ✓ Correct prediction!")
        else:
            print(f"  ✗ Incorrect prediction")


def main():
    """
    Main training pipeline
    """
    print("\n" + "="*70)
    print(" RF THREAT DETECTION SYSTEM - MODEL TRAINING PIPELINE")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Generate training data
    X, y = create_training_data(n_samples=1200, visualize=True)
    
    # Step 2: Split data
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 3: Train Random Forest
    rf_classifier, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    
    # Step 4: Train Isolation Forest (on non-threat data)
    X_normal = X_train[y_train != 2]  # Everything except threats
    anomaly_detector = train_isolation_forest(X_normal)
    
    # Step 5: Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Scale test features
    X_test_scaled = rf_classifier.scaler.transform(X_test)
    
    # Predictions
    y_pred = rf_classifier.classifier.predict(X_test_scaled)
    test_accuracy = np.mean(y_pred == y_test)
    
    print(f"\nTest Set Accuracy: {test_accuracy:.3%}")
    
    # Detailed metrics
    print("\nTest Set Classification Report:")
    print("-"*50)
    print(classification_report(y_test, y_pred, 
                              target_names=['Friendly', 'Military', 'Threat']))
    
    # Step 6: Save models
    metadata = {
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'features': rf_classifier.feature_extractor.feature_names,
        'n_features': X.shape[1],
        'train_accuracy': rf_metrics['train_accuracy'],
        'val_accuracy': rf_metrics['val_accuracy'],
        'test_accuracy': float(test_accuracy),
        'classes': ['Friendly', 'Military', 'Threat']
    }
    
    saved_paths = save_models(rf_classifier, anomaly_detector, metadata)
    
    # Step 7: Test on new signals
    test_on_new_signals(rf_classifier, anomaly_detector)
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModels are ready for deployment!")
    print(f"Test accuracy achieved: {test_accuracy:.1%}")
    
    if test_accuracy >= 0.85:
        print("✓ Meeting performance requirement (>85%)")
    else:
        print("✗ Below performance requirement (>85%) - may need tuning")
    
    return rf_classifier, anomaly_detector, saved_paths


if __name__ == "__main__":
    # Run the training pipeline
    rf_classifier, anomaly_detector, saved_paths = main()
    
    # print("\n" + "="*60)
    # print("Next steps:")
    # print("  1. Review the visualizations in docs/images/")
    # print("  2. Check the saved models in models/trained/")
    # print("  3. Proceed to build the real-time processor")
    # print("  4. Then create the GUI")
    # print("="*60)