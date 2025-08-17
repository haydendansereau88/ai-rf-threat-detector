"""
View Training Results and Model Performance
Shows confusion matrix, accuracy, and feature importance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
sys.path.append('src')

from threat_classifier import ThreatClassifier, RFFeatureExtractor
from data_generator import TrainingDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


def load_and_evaluate_model():
    """Load trained model and evaluate performance"""
    print("="*70)
    print("RF THREAT DETECTION - MODEL PERFORMANCE REPORT")
    print("="*70)
    
    # Load the trained model
    print("\n1. Loading trained model...")
    classifier = ThreatClassifier()
    try:
        classifier.load_model('../models/trained/rf_classifier_latest.pkl')
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    # Load metadata if available
    try:
        with open('../models/trained/training_metadata_latest.json', 'r') as f:
            metadata = json.load(f)
            print("\n2. Training Metadata:")
            print(f"   - Training samples: {metadata.get('training_samples', 'N/A')}")
            print(f"   - Validation samples: {metadata.get('validation_samples', 'N/A')}")
            print(f"   - Test samples: {metadata.get('test_samples', 'N/A')}")
            print(f"   - Number of features: {metadata.get('n_features', 'N/A')}")
            print(f"   - Training accuracy: {metadata.get('train_accuracy', 0):.3%}")
            print(f"   - Validation accuracy: {metadata.get('val_accuracy', 0):.3%}")
            print(f"   - Test accuracy: {metadata.get('test_accuracy', 0):.3%}")
    except:
        print("   ⚠ Metadata file not found")
    
    # Generate test data for evaluation
    print("\n3. Generating test data for evaluation...")
    generator = TrainingDataGenerator()
    X_test, y_test = generator.create_labeled_dataset(n_samples=300, signal_duration=0.5)
    
    # Make predictions
    print("\n4. Making predictions on test set...")
    X_test_scaled = classifier.scaler.transform(X_test)
    y_pred = classifier.classifier.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n5. Test Set Performance:")
    print(f"   Overall Accuracy: {accuracy:.3%}")
    
    # Classification report
    print("\n6. Detailed Classification Report:")
    print("-"*50)
    class_names = ['Friendly', 'Military', 'Threat']
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
    
    # Confusion Matrix
    print("\n7. Confusion Matrix:")
    print("-"*50)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print text version
    print("                 Predicted")
    print("              Fri   Mil   Thr")
    for i, actual in enumerate(class_names):
        print(f"Actual {actual:8} {cm[i,0]:4} {cm[i,1]:4} {cm[i,2]:4}")
    
    # Feature Importance
    print("\n8. Top 15 Most Important Features:")
    print("-"*50)
    
    feature_importance = classifier.classifier.feature_importances_
    feature_names = classifier.feature_extractor.feature_names
    
    if feature_names:
        # Sort features by importance
        importance_pairs = list(zip(feature_names, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (fname, importance) in enumerate(importance_pairs[:15], 1):
            bar_length = int(importance * 200)  # Scale for display
            bar = '█' * bar_length
            print(f"{i:2}. {fname:25} {importance:.4f} {bar}")
    
    # Create visualizations
    create_visualizations(cm, class_names, feature_names, feature_importance, accuracy)
    
    return cm, accuracy, classifier


def create_visualizations(cm, class_names, feature_names, feature_importance, accuracy):
    """Create and save visualization plots"""
    print("\n9. Creating visualizations...")
    
    # Set style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Confusion Matrix Heatmap
    ax1 = plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    
    # 2. Normalized Confusion Matrix
    ax2 = plt.subplot(2, 2, 2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    
    # 3. Feature Importance Bar Chart
    ax3 = plt.subplot(2, 2, 3)
    if feature_names and len(feature_names) > 0:
        # Get top 10 features
        importance_pairs = list(zip(feature_names, feature_importance))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        top_features = importance_pairs[:10]
        
        names = [f[0] for f in top_features]
        values = [f[1] for f in top_features]
        
        y_pos = np.arange(len(names))
        plt.barh(y_pos, values, color='cyan')
        plt.yticks(y_pos, names)
        plt.xlabel('Importance Score')
        plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
    
    # 4. Per-Class Accuracy
    ax4 = plt.subplot(2, 2, 4)
    per_class_accuracy = np.diag(cm) / cm.sum(axis=1)
    colors = ['green', 'yellow', 'red']
    bars = plt.bar(class_names, per_class_accuracy, color=colors)
    plt.ylim([0, 1.1])
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_accuracy):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.2%}', ha='center', va='bottom')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('RF Threat Detection - Model Performance Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    save_path = 'model_performance_report.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"   ✓ Visualizations saved to: {save_path}")
    
    plt.show()


def test_specific_scenarios():
    """Test model on specific threat scenarios"""
    print("\n10. Testing Specific Scenarios:")
    print("-"*50)
    
    # Load model
    classifier = ThreatClassifier()
    classifier.load_model('models/trained/rf_classifier_latest.pkl')
    
    # Create test scenarios
    generator = TrainingDataGenerator()
    scenarios = [
        ("Clean FM Radio", 'fm', 'friendly'),
        ("WiFi Network", 'wifi', 'friendly'),
        ("Cellular Signal", 'cellular', 'friendly'),
        ("Radar Pulses", 'radar', 'threat'),
        ("Noise Jammer", 'jammer', 'threat'),
        ("Frequency Sweep", 'sweep', 'threat'),
    ]
    
    print(f"{'Scenario':<20} {'Expected':<10} {'Predicted':<10} {'Confidence':<12} {'Result':<10}")
    print("-"*72)
    
    correct = 0
    total = len(scenarios)
    
    for scenario_name, signal_type, expected_category in scenarios:
        # Generate signal
        if expected_category == 'friendly':
            signal = generator.signal_gen.generate_friendly_signal(0.5, signal_type)
        else:
            signal = generator.signal_gen.generate_threat_signal(0.5, signal_type)
        
        # Add noise
        signal = generator.signal_gen.add_noise(signal, snr_db=15)
        
        # Extract features and predict
        features = generator.feature_extractor.extract_all_features(signal, 1e6)
        result = classifier.predict_threat(features)
        
        # Map expected category to class
        expected_class = {'friendly': 'Friendly', 'threat': 'Threat'}[expected_category]
        predicted_class = result['class_name']
        confidence = result['confidence']
        
        # Check if correct
        is_correct = predicted_class == expected_class
        if is_correct:
            correct += 1
            result_str = "✓ PASS"
        else:
            result_str = "✗ FAIL"
        
        print(f"{scenario_name:<20} {expected_class:<10} {predicted_class:<10} {confidence:<12.1%} {result_str:<10}")
    
    print("-"*72)
    print(f"Scenario Test Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")


def check_model_files():
    """Check if all required model files exist"""
    print("\n11. Model Files Check:")
    print("-"*50)
    
    files_to_check = [
    '../models/trained/rf_classifier_latest.pkl',
    '../models/trained/isolation_forest_latest.pkl',
    '../models/trained/training_metadata_latest.json',
    '../data/processed/training_data.npz'
]
    
    all_present = True
    for filepath in files_to_check:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024  # Convert to KB
            print(f"   ✓ {filepath:<45} ({size:.1f} KB)")
        else:
            print(f"   ✗ {filepath:<45} (NOT FOUND)")
            all_present = False
    
    return all_present


if __name__ == "__main__":
    # Check model files
    files_exist = check_model_files()
    
    if not files_exist:
        print("\n⚠ Some model files are missing. Run train_models.py first.")
    else:
        # Load and evaluate model
        cm, accuracy, classifier = load_and_evaluate_model()
        
        # Test specific scenarios
        test_specific_scenarios()
        
        print("\n" + "="*70)
        print("PERFORMANCE REPORT COMPLETE")
        print("="*70)
        print(f"\nFinal Model Accuracy: {accuracy:.1%}")
        
        if accuracy >= 0.85:
            print("✓ Model meets performance requirements (>85%)")
        else:
            print("⚠ Model below target performance (<85%)")
        
        print("\nVisualization saved as 'model_performance_report.png'")
        print("You can now run 'python main.py' to start the detection system")