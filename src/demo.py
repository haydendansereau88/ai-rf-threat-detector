"""
Demo Script - Shows system capabilities
"""
import time
import sys
import os
sys.path.append('src')

from signal_processor import RFSignalGenerator
from threat_classifier import ThreatClassifier, RFFeatureExtractor

def run_demo():
    print("\n" + "="*60)
    print("RF THREAT DETECTION SYSTEM - DEMO")
    print("="*60)
    
    # Load classifier
    print("\n1. Loading trained models...")
    classifier = ThreatClassifier()
    try:
        classifier.load_model('models/trained/rf_classifier_latest.pkl')
        print("   ✓ Models loaded")
    except:
        print("   ✗ Models not found - train first with train_models.py")
        return
    
    # Generate test signals
    print("\n2. Generating test scenarios...")
    generator = RFSignalGenerator()
    extractor = RFFeatureExtractor()
    
    scenarios = [
        ("Normal Communications", False),
        ("Detected Jamming Signal", True),
        ("Unknown Signal Pattern", True)
    ]
    
    for scenario_name, is_threat in scenarios:
        print(f"\n   Scenario: {scenario_name}")
        
        # Generate appropriate signal
        if is_threat:
            signal = generator.generate_threat_signal(duration=0.5, threat_type='jammer')
        else:
            signal = generator.generate_friendly_signal(duration=0.5, signal_type='fm')
        
        # Process
        features = extractor.extract_all_features(signal, 1e6)
        result = classifier.predict_threat(features)
        
        print(f"   Classification: {result['class_name']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Threat Level: {'HIGH' if result['class_name']=='Threat' else 'LOW'}")
        
        time.sleep(1)
    
    print("\n3. Demo complete!")
    print("\nRun 'python main.py' to start the full system")

if __name__ == "__main__":
    run_demo()