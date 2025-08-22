# 🛡️ RF Threat Detection System - Prototype

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

## 🎯 Overview

A **proof-of-concept** machine learning system for RF (Radio Frequency) signal classification, demonstrating the feasibility of automated threat detection in the electromagnetic spectrum. This prototype explores the intersection of signal processing, machine learning, and defense applications.

## 📌 Project Scope & Intent

This is a **foundational prototype** developed to:
- Explore RF signal processing techniques and their application to threat detection
- Demonstrate machine learning classification of electromagnetic signals
- Build a real-time processing pipeline with visualization
- Establish a modular architecture for future development with real RF hardware

**Current Implementation:**
- Trained and tested on **synthetic RF signal data**
- Achieves ~90% accuracy on simulated test scenarios
- Provides real-time visualization and classification (<500ms latency)
- Serves as a learning platform and foundation for expansion

## ✨ Key Features

- **Real-time Spectrum Analysis**: Live FFT-based spectrum visualization with waterfall display
- **Machine Learning Classification**: Random Forest classifier for signal categorization
- **Anomaly Detection**: Isolation Forest for identifying unusual signal patterns
- **Professional Military-Style Interface**: Dark-themed GUI inspired by operational systems
- **Multi-threaded Processing**: Demonstrates real-time processing architecture
- **Detection Logging**: Tracks classification history and system events

## 🚀 Technical Implementation

### Signal Processing
- Fast Fourier Transform (FFT) for frequency domain analysis
- Spectrogram generation with Hann windowing
- Feature extraction from time and frequency domains
- Synthetic signal generation (FM, WiFi, Radar, Jammers)

### Machine Learning
- Random Forest classifier for multi-class classification
- Isolation Forest for anomaly detection
- 25+ engineered features per signal
- Train/validation/test split methodology

### System Architecture
- Modular design with separate processing components
- Multi-threaded architecture using Python threading
- Queue-based communication between acquisition and processing
- Tkinter-based GUI with matplotlib integration

## 📋 Requirements

- Python 3.8 or higher
- Windows/Linux/MacOS
- 4GB RAM minimum
- Dependencies listed in requirements.txt

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai_rf_threat_detector.git
cd ai_rf_threat_detector
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Quick Demo
```bash
# Run the complete system (models must be trained first)
python main.py
```

### Training Models
```bash
# Generate synthetic data and train classifiers
python train_models.py
```

### View Performance Metrics
```bash
# Display confusion matrix and accuracy metrics
python view_training_results.py
```

## 📊 Signal Categories (Simulated)

| Category | Description | Simulated Examples |
|----------|-------------|-------------------|
| **Friendly** | Civilian communications | FM Radio, WiFi, Cellular |
| **Military** | Military-style signals | Pulsed radar patterns |
| **Threat** | Potentially hostile signals | Noise jammers, Frequency sweeps |

## 🏗️ Project Structure

```
ai_rf_threat_detector/
├── src/
│   ├── signal_processor.py      # Synthetic signal generation & FFT
│   ├── threat_classifier.py     # ML models (Random Forest, Isolation Forest)
│   ├── realtime_processor.py    # Threading and real-time simulation
│   ├── data_generator.py        # Synthetic training data creation
│   └── gui.py                   # Tkinter GUI implementation
├── models/
│   └── trained/                 # Saved sklearn models
├── data/
│   └── processed/               # Synthetic training datasets
├── train_models.py              # Model training script
└── main.py                      # Application entry point
```

## ⚠️ Current Limitations

- **Synthetic Data Only**: Not tested on real RF signals
- **Simplified Signal Models**: Basic modulation types without real-world effects
- **No Hardware Interface**: No SDR or antenna integration
- **Limited Signal Types**: 6 basic signal categories
- **Ideal Conditions**: No multipath, interference, or hardware impairments

## 🛠️ Possible Future Development Plans

### Phase 1: Hardware Integration
- [ ] RTL-SDR integration for real signal capture
- [ ] IQ sample processing from SDR hardware
- [ ] Real-world signal collection and labeling

### Phase 2: Enhanced Models
- [ ] Deep learning models (CNN for spectrograms)
- [ ] Transfer learning from synthetic to real signals
- [ ] Expanded signal library (50+ modulation types)

### Phase 3: Production Features
- [ ] GPU acceleration for real-time processing
- [ ] Database for signal fingerprinting
- [ ] Network-distributed sensor capability
- [ ] Web-based monitoring interface

## 🤔 Technical Decisions & Rationale

- **Random Forest over Deep Learning**: Chosen for interpretability and lower data requirements
- **Synthetic Data**: Allows controlled testing and doesn't require expensive RF equipment
- **Threading over Multiprocessing**: Simpler implementation for prototype, adequate for demo performance
- **Tkinter GUI**: No external dependencies, sufficient for prototype visualization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Signal processing techniques inspired by GNU Radio and scipy.signal
- ML approach influenced by academic papers on RF fingerprinting
- GUI design inspired by military spectrum analyzers

