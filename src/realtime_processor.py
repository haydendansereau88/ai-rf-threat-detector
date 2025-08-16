"""
Real-Time RF Threat Detection System
Complete implementation with processor and GUI
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import threading
import queue
import time
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signal_processor import RFSignalGenerator, SpectrumAnalyzer
from threat_classifier import RFFeatureExtractor, ThreatClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')


class RealTimeProcessor:
    """Handles real-time signal processing and classification"""
    
    def __init__(self, buffer_size=4096, sample_rate=1e6):
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.signal_queue = queue.Queue(maxsize=10)
        self.results_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # Load trained models
        self.load_models()
        
        # Initialize components
        self.signal_gen = RFSignalGenerator(sample_rate)
        self.analyzer = SpectrumAnalyzer(fft_size=1024)
        self.feature_extractor = RFFeatureExtractor()
        
    def load_models(self):
        """Load pre-trained models"""
        try:
            # Load Random Forest classifier
            model_path = os.path.join('models', 'trained', 'rf_classifier_latest.pkl')
            if not os.path.exists(model_path):
                # Try alternate path
                model_path = os.path.join('..', 'models', 'trained', 'rf_classifier_latest.pkl')
            
            self.classifier = ThreatClassifier()
            self.classifier.load_model(model_path)
            
            # Load Isolation Forest
            iso_path = os.path.join('models', 'trained', 'isolation_forest_latest.pkl')
            if not os.path.exists(iso_path):
                iso_path = os.path.join('..', 'models', 'trained', 'isolation_forest_latest.pkl')
            
            self.anomaly_detector = joblib.load(iso_path)
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Warning: Could not load models: {e}")
            print("Running in simulation mode...")
            self.classifier = None
            self.anomaly_detector = None
    
    def acquisition_thread(self):
        """Simulate real-time RF signal acquisition"""
        while self.running:
            try:
                # Generate realistic RF environment
                env = self.signal_gen.create_mixed_environment(
                    duration=self.buffer_size/self.sample_rate
                )
                
                # Add to queue
                self.signal_queue.put(env, timeout=0.1)
                
                # Simulate real-time delay
                time.sleep(0.1)
                
            except queue.Full:
                continue
            except Exception as e:
                print(f"Acquisition error: {e}")
    
    def processing_thread(self):
        """Process signals and classify threats"""
        while self.running:
            try:
                # Get signal from queue
                env = self.signal_queue.get(timeout=0.1)
                
                # Extract features
                features = self.feature_extractor.extract_all_features(
                    env['signal'], env['sample_rate']
                )
                
                # Classify
                if self.classifier and self.classifier.is_trained:
                    result = self.classifier.predict_threat(features)
                    
                    # Check for anomaly
                    features_array = np.array(list(features.values())).reshape(1, -1)
                    features_scaled = self.classifier.scaler.transform(features_array)
                    is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
                else:
                    # Simulation mode
                    result = {
                        'prediction': 2 if env['has_threat'] else 0,
                        'class_name': 'Threat' if env['has_threat'] else 'Friendly',
                        'confidence': np.random.uniform(0.7, 0.95),
                        'is_anomaly': False
                    }
                    is_anomaly = False
                
                # Compute spectrum for display
                freqs, spectrum = self.analyzer.compute_spectrum(
                    env['signal'], env['sample_rate']
                )
                
                # Package results
                output = {
                    'timestamp': time.time(),
                    'classification': result,
                    'is_anomaly': is_anomaly,
                    'spectrum': spectrum[:512],  # Limit size
                    'frequencies': freqs[:512],
                    'signal': env['signal'][:1000],  # Sample for display
                    'actual_threat': env['has_threat'],
                    'components': env['components']
                }
                
                # Send to GUI
                self.results_queue.put(output, timeout=0.1)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def start(self):
        """Start real-time processing"""
        self.running = True
        
        # Start threads
        self.acq_thread = threading.Thread(target=self.acquisition_thread, daemon=True)
        self.proc_thread = threading.Thread(target=self.processing_thread, daemon=True)
        
        self.acq_thread.start()
        self.proc_thread.start()
        
        print("Real-time processing started!")
    
    def stop(self):
        """Stop processing"""
        self.running = False
        time.sleep(0.5)  # Allow threads to finish
        print("Real-time processing stopped!")


class ThreatDetectionGUI:
    """Professional military-style GUI for RF threat detection"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI-Powered RF Threat Detection System")
        self.root.geometry("1400x900")
        
        # Dark theme colors
        self.bg_color = "#0a0e1a"
        self.fg_color = "#00ff41"  # Matrix green
        self.threat_color = "#ff0000"
        self.warning_color = "#ffaa00"
        
        # Configure style
        self.setup_style()
        
        # Initialize processor
        self.processor = RealTimeProcessor()
        
        # Create GUI components
        self.create_widgets()
        
        # Start animation
        self.animate_display()
        
        # Start processing
        self.processor.start()
        
    def setup_style(self):
        """Configure dark military theme"""
        self.root.configure(bg=self.bg_color)
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Dark.TFrame', background=self.bg_color)
        style.configure('Dark.TLabel', background=self.bg_color, foreground=self.fg_color)
        style.configure('Dark.TButton', background="#1a1a2e", foreground=self.fg_color)
        
    def create_widgets(self):
        """Create all GUI components"""
        
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top section - Status and controls
        self.create_status_panel(main_frame)
        
        # Middle section - Displays
        display_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left side - Spectrum display
        self.create_spectrum_display(display_frame)
        
        # Right side - Threat status and log
        self.create_threat_panel(display_frame)
        
        # Bottom section - Control panel
        self.create_control_panel(main_frame)
        
    def create_status_panel(self, parent):
        """Create top status panel"""
        status_frame = ttk.Frame(parent, style='Dark.TFrame', height=100)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title = tk.Label(
            status_frame,
            text="RF THREAT DETECTION SYSTEM",
            font=('Consolas', 24, 'bold'),
            bg=self.bg_color,
            fg=self.fg_color
        )
        title.pack()
        
        # Status indicators
        indicator_frame = ttk.Frame(status_frame, style='Dark.TFrame')
        indicator_frame.pack(pady=10)
        
        # System status
        self.status_label = tk.Label(
            indicator_frame,
            text="● SYSTEM: ONLINE",
            font=('Consolas', 14),
            bg=self.bg_color,
            fg=self.fg_color
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Threat level
        self.threat_label = tk.Label(
            indicator_frame,
            text="● THREAT LEVEL: CLEAR",
            font=('Consolas', 14),
            bg=self.bg_color,
            fg=self.fg_color
        )
        self.threat_label.pack(side=tk.LEFT, padx=20)
        
        # Anomaly indicator
        self.anomaly_label = tk.Label(
            indicator_frame,
            text="● ANOMALY: NONE",
            font=('Consolas', 14),
            bg=self.bg_color,
            fg=self.fg_color
        )
        self.anomaly_label.pack(side=tk.LEFT, padx=20)
        
    def create_spectrum_display(self, parent):
        """Create real-time spectrum display"""
        spectrum_frame = ttk.Frame(parent, style='Dark.TFrame')
        spectrum_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(8, 6), facecolor=self.bg_color
        )
        
        # Configure axes
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor(self.bg_color)
            ax.tick_params(colors=self.fg_color)
            ax.spines['bottom'].set_color(self.fg_color)
            ax.spines['left'].set_color(self.fg_color)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, color=self.fg_color)
        
        self.ax1.set_title('Time Domain Signal', color=self.fg_color)
        self.ax1.set_xlabel('Time (ms)', color=self.fg_color)
        self.ax1.set_ylabel('Amplitude', color=self.fg_color)
        
        self.ax2.set_title('Frequency Spectrum', color=self.fg_color)
        self.ax2.set_xlabel('Frequency (kHz)', color=self.fg_color)
        self.ax2.set_ylabel('Power (dB)', color=self.fg_color)
        
        # Initialize plots
        self.time_line, = self.ax1.plot([], [], color=self.fg_color, linewidth=0.5)
        self.spectrum_line, = self.ax2.plot([], [], color=self.fg_color, linewidth=1)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, spectrum_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        plt.tight_layout()
        
    def create_threat_panel(self, parent):
        """Create threat status and detection log"""
        threat_frame = ttk.Frame(parent, style='Dark.TFrame')
        threat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Threat indicator
        self.threat_indicator = tk.Label(
            threat_frame,
            text="NO THREAT",
            font=('Consolas', 28, 'bold'),
            bg=self.bg_color,
            fg=self.fg_color,
            pady=20
        )
        self.threat_indicator.pack()
        
        # Confidence meter
        self.confidence_label = tk.Label(
            threat_frame,
            text="Confidence: ---%",
            font=('Consolas', 14),
            bg=self.bg_color,
            fg=self.fg_color
        )
        self.confidence_label.pack()
        
        # Classification details
        self.class_frame = ttk.Frame(threat_frame, style='Dark.TFrame')
        self.class_frame.pack(pady=20, padx=20, fill=tk.X)
        
        # Detection log
        log_label = tk.Label(
            threat_frame,
            text="DETECTION LOG",
            font=('Consolas', 12, 'bold'),
            bg=self.bg_color,
            fg=self.fg_color
        )
        log_label.pack(pady=(20, 5))
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(
            threat_frame,
            height=20,
            width=50,
            bg="#0d1117",
            fg=self.fg_color,
            font=('Consolas', 10),
            wrap=tk.WORD
        )
        self.log_text.pack(padx=20, fill=tk.BOTH, expand=True)
        
    def create_control_panel(self, parent):
        """Create bottom control panel"""
        control_frame = ttk.Frame(parent, style='Dark.TFrame')
        control_frame.pack(fill=tk.X, pady=10)
        
        # Sensitivity control
        tk.Label(
            control_frame,
            text="Detection Sensitivity:",
            font=('Consolas', 11),
            bg=self.bg_color,
            fg=self.fg_color
        ).pack(side=tk.LEFT, padx=10)
        
        self.sensitivity = tk.Scale(
            control_frame,
            from_=0.5, to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            bg=self.bg_color,
            fg=self.fg_color,
            highlightthickness=0,
            length=200
        )
        self.sensitivity.set(0.75)
        self.sensitivity.pack(side=tk.LEFT)
        
        # Buttons
        self.pause_btn = tk.Button(
            control_frame,
            text="PAUSE",
            font=('Consolas', 11, 'bold'),
            bg="#1a1a2e",
            fg=self.fg_color,
            command=self.toggle_pause,
            padx=20
        )
        self.pause_btn.pack(side=tk.RIGHT, padx=10)
        
        self.clear_btn = tk.Button(
            control_frame,
            text="CLEAR LOG",
            font=('Consolas', 11, 'bold'),
            bg="#1a1a2e",
            fg=self.fg_color,
            command=self.clear_log,
            padx=20
        )
        self.clear_btn.pack(side=tk.RIGHT, padx=10)
        
        self.paused = False
        
    def animate_display(self):
        """Animation loop for real-time updates"""
        def update(frame):
            try:
                if not self.paused:
                    # Get latest result
                    result = self.processor.results_queue.get_nowait()
                    
                    # Update time domain plot
                    time_data = result['signal']
                    time_axis = np.arange(len(time_data)) / self.processor.sample_rate * 1000
                    self.time_line.set_data(time_axis, time_data)
                    self.ax1.set_xlim(0, max(time_axis))
                    self.ax1.set_ylim(np.min(time_data)*1.1, np.max(time_data)*1.1)
                    
                    # Update spectrum plot
                    freqs = result['frequencies'] / 1000  # Convert to kHz
                    spectrum = result['spectrum']
                    self.spectrum_line.set_data(freqs, spectrum)
                    self.ax2.set_xlim(0, 500)  # Show up to 500 kHz
                    self.ax2.set_ylim(np.min(spectrum), np.max(spectrum)+10)
                    
                    # Update threat status
                    self.update_threat_status(result)
                    
                    # Log detection
                    self.log_detection(result)
                    
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Display error: {e}")
            
            return self.time_line, self.spectrum_line
        
        self.ani = FuncAnimation(
            self.fig, update, interval=100, blit=True, cache_frame_data=False
        )
        
    def update_threat_status(self, result):
        """Update threat indicators"""
        classification = result['classification']
        
        # Update threat indicator
        if classification['class_name'] == 'Threat':
            self.threat_indicator.config(text="THREAT DETECTED", fg=self.threat_color)
            self.threat_label.config(text="● THREAT LEVEL: HIGH", fg=self.threat_color)
            
            # Flash effect
            self.root.after(100, lambda: self.threat_indicator.config(bg="#3a0000"))
            self.root.after(200, lambda: self.threat_indicator.config(bg=self.bg_color))
            
        elif classification['class_name'] == 'Military':
            self.threat_indicator.config(text="MILITARY SIGNAL", fg=self.warning_color)
            self.threat_label.config(text="● THREAT LEVEL: MEDIUM", fg=self.warning_color)
        else:
            self.threat_indicator.config(text="NO THREAT", fg=self.fg_color)
            self.threat_label.config(text="● THREAT LEVEL: CLEAR", fg=self.fg_color)
        
        # Update confidence
        confidence = classification['confidence'] * 100
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        
        # Update anomaly indicator
        if result['is_anomaly']:
            self.anomaly_label.config(text="● ANOMALY: DETECTED", fg=self.warning_color)
        else:
            self.anomaly_label.config(text="● ANOMALY: NONE", fg=self.fg_color)
    
    def log_detection(self, result):
        """Add detection to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        classification = result['classification']['class_name']
        confidence = result['classification']['confidence'] * 100
        
        # Determine log color tag
        if classification == 'Threat':
            tag = 'threat'
            self.log_text.tag_config('threat', foreground=self.threat_color)
        elif classification == 'Military':
            tag = 'warning'
            self.log_text.tag_config('warning', foreground=self.warning_color)
        else:
            tag = 'normal'
            self.log_text.tag_config('normal', foreground=self.fg_color)
        
        # Add to log
        log_entry = f"[{timestamp}] {classification} (Conf: {confidence:.1f}%)"
        
        if result['is_anomaly']:
            log_entry += " [ANOMALY]"
        
        # Add actual components for debugging
        if 'components' in result:
            log_entry += f" | Signals: {', '.join(result['components'])}"
        
        log_entry += "\n"
        
        self.log_text.insert(tk.END, log_entry, tag)
        self.log_text.see(tk.END)  # Auto-scroll
        
        # Limit log size
        if int(self.log_text.index('end-1c').split('.')[0]) > 100:
            self.log_text.delete('1.0', '2.0')
    
    def toggle_pause(self):
        """Pause/resume processing"""
        self.paused = not self.paused
        self.pause_btn.config(text="RESUME" if self.paused else "PAUSE")
    
    def clear_log(self):
        """Clear detection log"""
        self.log_text.delete('1.0', tk.END)
    
    def run(self):
        """Start the GUI"""
        # Handle closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start main loop
        self.root.mainloop()
    
    def on_closing(self):
        """Clean shutdown"""
        self.processor.stop()
        self.root.quit()
        self.root.destroy()


# Main application entry point
def main():
    """Run the complete RF Threat Detection System"""
    print("="*60)
    print("AI-POWERED RF THREAT DETECTION SYSTEM")
    print("="*60)
    print("Initializing...")
    
    # Create and run GUI
    app = ThreatDetectionGUI()
    
    print("System ready!")
    print("Controls:")
    print("  - Sensitivity slider: Adjust detection threshold")
    print("  - PAUSE button: Pause/resume real-time processing")
    print("  - CLEAR LOG: Clear detection history")
    print("\nMonitoring RF spectrum...")
    
    app.run()


if __name__ == "__main__":
    main()