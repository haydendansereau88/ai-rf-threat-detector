"""
Main Application Launcher
Run this to start the complete RF Threat Detection System
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from realtime_processor import main

if __name__ == "__main__":
    print("\nLaunching RF Threat Detection System...")
    print("-" * 40)
    main()