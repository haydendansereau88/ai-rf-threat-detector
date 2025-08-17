"""
Main Application Launcher
"""
import sys
import os

# Add both possible paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Now import and run
from realtime_processor import main

if __name__ == "__main__":
    print("\nLaunching RF Threat Detection System...")
    print("-" * 40)
    try:
        main()
    except KeyboardInterrupt:
        print("\nSystem shutdown by user")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying alternative launch...")
        # Try running directly
        os.system("python src/realtime_processor.py")