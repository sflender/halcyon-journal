#!/usr/bin/env python3
"""
Launcher for Halcyon Journal GUI
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from halcyon_gui import main
    main()
except ImportError as e:
    print(f"Error importing GUI: {e}")
    print("Please ensure PySide6 is installed: pip install PySide6")
    sys.exit(1)
except Exception as e:
    print(f"Error launching GUI: {e}")
    sys.exit(1) 