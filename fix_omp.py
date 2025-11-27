"""
Fix for OMP duplicate library issue on Windows.
Import this at the beginning of your training scripts.
"""
import os

# Set environment variable to allow duplicate OpenMP libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("✓ OMP duplicate library warning suppressed")
