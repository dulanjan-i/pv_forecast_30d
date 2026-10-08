"""
Pytest configuration for MiRACLE tests.

Automatically adds repo root to sys.path so tests can import from src/.
"""
import sys
from pathlib import Path

# Add repo root to Python path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
