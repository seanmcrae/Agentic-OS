# conftest.py for Dzogchen tests — no legacy src imports
import sys
import os

# Ensure the Agentic-OS root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
