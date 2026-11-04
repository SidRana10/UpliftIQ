"""Test session setup."""

import os

# One OpenMP thread for PyTorch: its runtime clashes with LightGBM's in the same process on macOS
# (see modeling/criteo_tarnet.py). Must be set before torch is imported by any test module.
os.environ.setdefault("OMP_NUM_THREADS", "1")
