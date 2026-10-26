"""
General utility functions used for the lab.
"""

import pickle
import sys
from pathlib import Path
import numpy as np
import cv2
import rawpy
import matplotlib.pyplot as plt

# ========================================
# Module constants and members
# ========================================

MAX_PIXEL : float = 1023
"""Max value of RAW pixels."""

BLACK_LEVEL: int = 16
"""Black level of sensor."""

FIXED_PATTERN_PICKLE_PATH: str = "fixed_pattern.pkl"
"""Path to pickle file with fixed pattern noise"""

fixed_pattern: np.ndarray
"""Fixed pattern of sensor."""

# ========================================
# Initialise module
# ========================================

# Load pickle
try:
    with open(FIXED_PATTERN_PICKLE_PATH, "rb") as f:
        fixed_pattern = pickle.load(f)
    print("Info: Fixed pattern pickle loaded.")
except FileNotFoundError:
    print("Warning: Fixed pattern pickle not found.")


# ========================================
# Module defs
# ========================================

def read_raw(filename: str) -> np.ndarray:
    """Reads the DNG file and returns as an np.ndarray."""
    try:
        # Load and view the raw bayer image
        with rawpy.imread(filename) as raw:
            return np.array(raw.raw_image, dtype=np.int16, copy=True, order="C")

    except FileNotFoundError as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)

def debayer_raw(raw: np.ndarray) -> np.ndarray:
    """Convert raw DNG into a float32 debayered RGB image."""
    raw = np.asarray(raw, dtype=np.int16, order="C")
    raw = np.clip(raw, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    rgb16 = cv2.cvtColor(raw, cv2.COLOR_BayerGR2RGB)  # pylint: disable=no-member

    return rgb

def rotate_image(img: np.ndarray, k: int = -1) -> np.ndarray:
    """Rotate the image, default value of -1 rotates to match lab setup."""
    return np.rot90(img, k=k)

def process_raw(filename: str) -> np.ndarray:
    """Do full pre-processing on raw image from provided file path."""
    raw = read_raw(filename)
    debayer = debayer_raw(raw)
    green = debayer[:,:,1] - BLACK_LEVEL
    if fixed_pattern:
        green -= fixed_pattern
    rotated = rotate_image(green)
    
    return rotated
