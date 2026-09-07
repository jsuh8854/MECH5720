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


MAX_PIXEL : float = 1023
"""Max value of RAW pixels."""

BLACK_LEVEL: int = 16
"""Black level of sensor."""

FIXED_PATTERN_PICKLE_PATH: str = "fixed_pattern.pkl"
"""Path to pickle file with fixed pattern noise"""

fixed_pattern: np.ndarray
"""Fixed pattern of sensor."""

# Load pickle
try:
    with open(FIXED_PATTERN_PICKLE_PATH, "rb") as f:
        fixed_pattern = pickle.load(f)
except FileNotFoundError:
    print("Warning: Fixed pattern pickle not found.")


def remove_black_level(img: np.ndarray) -> np.ndarray:
    """Removes the black level from the image."""
    return img - BLACK_LEVEL

def remove_fixed_pattern(img: np.ndarray) -> np.ndarray:
    """Removes the black level from the image."""
    return img - fixed_pattern

def reduce_image(img: np.ndarray, scale: int) -> np.ndarray:
    """ Digital gain image to maximise brightness, at the cost of noise being amplified.
        With gain of -1.0, it automatically scales to maximum brightness.
    """
    return img[::scale, ::scale]

def brighten_image(img: np.ndarray, gain: float = -1.0) -> np.ndarray:
    """ Digital gain image to maximise brightness, at the cost of noise being amplified.
        With gain of -1.0, it automatically scales to maximum brightness.
    """
    max_brightness: float = np.max(img)

    if gain < 0.0:
        gain = MAX_PIXEL / max_brightness

    brightened_image: np.ndarray = img * gain

    return brightened_image

def rotate_image(img: np.ndarray, k: int = -1) -> np.ndarray:
    """Rotate the image, default value of -1 rotates to match lab setup."""
    return np.rot90(img, k=k)

def white_balance_rgb(rgb: np.ndarray, p: float = 4.0) -> np.ndarray:
    """Returns a Minkowski white balanced image."""
    # Estimate illuminant strength for each channel.
    norms = np.mean(np.abs(rgb) ** p, axis=(0, 1)) ** (1.0 / p)

    # Normalize so the geometric mean gain is 1.
    gains = np.exp(np.mean(np.log(norms))) / norms

    balanced = rgb * gains

    return balanced

def debayer_raw(raw: np.ndarray) -> np.ndarray:
    """Convert the raw DNG into an float32 debayered RGB image, needs white balance."""
    raw = np.asarray(raw, dtype=np.uint16, order="C")

    rgb16 = cv2.cvtColor(raw, cv2.COLOR_BayerGR2RGB) # pylint: disable=no-member
    rgb = rgb16.astype(np.float32)

    return rgb

def read_raw(filename: str) -> np.ndarray:
    """Reads the DNG file and returns as an np.ndarray."""
    try:
        # Load and view the raw bayer image
        with rawpy.imread(filename) as raw:
            return np.array(raw.raw_image, dtype=np.int16, copy=True, order="C")

    except FileNotFoundError as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)

def convert_to_rgb8(img: np.ndarray) -> np.ndarray:
    """Convert the float32 rgb data to rgb8 for export."""
    return np.clip(img * 256.0 / (MAX_PIXEL + 1), 0, 255).astype(np.uint8)

def normalise_float_image(img: np.ndarray) -> np.ndarray:
    """Convert from [0, MAX_PIXEL] to [0, 1]."""
    return img / MAX_PIXEL

def show_image(data: np.ndarray, filename: str = "You forgot the title!") -> None:
    """Show the image data in a Matplotlib graph."""
    # Convert to uint8 format for display.
    rgb8 = convert_to_rgb8(data)
    # Display using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(rgb8)
    plt.axis("off")  # Hide coordinate axes
    plt.title(filename)
    plt.tight_layout()
    plt.show()

def save_image(data: np.ndarray, filename: str, output_path: str) -> None:
    """Output the data as an float RGB png at the provided path and file name."""
    norm = normalise_float_image(data)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{filename}.png"

    plt.imsave(output_file, norm)
