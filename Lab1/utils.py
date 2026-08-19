import numpy as np
import cv2
import rawpy
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

MAX_PIXEL = 1024


def update_working_directory() -> None:
    abspath = os.path.abspath(__file__)
    dirname = os.path.dirname(abspath)
    os.chdir(dirname)

# Digital gain image to maximise brightness, at the cost of noise being amplified.
def brighten_image(img: np.ndarray) -> np.ndarray:
    max_brightness : float = np.max(img)
    
    brightened_image : np.ndarray = img * 1024.0 / max_brightness

    return brightened_image

# Rotate the image, default value of -1 rotates to match lab setup.
def rotate_image(img: np.ndarray, k: int = -1) -> np.ndarray:
    return np.rot90(img, k=k)

# Returns a Minkowski white balanced image.
def white_balance_rgb(rgb: np.ndarray, p: float = 4.0) -> np.ndarray:
    # Estimate illuminant strength for each channel.
    norms = np.mean(np.abs(rgb) ** p, axis=(0, 1)) ** (1.0 / p)

    # Normalize so the geometric mean gain is 1.
    gains = np.exp(np.mean(np.log(norms))) / norms

    balanced = rgb * gains

    return balanced

# Convert the raw DNG into an float32 debayered RGB image, needs white balance.
def debayer_raw(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.uint16, order="C")

    rgb16 = cv2.cvtColor(raw, cv2.COLOR_BayerGR2RGB)
    rgb = rgb16.astype(np.float32)

    return rgb

# Reads the DNG file and returns as an np.ndarray.
def read_raw(filename: str) -> np.ndarray:
    try:
        # Load and view the raw bayer image
        with rawpy.imread(filename) as raw:
            return np.array(raw.raw_image, dtype=np.uint16, copy=True, order="C")

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)


def show_image(data: np.ndarray, filename: str = "You forgot the title!") -> None:
    # Convert to uint8 format for display
    rgb8 = np.clip(data / 4.0, 0, 255).astype(np.uint8)
    # Display using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(rgb8)
    plt.axis("off")  # Hide coordinate axes
    plt.title(filename)
    plt.tight_layout()
    plt.show()

def save_image(data: np.ndarray, filename: str, output_path: str) -> None:
    # Convert 10-bit float data to uint8
    rgb8 = np.clip(data / 4.0, 0, 255).astype(np.uint8)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{filename}.png"

    plt.imsave(output_file, rgb8)