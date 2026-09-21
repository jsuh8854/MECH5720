"""
Lab 4
"""

import os
from PIL import Image
import utils
import yaml
import numpy as np
from matplotlib import pyplot as plt

# 24 mm diameter

# 39 mm long tube
# 18 mm short tube

# ===================================
# Functions yippie
# ===================================

def get_autocorrelation(img: np.ndarray) -> np.ndarray:
    """Calculates and returns the auto-correlation from raw data."""
    return get_autocorrelation_from_fft( np.abs(np.fft.fft2(img)) ** 2 )

def get_autocorrelation_from_fft(fft: np.ndarray) -> np.ndarray:
    """
    Calculates and returns the auto-correlation from fft.
    Use get_autocorrelation instead usually.
    """
    return np.real(np.fft.ifft2(fft))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    psf = Image.open("psf_sample.tif")
    plt.imshow(psf, cmap="gray")
    plt.show()

    # psf = np.fft.fftshift(psf)

    ac = get_autocorrelation(psf)
    plt.imshow(np.fft.fftshift(ac), cmap="gray")
    plt.show()
