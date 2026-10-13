"""
Lab 4
"""

import os
from PIL import Image
import utils
import numpy as np
from matplotlib import pyplot as plt
from ADMM import reconstruct_image

# 24 mm diameter

# 39 mm long tube
# 18 mm short tube

# ===================================
# Paths
# ===================================

PART_1_FOLDER: str = "part1"

PART_1_OUT: str = "part1_out"

# ===================================
# Functions yippie
# ===================================

def get_autocorrelation(img: np.ndarray) -> np.ndarray:
    """Calculates and returns the auto-correlation from raw data."""
    return get_autocorrelation_from_fft( np.abs(np.fft.fft2(img)) ** 2 )

def get_autocorrelation_from_fft(fft: np.ndarray) -> np.ndarray:
    """
    Calculates and returns the normalised autocorrelation from the FFT power spectrum.
    The maximum autocorrelation value is normalised to 1.
    """
    ac = np.real(np.fft.ifft2(fft))

    max_val = np.max(ac)
    if max_val != 0:
        ac = ac / max_val

    return ac
def get_radial_profile(autocorr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return radii and angle averaged intensity of a 2D autocorrelation.
    """

    # Centre the zero-lag peak
    autocorr = np.fft.fftshift(autocorr)

    # Pixel coordinates relative to the centre
    h, w = autocorr.shape
    y, x = np.indices((h, w))
    r = np.sqrt((x - w // 2)**2 + (y - h // 2)**2)

    # Bin pixels by integer radius
    r_int = r.astype(np.int32)

    # Compute mean intensity at each radius
    radial_sum = np.bincount(r_int.ravel(), weights=autocorr.ravel())
    radial_count = np.bincount(r_int.ravel())
    radial_profile = radial_sum / radial_count

    radii = np.arange(len(radial_profile))

    return radii, radial_profile


# ===================================
# Sections
# ===================================
def part_1():
    """Part 1"""

    fig, ax = plt.subplots()

    for file_name in ["test"]:  # Add more filenames here
        psf = Image.open(f"{PART_1_FOLDER}/{file_name}.png")
        psf = np.asarray(psf)
        # psf = psf[:,:,1]

        # Save PSF image
        psf_fig, psf_ax = plt.subplots()
        psf_ax.imshow(psf, cmap="gray")
        psf_ax.axis("off")
        psf_fig.tight_layout()
        psf_fig.savefig(
            f"{PART_1_OUT}/psf_{file_name}.png",
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close(psf_fig)

        # Calculate autocorrelation and radial profile
        ac = get_autocorrelation(psf)
        radii, radial_profile = get_radial_profile(ac)

        # Add this image's profile to the shared plot
        ax.plot(radii, radial_profile, label=file_name)

    # Format and save the plot
    ax.set_xlabel("Radius")
    ax.set_ylabel("Intensity")
    ax.set_title("Average Intensity over Radius")
    ax.grid(True)
    ax.legend()
    ax.set_xlim(left=-50)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(f"{PART_1_OUT}/auto_correlation.png")
    plt.close(fig)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(PART_1_OUT, exist_ok=True)

    part_1()
