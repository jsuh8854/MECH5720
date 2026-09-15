"""
lab.py

Lab 3 code

"""

import utils
import numpy as np
from matplotlib import pyplot as plt


# ========================================
# Constants and members
# ========================================

P1_RAW_PATH: str = "Lab3/lab3_p1_cap"
"""Folder path to PSF characterisation raws."""

P1_PNG_PATH: str = "Lab3/p1_pngs"
"""Folder path where to save Part 1 debayered images."""

P1_FFT_PATH: str = "Lab3/p1_ffts"
"""blurg"""

CENTRE_NAME: str = "Lab3/arst.png"
"""Name of the centre dngs for Part 2."""

APERTURE_NAMES: list[str] = [
    "Lab3/psf_levin",
    "Lab3/psf_nayar",
    "Lab3/psf_circ",
]

DEPTH_NAMES: list[str] = [
    "-2", 
    "0", 
    "2"
]

CORNER_NAMES: list[str] = [
    "top_left",
    "bot_left",
    "top_right",
    "bot_right",
]
"""Names of the corner dngs for Part 2."""

DESIGNED_PSFS: list[np.ndarray] = [
    np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,1,1,0,1,0,1,1,0,0,0,],
        [0,0,0,0,0,0,1,0,0,0,0,0,0,],
        [0,1,1,0,0,0,0,0,0,0,1,1,0,],
        [0,0,1,0,0,1,1,1,0,0,1,0,0,],
        [0,1,0,0,0,0,1,0,0,0,0,1,0,],
        [0,1,0,0,1,1,1,1,1,0,0,1,0,],
        [0,1,0,0,0,0,1,0,0,0,0,1,0,],
        [0,0,1,0,0,1,1,1,0,0,1,0,0,],
        [0,1,1,0,0,0,0,0,0,0,1,1,0,],
        [0,0,0,0,0,0,1,0,0,0,0,0,0,],
        [0,0,0,1,1,0,1,0,1,1,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,]
    ]),
    np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0,0,],
        [0,0,0,1,0,1,1,1,1,1,0,0,0,],
        [0,0,0,0,0,0,0,1,1,0,0,0,0,],
        [0,1,0,1,0,0,0,1,1,1,1,1,0,],
        [0,1,0,0,1,1,1,1,0,0,0,1,0,],
        [0,1,1,0,0,0,0,0,0,1,0,1,0,],
        [0,1,1,0,1,1,1,0,0,1,0,0,0,],
        [0,1,1,0,1,0,0,0,1,1,0,1,0,],
        [0,1,1,0,1,0,0,0,0,0,0,1,0,],
        [0,1,1,0,0,1,0,1,0,0,1,1,0,],
        [0,0,1,1,1,0,0,1,1,1,1,0,0,],
        [0,0,0,0,0,0,0,1,1,1,0,0,0,],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,],
    ])
]

PSF_SIZE: int = 128
"""Size of psf crop"""

# ========================================
# Lab part subroutines
# ========================================

def show_fixed_noise() -> None:
    """Part 0"""
    plt.imshow(utils.fixed_pattern, cmap="gray", vmax=np.mean(utils.fixed_pattern))
    plt.axis("off")
    plt.title("Fixed Pattern Noise")
    plt.show()

def lens_psf_process() -> None:
    """Part 1"""

    for i in range(-2, 3):
        raw_path: str = f"{P1_RAW_PATH}/{i}.dng"

        raw: np.ndarray =  utils.read_raw(raw_path)
        black_removed: np.ndarray = utils.remove_black_level(raw)
        fixed_pattern_removed: np.ndarray = utils.remove_fixed_pattern(black_removed)
        rgb: np.ndarray = utils.debayer_raw(fixed_pattern_removed)
        green: np.ndarray = rgb[:, :, 1]

        utils.save_image(green, f"{i}", P1_PNG_PATH, greyscale=True)

        fft: np.ndarray = np.fft.fft2(green)
        fft_norm: np.ndarray = np.log(np.abs(fft))

        utils.save_image(fft_norm, f"{i}", P1_FFT_PATH)

def lens_aperture_psf_process(idx: int) -> None:
    """Part 2-6a"""

    folder_name: str = APERTURE_NAMES[idx]

    for file_name in DEPTH_NAMES:
        raw_path: str = f"{folder_name}/{file_name}.dng"

        raw: np.ndarray = utils.read_raw(raw_path)
        black_removed: np.ndarray = utils.remove_black_level(raw)
        fixed_pattern_removed: np.ndarray = utils.remove_fixed_pattern(black_removed)

        rgb: np.ndarray = utils.debayer_raw(fixed_pattern_removed)
        green: np.ndarray = rgb[:, :, 1]

        # Find the location of the maximum green value
        max_y, max_x = np.unravel_index(np.argmax(green), green.shape)

        # Crop a 64x64 region centered on the maximum
        y_start = max_y - PSF_SIZE // 2
        x_start = max_x - PSF_SIZE // 2

        # Keep the crop within image bounds
        y_start = np.clip(y_start, 0, green.shape[0] - PSF_SIZE)
        x_start = np.clip(x_start, 0, green.shape[1] - PSF_SIZE)

        green_crop = green[
            y_start:y_start + PSF_SIZE,
            x_start:x_start + PSF_SIZE
        ]

        utils.save_image(
            green_crop,
            file_name,
            f"{folder_name}_debayer",
            greyscale=True
        )

        # FFT
        fft: np.ndarray = np.fft.fftshift(np.fft.fft2(green_crop))
        fft_norm: np.ndarray = np.log(np.abs(fft))

        utils.save_image(
            fft_norm,
            file_name,
            f"{folder_name}_fft"
        )

    show_theoretical_psf(idx)

    for file_name in CORNER_NAMES:
        continue
        # I'll fix latetr
        raw_path: str = f"{folder_name}/{file_name}.dng"

        raw: np.ndarray =  utils.read_raw(raw_path)
        black_removed: np.ndarray = utils.remove_black_level(raw)
        fixed_pattern_removed: np.ndarray = utils.remove_fixed_pattern(black_removed)

        rgb: np.ndarray = utils.debayer_raw(fixed_pattern_removed)
        green: np.ndarray = rgb[:, :, 1]

        utils.save_image(green, file_name, f"{folder_name}_debayer", greyscale=True)

def show_theoretical_psf(idx: int) -> None:
    """oh my god i don't care"""
    psf = DESIGNED_PSFS[idx]

    padded_psf = np.zeros((PSF_SIZE, PSF_SIZE), dtype=psf.dtype)

    h, w = psf.shape
    y_start = (PSF_SIZE - h) // 2
    x_start = (PSF_SIZE - w) // 2

    padded_psf[
        y_start:y_start + h,
        x_start:x_start + w
    ] = psf

    plt.imshow(padded_psf)
    plt.show()

    fft: np.ndarray = np.fft.fftshift(np.fft.fft2(padded_psf))
    fft_norm: np.ndarray = np.log(np.abs(fft) + 1e-12)

    plt.imshow(fft_norm)
    plt.show()


# ========================================
# Main
# ========================================

if __name__ == "__main__":
    # show_fixed_noise() # Part 0
    # lens_psf_process() # Part 1

    # Part 6
    lens_aperture_psf_process(0)
