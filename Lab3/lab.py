"""
lab.py

Lab 3 code

"""

import utils
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2

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

LEVIN: int = 0
NAYAR: int = 1
NAYAR_EQUIVALENT: int = 2

APERTURE_NAMES: list[str] = ["levin", "nayar", "nayar_equiv"]

APERTURE_FOLDER_NAMES: list[str] = [
    "Lab3/psf_levin",
    "Lab3/psf_nayar",
    "Lab3/psf_circ",
]

THEORETICAL_PSF_FOLDER: str = "Lab3/theoretical_psf"

DEPTH_FILE_NAMES: list[str] = [
    "-2", 
    "0", 
    "2"
]
"""The names of the files of each depth"""

DEPTH_DISPLAY_NAMES: list[str] = [
    "Close", 
    "Focus", 
    "Far",
    "Theoretical"
]
"""The names of each depth for display"""

DEPTH_COLORS: list[str] = ['r', 'g', 'b', 'k']
"""Colours used for plotting different depths"""

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

FFT_SIZE: int = 256
"""Size to pad PSF before applying FFT"""

FFT_CMAP: str = "inferno"
"""Colour map to use for the FFT plots."""

# ========================================
# Lab helpers
# ========================================

def process_raw(raw: np.ndarray) -> np.ndarray:
    """Takes in a raw image and returns the black-level removed, 
    fixed pattern removed, green channel image.
    """
    black_removed: np.ndarray = utils.remove_black_level(raw)
    fixed_pattern_removed: np.ndarray = utils.remove_fixed_pattern(black_removed)
    rgb: np.ndarray = utils.debayer_raw(fixed_pattern_removed)
    green: np.ndarray = rgb[:, :, 1]

    return green

def get_fft(img: np.ndarray) -> np.ndarray:
    """Returns the FFT of the image."""
    fft: np.ndarray = np.fft.fftshift(
        np.fft.fft2(img)
    )
    return np.log(np.abs(fft))

def get_fft_db(img: np.ndarray) -> np.ndarray:
    """Returns the FFT magnitude in dB relative to its peak."""
    fft = np.fft.fftshift(np.fft.fft2(img))
    magnitude = np.abs(fft)

    magnitude = np.maximum(magnitude, 1e-12) # Epsilon to avoid division by zero possibiity

    return 20 * np.log10(magnitude / np.max(magnitude))

def pad_image(img: np.ndarray, size: int) -> np.ndarray:
    """Pads the image to the requested size."""
    padded: np.ndarray = np.zeros((size, size), dtype=img.dtype)

    h, w = img.shape
    y_start = (size - h) // 2
    x_start = (size - w) // 2

    padded[
        y_start:y_start + h,
        x_start:x_start + w
    ] = img

    return padded


def get_theoretical_psf(idx: int) -> None:
    """Returns the theoretical PSF and its FFT"""
    psf = DESIGNED_PSFS[idx]

    padded_psf = pad_image(psf, PSF_SIZE)

    utils.save_image(
        padded_psf,
        f"{APERTURE_NAMES[idx]}_psf",
        THEORETICAL_PSF_FOLDER,
        "gray"
    )

    return padded_psf

def crop_from_brightest(img: np.ndarray, crop_size: int) -> np.ndarray:
    """Crop around the brightest blob, favouring blobs with their centre in the crop."""

    # Smooth the image so that we find the centre of a bright region
    # rather than an individual bright pixel.
    brightness = gaussian_filter(img.astype(float), crop_size / 4)

    # Get co-ordinates of brightest pixel after blurring
    max_y, max_x = np.unravel_index( # pylint: disable=unbalanced-tuple-unpacking
        np.argmax(brightness),
        brightness.shape,
    )

    y_start = max_y - crop_size // 2
    x_start = max_x - crop_size // 2

    return img[
        y_start:y_start + crop_size,
        x_start:x_start + crop_size,
    ]

def radial_line_trace(img: np.ndarray, size: int = FFT_SIZE) -> np.ndarray:
    """Return the plottable line-trace"""

    center = (size // 2, size // 2)

    angles = size
    max_radius = size // 2

    polar = cv2.warpPolar( # pylint: disable=no-member
        img,
        (max_radius, angles),
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR # pylint: disable=no-member
    )

    mean_trace = np.mean(polar, axis=0)
    min_trace = np.min(polar, axis=0)

    # Pixel-radius coordinate for each row
    radius = np.arange(max_radius)

    return radius, mean_trace, min_trace

# ========================================
# Lab part subroutines
# ========================================

def create_circle_psf() -> None:
    """Creates the theoretical circle PSF."""
    y, x = np.ogrid[:16, :16]
    circle: np.ndarray = ((x - 8)**2 + (y - 8)**2 <= 8**2).astype(int)

    DESIGNED_PSFS.append(circle)

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
        img: np.ndarray = process_raw(raw)

        img_crop = crop_from_brightest(img, PSF_SIZE)
        fft: np.ndarray = get_fft(img_crop)

        utils.save_image(img_crop, f"{i}", P1_PNG_PATH, "gray")
        utils.save_image(fft, f"{i}", P1_FFT_PATH, FFT_CMAP)

def aperture_psf_depth(idx: int) -> None:
    """Part 6a"""

    # ====================
    # Local members
    # ====================

    folder_name: str = APERTURE_FOLDER_NAMES[idx]
    # Store the fft graphs to be able to have a unified colour bar
    ffts: list[np.ndarray] = []
    # Store line-plot values for plotting all at once
    line_plot_radii: list[np.ndarray] = []
    line_plot_mean_trace: list[np.ndarray] = []
    line_plot_min_trace: list[np.ndarray] = []

    psf_fig, psf_axes = plt.subplots(1, 4)
    fft_fig, fft_axes = plt.subplots(
        1, 4,
        figsize=(16, 4),
        constrained_layout=True
    )
    trace_fig, trace_ax = plt.subplots(figsize=(16, 9))

    # ====================
    # Get the PSFs, FFTs, and line-trace
    # ====================

    for i, (psf_ax, fft_ax) in enumerate(zip(psf_axes, fft_axes)):
        if i == 3: # Skip the theoretical one
            continue

        # Import the files
        file_name = DEPTH_FILE_NAMES[i]
        raw_path: str = f"{folder_name}/{file_name}.dng"

        raw: np.ndarray = utils.read_raw(raw_path)
        img: np.ndarray = process_raw(raw)

        # PSF
        img_crop = crop_from_brightest(img, PSF_SIZE)

        psf_ax.imshow(img_crop, cmap="gray")
        psf_ax.set_title(DEPTH_DISPLAY_NAMES[i])
        psf_ax.axis("off")

        # FFT
        img_pad: np.ndarray = pad_image(img_crop, FFT_SIZE)

        fft: np.ndarray = get_fft_db(img_pad)
        ffts.append(fft)

        # Line trace
        radius, mean_trace, min_trace = radial_line_trace(fft, FFT_SIZE)

        line_plot_radii.append(radius)
        line_plot_mean_trace.append(mean_trace)
        line_plot_min_trace.append(min_trace)

    # ====================
    # Add in theoretical PSF, FFT, and line-trace
    # ====================

    # Get theoretical PSF and FFT
    theoretical_psf: np.ndarray = get_theoretical_psf(idx)
    img_pad: np.ndarray = pad_image(theoretical_psf, FFT_SIZE)
    fft: np.ndarray = get_fft_db(img_pad)
    ffts.append(fft)

    # Line trace
    radius, mean_trace, min_trace = radial_line_trace(fft, FFT_SIZE)

    line_plot_radii.append(radius)
    line_plot_mean_trace.append(mean_trace)
    line_plot_min_trace.append(min_trace)

    # ====================
    # Finish creating plots
    # ====================

    # PSF theoretical
    psf_axes[3].imshow(theoretical_psf, cmap="gray")
    psf_axes[3].set_title("Theoretical")
    psf_axes[3].axis("off")

    # Use one common colour range for all FFTs
    fft_vmin = min(np.min(fft) for fft in ffts)
    fft_vmax = max(np.max(fft) for fft in ffts)

    for fft_ax, fft, name in zip(fft_axes, ffts, DEPTH_DISPLAY_NAMES):
        im = fft_ax.imshow(
            fft,
            cmap=FFT_CMAP,
            vmin=fft_vmin,
            vmax=fft_vmax
        )

        fft_ax.set_title(name)
        fft_ax.axis("off")

    # One shared colour bar
    fft_fig.colorbar(
        im,
        ax=fft_axes,
        label="dB",
        shrink=0.9
    )

    psf_fig.tight_layout()

    # ====================
    # Create line-trace of frequence response over radius
    # ====================

    for radius, mean_trace, min_trace, name, color in zip(
        line_plot_radii,
        line_plot_mean_trace,
        line_plot_min_trace,
        DEPTH_DISPLAY_NAMES,
        DEPTH_COLORS
    ):
        trace_ax.plot(
            radius,
            mean_trace,
            f"{color}-",
            label=f"{name} Mean"
        )
        trace_ax.plot(
            radius,
            min_trace,
            f"{color}--",
            label=f"{name} Min."
        )

    trace_ax.set_xlabel("Distance from centre (pixels)")
    trace_ax.set_ylabel("Intensity")
    trace_ax.set_title(f"{APERTURE_NAMES[idx]} radial line trace")
    trace_ax.grid(True, alpha=0.3)
    trace_ax.legend()

    trace_fig.tight_layout()

    # ====================
    # Save plots
    # ====================

    psf_fig.savefig(
        f"{folder_name}_psf_depth.png",
        dpi=300,
        bbox_inches="tight"
    )

    fft_fig.savefig(
        f"{folder_name}_fft_depth.png",
        dpi=300,
        bbox_inches="tight"
    )

    trace_fig.savefig(
        f"{folder_name}_fft_trace.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(psf_fig)
    plt.close(fft_fig)
    plt.close(trace_fig)


# ========================================
# Main
# ========================================

if __name__ == "__main__":
    # Setup
    create_circle_psf()

    # Lab subroutines

    # show_fixed_noise() # Part 0
    # lens_psf_process() # Part 1

    # Part 6
    aperture_psf_depth(LEVIN)
    # aperture_psf_depth(NAYAR)
    aperture_psf_depth(NAYAR_EQUIVALENT)
