"""
lab.py

Lab 3 code

"""

import utils
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
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

P7_RAW_PATH: str = "Lab3/mtf"
"""Folder path to siemen star raws"""

CENTRE_NAME: str = "Lab3/arst.png"
"""Name of the centre dngs for Part 2."""

LEVIN: int = 0
NAYAR: int = 1
NAYAR_EQUIVALENT: int = 2

APERTURE_NAMES: list[str] = ["levin", "nayar", "circ"]

APERTURE_FOLDER_NAMES: list[str] = [
    "Lab3/psf_levin",
    "Lab3/psf_nayar",
    "Lab3/psf_circ",
]

THEORETICAL_PSF_FOLDER: str = "Lab3/theoretical_psf"
"""Folder to save the theoretical psfs"""

CIRCLES_FOLDER: str = "Lab3/circles"
"""Folder to save the Siemen Star images"""

DEPTH_FILE_NAMES: list[str] = [
    "",
    "0", 
    "1", 
    "2",
    "top_left",
    "bot_left",
    "top_right",
    "bot_right"
]
"""The names of the files of each depth and corner."""

DEPTH_DISPLAY_NAMES: list[str] = [
    "Theoretical",
    "Focus", 
    "Far", 
    "Farther",
    "Top Left",
    "Bottom Left",
    "Top Right",
    "Bottom Right",
]
"""The names of each depth for display"""

DEPTH_COLORS: list[str] = [
    "k",
    "r",
    "g",
    "b",
    "tab:orange",
    "tab:purple",
    "tab:brown",
    "tab:cyan",

]
"""Colours used for plotting different depths"""


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

THEORETICAL_PSF_SCALES : list[float] = [2.0, 2.0, 1.25]
"""Experimentally obtained scales for the PSF."""

PSF_SIZE: int = 128
"""Size of psf crop"""

FFT_SIZE: int = 256
"""Size to pad PSF before applying FFT"""

FFT_CMAP: str = "inferno"
"""Colour map to use for the FFT plots."""

STAR_CENTERS: list[tuple[int, int]] = [
    (996, 557),
    (998, 525),
    (1025, 594),
]
"""(x, y) centres of the stars, found from trial and error..."""

STAR_PADDING_OFFSET: tuple[int, int] = (540, 960)
"""Offset to the centre introduced from padding"""

STAR_PAD_SIZE: int = 3000
"""Size to pad image for radius finding."""

STAR_BLACK_THRESHOLD: int = 16
"""Brightness to count as a black band for finding the radius of the star"""

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

def pad_image(img: np.ndarray, size: int, value: int = 0) -> np.ndarray:
    """Pads the image to the requested size and value if requested."""
    padded: np.ndarray = np.full(
        (size, size),
        value,
        dtype=img.dtype,
    )

    h, w = img.shape
    y_start = (size - h) // 2
    x_start = (size - w) // 2

    padded[
        y_start:y_start + h,
        x_start:x_start + w,
    ] = img

    return padded

def get_theoretical_psf(idx: int, scale: float = 1.0) -> np.ndarray:
    """Return the theoretical PSF, scaled and padded."""

    psf = DESIGNED_PSFS[idx].astype(np.float32)

    if scale != 1.0:
        height, width = psf.shape[:2]
        psf = cv2.resize( # pylint: disable=no-member
            psf,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_LINEAR # pylint: disable=no-member
        )

    padded_psf = pad_image(psf, PSF_SIZE)

    utils.save_image(
        padded_psf,
        f"{APERTURE_NAMES[idx]}_psf",
        THEORETICAL_PSF_FOLDER,
        "gray",
    )

    return padded_psf

def crop_from_brightest(img: np.ndarray, crop_size: int) -> np.ndarray:
    """Crop around the brightest blob, clamping the crop to image edges."""

    height, width = img.shape[:2]

    if crop_size > height or crop_size > width:
        raise ValueError(
            f"crop_size ({crop_size}) must not exceed image dimensions "
            f"({width}x{height})"
        )

    # Smooth the image so that we find the centre of a bright region
    # rather than an individual bright pixel.
    brightness = gaussian_filter(img.astype(float), crop_size / 4)

    # Get coordinates of brightest pixel after blurring.
    max_y, max_x = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
        np.argmax(brightness),
        brightness.shape,
    )

    # Centre the crop on the brightest point, then clamp the start
    # coordinate so the crop remains entirely inside the image.
    y_start = np.clip(max_y - crop_size // 2, 0, height - crop_size)
    x_start = np.clip(max_x - crop_size // 2, 0, width - crop_size)

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

def get_contrast_per_angle_at_radius(polar: np.ndarray, angles: np.ndarray, radius: int) -> np.ndarray:
    """Returns contrast as a function of angle"""
    circle = polar[:, radius]

    peaks, _ = find_peaks(circle, distance=16)
    troughs, _ = find_peaks(-circle, distance=16)

    i_max = np.interp(angles, peaks, circle[peaks])
    i_min = np.interp(angles, troughs, circle[troughs])
    return np.abs((i_max - i_min) / (i_max + i_min))

def get_average_contrast_per_radius(polar: np.ndarray, max_radius: int) -> np.ndarray:
    """Returns a list of contrast values at each radius"""
    contrast = np.zeros(max_radius)

    for r in range(max_radius):
        peaks, _ = find_peaks(polar[:, r], distance=16)
        troughs, _ = find_peaks(-polar[:, r], distance=16)

        i_max = np.mean(peaks)
        i_min = np.mean(troughs)

        contrast[r] = np.abs((i_max - i_min) / (i_max + i_min))

    return contrast

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
    """Part 6"""

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

    psf_fig, psf_axes = plt.subplots(2, 4)
    fft_fig, fft_axes = plt.subplots(
        2, 4,
        figsize=(16, 8),
        constrained_layout=True
    )
    trace_ave_fig, trace_ave_ax = plt.subplots(figsize=(16, 9))
    trace_min_fig, trace_min_ax = plt.subplots(figsize=(16, 9))

    # ====================
    # Add in theoretical PSF, FFT, and line-trace
    # ====================

    # Get theoretical PSF and FFT
    theoretical_psf: np.ndarray = get_theoretical_psf(idx, THEORETICAL_PSF_SCALES[idx])
    img_pad: np.ndarray = pad_image(theoretical_psf, FFT_SIZE)
    fft: np.ndarray = get_fft_db(img_pad)
    ffts.append(fft)

    # Line trace
    radius, mean_trace, min_trace = radial_line_trace(fft, FFT_SIZE)

    line_plot_radii.append(radius)
    line_plot_mean_trace.append(mean_trace)
    line_plot_min_trace.append(min_trace)

    # Add PSF theoretical to graph
    psf_axes[0, 0].imshow(theoretical_psf, cmap="gray")
    psf_axes[0, 0].set_title("Theoretical")
    psf_axes[0, 0].axis("off")

    # ====================
    # Get the PSFs, FFTs, and line-trace of measured
    # ====================

    for i, (psf_ax, fft_ax) in enumerate(zip(psf_axes.flat, fft_axes.flat)):
        # Skip theoretical
        if i == 0:
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
    # Finish creating plots
    # ====================

    for fft_ax, fft, name in zip(fft_axes.flat, ffts, DEPTH_DISPLAY_NAMES):
        im = fft_ax.imshow(
            fft,
            cmap=FFT_CMAP,
            vmin=-100,
            vmax=0
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

    for i, (radius, mean_trace, min_trace, name, color) in enumerate(zip(
        line_plot_radii,
        line_plot_mean_trace,
        line_plot_min_trace,
        DEPTH_DISPLAY_NAMES,
        DEPTH_COLORS
    )):
        trace_ave_ax.plot(
            radius,
            mean_trace,
            color=color,
            linestyle="-" if i == 0 else ("--" if i < 4 else ":"),
            label=f"{name}"
        )
        trace_min_ax.plot(
            radius,
            min_trace,
            color=color,
            linestyle="-" if i == 0 else ("--" if i < 4 else ":"),
            label=f"{name}"
        )

    trace_ave_ax.set_xlabel("Distance from centre (pixels)")
    trace_ave_ax.set_ylabel("Intensity")
    trace_ave_ax.set_title(f"{APERTURE_NAMES[idx]} radial line trace (average)")
    trace_ave_ax.grid(True, alpha=0.3)
    trace_ave_ax.legend()

    trace_ave_fig.tight_layout()

    trace_min_ax.set_xlabel("Distance from centre (pixels)")
    trace_min_ax.set_ylabel("Intensity")
    trace_min_ax.set_title(f"{APERTURE_NAMES[idx]} radial line trace (minimum)")
    trace_min_ax.grid(True, alpha=0.3)
    trace_min_ax.legend()

    trace_min_fig.tight_layout()

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

    trace_ave_fig.savefig(
        f"{folder_name}_fft_trace_ave.png",
        dpi=300,
        bbox_inches="tight"
    )

    trace_min_fig.savefig(
        f"{folder_name}_fft_trace_min.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(psf_fig)
    plt.close(fft_fig)
    plt.close(trace_ave_fig)
    plt.close(trace_min_fig)

def compute_contrast_and_mtf(idx: int) -> None:
    """Part 7"""

    # --------------------
    # Load image
    # --------------------
    file_name = APERTURE_NAMES[idx]

    raw_path: str = f"{P7_RAW_PATH}/mtf_{file_name}.dng"
    raw: np.ndarray = utils.read_raw(raw_path)
    img: np.ndarray = process_raw(raw)

    # utils.save_image(img, file_name, CIRCLES_FOLDER, cmap="gray")

    # --------------------
    # Calculate contrast
    # --------------------

    # Convert to polar to find radius
    angle_resolution = 3

    angle_count = 360 * angle_resolution
    max_radius = 1500

    img_padded = pad_image(img, STAR_PAD_SIZE, 32)

    polar = cv2.warpPolar(  # pylint: disable=no-member
        img_padded,
        (max_radius, angle_count),
        (
            STAR_CENTERS[idx][0] + STAR_PADDING_OFFSET[0],
            STAR_CENTERS[idx][1] + STAR_PADDING_OFFSET[1]
        ),
        max_radius,
        cv2.WARP_POLAR_LINEAR # pylint: disable=no-member
    )

    # Scan through each column until there's no more than 1% of pixels
    # below STAR_BLACK_THRESHOLD on that column
    black_fraction = np.mean(
        polar <= STAR_BLACK_THRESHOLD,
        axis=0,
    )

    # Start from somewhere in the middle to not get a false early reading
    start_offset = 128

    valid_columns = black_fraction[start_offset:] <= 0.01
    radius = np.argmax(valid_columns) + start_offset
    angles = np.arange(angle_count)

    close_contrast = get_contrast_per_angle_at_radius(polar, angles, radius // 3)
    middle_contrast = get_contrast_per_angle_at_radius(polar, angles,radius // 2)
    far_contrast = get_contrast_per_angle_at_radius(polar, angles,2 * radius // 3)

    fig, ax = plt.subplots(figsize=(16, 9))

    ax.plot(angles // angle_resolution, close_contrast, "r", label="Close")
    ax.plot(angles // angle_resolution, middle_contrast, "g", label="Middle")
    ax.plot(angles // angle_resolution, far_contrast, "b", label="Far")
    ax.set_ylabel("Contrast")
    ax.set_xlabel("Angle")
    ax.set_xticks(np.arange(0, 361, 30))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        f"Lab3/mtf_{APERTURE_NAMES[idx]}_contrast.png",
        dpi=300,
        bbox_inches="tight"
    )

    # --------------------
    # Calculate MTF
    # --------------------

    n: int = 64 # Spoke count
    r = np.arange(radius) # Radii
    frequency = n / (2 * np.pi * r)
    mtf = get_average_contrast_per_radius(polar, radius)
    mtf_norm = mtf / 0.1
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(frequency[5:], mtf_norm[5:])
    ax.set_xlabel("Spatial frequency (cycles/pixel)")
    ax.set_ylabel("MTF normalised magnitude")
    ax.set_title(f"{APERTURE_NAMES[idx]} MTF from Siemens-star target")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        f"Lab3/mtf_{APERTURE_NAMES[idx]}_MTF.png",
        dpi=300,
        bbox_inches="tight"
    )


# ========================================
# Main
# ========================================

if __name__ == "__main__":
    # --------------------
    # Setup
    # --------------------

    create_circle_psf()

    # --------------------
    # Lab subroutines
    # --------------------

    # show_fixed_noise() # Part 0
    # lens_psf_process() # Part 1

    # Part 6
    # aperture_psf_depth(LEVIN)
    # aperture_psf_depth(NAYAR)
    # aperture_psf_depth(NAYAR_EQUIVALENT)

    # Part 7
    # compute_contrast_and_mtf(LEVIN)
    # compute_contrast_and_mtf(NAYAR)
    # compute_contrast_and_mtf(NAYAR_EQUIVALENT)

    # Part 8
    
