"""
lab.py

Lab 3 code
"""

import utils
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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

P9_RAW_PATH: str = "Lab3/part9"
"""blegh"""

P9_PNG_PATH: str = "Lab3/p9_pngs"
"""Folder path where to save Part 9 debayered images."""

P9_FFT_PATH: str = "Lab3/p9_ffts"
"""blupu"""

P9_DECONV_PATH: str = "Lab3/p9_deconvs"
"""Folder path where to save Part 9 deconvolved images."""

BONUS_RAW_PATH: str = "Lab3/bonus"
"""Path to bonus images"""

BONUS_PNG_PATH: str = "Lab3/bonus"
"""Path to bonus images"""

MONOCHROME_CMAPS = [
    LinearSegmentedColormap.from_list(
        "red",
        [(0, 0, 0), (1, 0, 0)]
    ),
    LinearSegmentedColormap.from_list(
        "green",
        [(0, 0, 0), (0, 1, 0)]
    ),
    LinearSegmentedColormap.from_list(
        "blue",
        [(0, 0, 0), (0, 0, 1)]
    ),
]

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

PSF_SIZE_2: int = 256
"""Size of psf crop for part 9 as it has a better capture"""


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

GOLD_STANDARD_COUNT: int = 10
"""Number of gold standard shots"""

NATURAL_FOLDERS: list[str] = [
    "Lab3/nat_levin",
    "Lab3/nat_nayar",
    "Lab3/nat_circ"
]
"""Folder for natural image raws"""

# ========================================
# Lab helpers
# ========================================

def process_raw(raw: np.ndarray, greyscale: bool = True) -> np.ndarray:
    """Takes in a raw image and returns the black-level removed, 
    fixed pattern removed. Green only if greyscale is True.
    """
    black_removed: np.ndarray = utils.remove_black_level(raw)
    fixed_pattern_removed: np.ndarray = utils.remove_fixed_pattern(black_removed)
    rgb: np.ndarray = utils.debayer_raw(fixed_pattern_removed)
    if greyscale:
        return rgb[:, :, 1]

    return rgb

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

def pad_image(img: np.ndarray, size: (int, int), value: int = 0) -> np.ndarray:
    """Pads the image to the requested size (y, x) and value if requested."""
    padded: np.ndarray = np.full(
        size,
        value,
        dtype=img.dtype,
    )

    h, w = img.shape
    y_start = (size[0] - h) // 2
    x_start = (size[1] - w) // 2

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

    padded_psf = pad_image(psf, (PSF_SIZE, PSF_SIZE))

    utils.save_image(
        padded_psf,
        f"{APERTURE_NAMES[idx]}_psf",
        THEORETICAL_PSF_FOLDER,
        "gray",
    )

    return padded_psf

def crop_from_brightest(img: np.ndarray, crop_size: int) -> np.ndarray:
    """Crop around the brightest blob of pixels after blurring, clamping the crop to image edges."""

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

def get_contrast_per_angle_at_radius(
    polar: np.ndarray,
    angles: np.ndarray,
    radius: int
) -> np.ndarray:
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
    img_pad: np.ndarray = pad_image(theoretical_psf, (FFT_SIZE, FFT_SIZE))
    fft: np.ndarray = get_fft_db(img_pad)
    ffts.append(fft)

    # Line trace
    radius, mean_trace, min_trace = radial_line_trace(fft, (FFT_SIZE, FFT_SIZE))

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
        img_pad: np.ndarray = pad_image(img_crop, (FFT_SIZE, FFT_SIZE))

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

    img_padded = pad_image(img, (STAR_PAD_SIZE, STAR_PAD_SIZE), 32)

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
    plt.close(fig)

    # --------------------
    # Calculate MTF
    # --------------------

    N: int = 64 # pylint: disable=invalid-name
    r = np.arange(radius) # Radii
    frequency = N / (2 * np.pi * r)
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
    plt.close(fig)

def deconvole(idx: int) -> None:
    """Part 8, Deconvolve using calculated PSF"""

    raw_star_path: str = f"Lab3/mtf/mtf_{APERTURE_NAMES[idx]}.dng"
    raw_star: np.ndarray =  utils.read_raw(raw_star_path)
    star: np.ndarray = process_raw(raw_star)

    raw_psf_path: str = f"{APERTURE_FOLDER_NAMES[idx]}/0.dng"
    raw_psf: np.ndarray =  utils.read_raw(raw_psf_path)
    psf: np.ndarray = process_raw(raw_psf)
    psf_crop: np.ndarray = crop_from_brightest(psf, PSF_SIZE)
    psf_padded: np.ndarray =  pad_image(psf_crop, star.shape)
    # Normalise PSF so that its total energy is 1
    psf_norm: np.ndarray = psf_padded / np.sum(psf_padded)

    H: np.ndarray = np.fft.fft2(np.fft.fftshift(psf_norm)) # pylint: disable=invalid-name
    G = np.fft.fft2(star) # pylint: disable=invalid-name

    # --------------------
    # Wiener
    # --------------------
    K_vals: np.ndarray = np.logspace(-6, np.log10(5e-1), 30) # pylint: disable=invalid-name
    laplacian_vars: list[float] = []

    for K in K_vals: # pylint: disable=invalid-name
        W = np.conj(H) / (np.abs(H)**2 + K) # pylint: disable=invalid-name
        F_hat = W * G # pylint: disable=invalid-name

        X_hat = np.real(np.fft.ifft2(F_hat)) # pylint: disable=invalid-name

        # Variance of Laplacian
        laplacian = cv2.Laplacian(X_hat, cv2.CV_64F) # pylint: disable=no-member
        variance = laplacian.var()

        laplacian_vars.append(variance)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.semilogx(K_vals, laplacian_vars, 'o-')
    ax.set_xlabel("K")
    ax.set_ylabel("Variance of Laplacian")
    ax.set_title("Reconstruction quality vs Wiener regularisation")
    ax.grid()

    fig.savefig(
        f"{NATURAL_FOLDERS[idx]}_laplace.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    best_K = K_vals[np.argmax(laplacian_vars)] # pylint: disable=invalid-name

    # Create "best"
    W = np.conj(H) / (np.abs(H)**2 + best_K) # pylint: disable=invalid-name
    F_hat = W * G                            # pylint: disable=invalid-name

    X_hat = np.real(np.fft.ifft2(F_hat))     # pylint: disable=invalid-name

    X_hat = utils.rotate_image(X_hat)        # pylint: disable=invalid-name

    utils.save_image(X_hat, f"{NATURAL_FOLDERS[idx]}_best_star", "", "gray")

    # Create less-distorted
    alt_K = 5e-2                            # pylint: disable=invalid-name
    W = np.conj(H) / (np.abs(H)**2 + alt_K) # pylint: disable=invalid-name
    F_hat = W * G                           # pylint: disable=invalid-name

    X_hat = np.real(np.fft.ifft2(F_hat))    # pylint: disable=invalid-name

    X_hat = utils.rotate_image(X_hat)       # pylint: disable=invalid-name

    utils.save_image(X_hat, f"{NATURAL_FOLDERS[idx]}_alt_star", "", "gray")

    # --------------------
    # Natural Image
    # --------------------

    for i in range(3):
        raw_path: str = f"{NATURAL_FOLDERS[idx]}/{i}.dng"
        raw: np.ndarray =  utils.read_raw(raw_path)
        img: np.ndarray = process_raw(raw)


        W = np.conj(H) / (np.abs(H)**2 + alt_K) # pylint: disable=invalid-name
        G = np.fft.fft2(img)                    # pylint: disable=invalid-name
        F_hat = W * G                           # pylint: disable=invalid-name
        X_hat = np.real(np.fft.ifft2(F_hat))    # pylint: disable=invalid-name

        X_hat = utils.rotate_image(X_hat)       # pylint: disable=invalid-name
        img = utils.rotate_image(img)

        utils.save_image(img, f"{i}_original", f"{NATURAL_FOLDERS[idx]}_wiener", cmap="gray")
        utils.save_image(X_hat, f"{i}_deconvoluted", f"{NATURAL_FOLDERS[idx]}_wiener", cmap="gray")
    # --------------------
    # Gold Standard Load
    # --------------------

    # Only proceed for Levin or Nayar
    if idx == NAYAR_EQUIVALENT:
        return

    gold_standard_list: list[np.ndarray] = []
    first_image: np.ndarray

    for i in range(GOLD_STANDARD_COUNT):
        raw_path: str = f"{NATURAL_FOLDERS[idx]}/cap{i+1}.dng"
        raw: np.ndarray =  utils.read_raw(raw_path)
        img: np.ndarray = process_raw(raw)

        gold_standard_list.append(img)
        if i == 0:
            first_image = img

    gold_standard_stack: np.ndarray = np.stack(gold_standard_list)

    gold_standard = np.mean(gold_standard_stack, axis=0)

    # --------------------
    # First frame deconvolve
    # --------------------

    G = np.fft.fft2(first_image)            # pylint: disable=invalid-name

    # Standard on first frame
    F_hat = G / H                           # pylint: disable=invalid-name
    X_hat = np.real(np.fft.ifft2(F_hat))    # pylint: disable=invalid-name

    X_hat = utils.rotate_image(X_hat)       # pylint: disable=invalid-name

    utils.save_image(
        X_hat,
        f"Lab3/{APERTURE_NAMES[idx]}_first_frame_standard",
        "",
        cmap="gray"
    )

    # Weiner on first frame
    W = np.conj(H) / (np.abs(H)**2 + alt_K) # pylint: disable=invalid-name
    F_hat = W * G                           # pylint: disable=invalid-name
    X_hat = np.real(np.fft.ifft2(F_hat))    # pylint: disable=invalid-name

    X_hat = utils.rotate_image(X_hat)       # pylint: disable=invalid-name

    utils.save_image(
        X_hat,
        f"Lab3/{APERTURE_NAMES[idx]}_first_frame_weiner",
        "",
        cmap="gray"
    )

    # --------------------
    # Gold Standard deconvolve
    # --------------------

    G = np.fft.fft2(gold_standard)          # pylint: disable=invalid-name

    # Standard on gold standard
    F_hat = G / H                           # pylint: disable=invalid-name
    X_hat = np.real(np.fft.ifft2(F_hat))    # pylint: disable=invalid-name

    X_hat = utils.rotate_image(X_hat)       # pylint: disable=invalid-name

    utils.save_image(
        X_hat,
        f"Lab3/{APERTURE_NAMES[idx]}_gold_standard_standard",
        "",
        cmap="gray"
    )

    # Weiner on gold standard
    W = np.conj(H) / (np.abs(H)**2 + alt_K) # pylint: disable=invalid-name
    F_hat = W * G                           # pylint: disable=invalid-name
    X_hat = np.real(np.fft.ifft2(F_hat))    # pylint: disable=invalid-name

    X_hat = utils.rotate_image(X_hat)       # pylint: disable=invalid-name

    utils.save_image(
        X_hat,
        f"Lab3/{APERTURE_NAMES[idx]}_gold_standard_weiner",
        "",
        cmap="gray"
    )

def multiple_depths() -> None:
    """Part 9"""
    K = 5e-2 # pylint: disable=invalid-name
    raw_path: str = f"{P9_RAW_PATH}/scene_cap.dng"
    raw: np.ndarray =  utils.read_raw(raw_path)
    scene_img: np.ndarray = process_raw(raw)

    G = np.fft.fft2(scene_img) # pylint: disable=invalid-name

    for i in range(-2, 3):
        raw_path: str = f"{P9_RAW_PATH}/{i}.dng"
        raw: np.ndarray =  utils.read_raw(raw_path)
        psf: np.ndarray = process_raw(raw)

        psf_crop = crop_from_brightest(psf, PSF_SIZE_2)
        fft: np.ndarray = get_fft(psf_crop)

        psf_padded: np.ndarray =  pad_image(psf_crop, scene_img.shape)
        psf_norm: np.ndarray = psf_padded / np.sum(psf_padded)
        H: np.ndarray = np.fft.fft2(np.fft.fftshift(psf_norm)) # pylint: disable=invalid-name

        W = np.conj(H) / (np.abs(H)**2 + K)  # pylint: disable=invalid-name
        F_hat = W * G                        # pylint: disable=invalid-name
        X_hat = np.real(np.fft.ifft2(F_hat)) # pylint: disable=invalid-name

        X_hat = utils.rotate_image(X_hat)

        utils.save_image(psf_crop, f"{i}", P9_PNG_PATH, "gray")
        utils.save_image(fft, f"{i}", P9_FFT_PATH, FFT_CMAP)
        utils.save_image(X_hat, f"{i}", P9_DECONV_PATH, "gray")

    scene_img = utils.rotate_image(scene_img)

    utils.save_image(scene_img, "natural_image", P9_PNG_PATH, "gray")

def angled_aperture() -> None:
    """Bonus 1"""

    # Natural scene
    raw_path: str = f"{BONUS_RAW_PATH}/nat.dng"
    raw: np.ndarray =  utils.read_raw(raw_path)
    nat: np.ndarray = process_raw(raw)

    G = np.fft.fft2(nat)          # pylint: disable=invalid-name

    # PSF capture
    raw_path: str = f"{BONUS_RAW_PATH}/psf.dng"
    raw: np.ndarray =  utils.read_raw(raw_path)
    psf: np.ndarray = process_raw(raw)

    psf_crop = crop_from_brightest(psf, PSF_SIZE)
    fft: np.ndarray = get_fft(psf_crop)

    psf_padded: np.ndarray =  pad_image(psf_crop, nat.shape)
    psf_norm: np.ndarray = psf_padded / np.sum(psf_padded)
    K = 5e-2                                               # pylint: disable=invalid-name
    H: np.ndarray = np.fft.fft2(np.fft.fftshift(psf_norm)) # pylint: disable=invalid-name
    W = np.conj(H) / (np.abs(H)**2 + K)                    # pylint: disable=invalid-name
    F_hat = W * G                                          # pylint: disable=invalid-name
    X_hat = np.real(np.fft.ifft2(F_hat))                   # pylint: disable=invalid-name

    # Rotate images
    nat = utils.rotate_image(nat)
    X_hat = utils.rotate_image(X_hat)

    utils.save_image(nat, "nat", BONUS_PNG_PATH, cmap="gray")
    utils.save_image(psf_crop, "psf", BONUS_PNG_PATH, cmap="gray")
    utils.save_image(fft, "fft", BONUS_PNG_PATH, cmap=FFT_CMAP)
    utils.save_image(X_hat, "dec", BONUS_PNG_PATH, cmap="gray")

def straight_aperture() -> None:
    """Bonus 1, ok I stopped caring I just need it to work now"""

    # Natural scene
    raw_path: str = "Lab3/nat_levin/0.dng"
    raw: np.ndarray =  utils.read_raw(raw_path)
    nat: np.ndarray = process_raw(raw)

    G = np.fft.fft2(nat) # pylint: disable=invalid-name

    # PSF capture
    raw_path: str = "Lab3/psf_levin/0.dng"
    raw: np.ndarray =  utils.read_raw(raw_path)
    psf: np.ndarray = process_raw(raw)

    psf_crop = crop_from_brightest(psf, PSF_SIZE)
    fft: np.ndarray = get_fft(psf_crop)

    psf_padded: np.ndarray =  pad_image(psf_crop, nat.shape)
    psf_norm: np.ndarray = psf_padded / np.sum(psf_padded)
    K = 5e-2                                               # pylint: disable=invalid-name
    H: np.ndarray = np.fft.fft2(np.fft.fftshift(psf_norm)) # pylint: disable=invalid-name
    W = np.conj(H) / (np.abs(H)**2 + K)                    # pylint: disable=invalid-name
    F_hat = W * G                                          # pylint: disable=invalid-name
    X_hat = np.real(np.fft.ifft2(F_hat))                   # pylint: disable=invalid-name

    # Rotate images
    nat = utils.rotate_image(nat)
    X_hat = utils.rotate_image(X_hat)

    utils.save_image(nat, "nat_straight", BONUS_PNG_PATH, cmap="gray")
    utils.save_image(psf_crop, "psf_straight", BONUS_PNG_PATH, cmap="gray")
    utils.save_image(fft, "fft_straight", BONUS_PNG_PATH, cmap=FFT_CMAP)
    utils.save_image(X_hat, "dec_straight", BONUS_PNG_PATH, cmap="gray")

def chromatics_green_all() -> np.ndarray:
    """Bonus 2"""

    # Natural scene
    raw_path: str = "Lab3/nat_levin/0.dng"
    raw: np.ndarray =  utils.read_raw(raw_path)
    nat_color: np.ndarray = process_raw(raw, greyscale=False)
    nat_min = nat_color.min()
    nat_max = nat_color.max()
    nat_color = (nat_color - nat_min) / (nat_max - nat_min)

    raw_path: str = "Lab3/psf_levin/0.dng"
    raw: np.ndarray =  utils.read_raw(raw_path)
    psf: np.ndarray = process_raw(raw)
    psf_crop = crop_from_brightest(psf, PSF_SIZE_2)
    psf_padded: np.ndarray =  pad_image(psf_crop, nat_color[:,:,0].shape)
    psf_norm: np.ndarray = psf_padded / np.sum(psf_padded)
    K = 5e-2                                               # pylint: disable=invalid-name
    H: np.ndarray = np.fft.fft2(np.fft.fftshift(psf_norm)) # pylint: disable=invalid-name

    channels: list[np.ndarray] = []

    for i in range(3):
        nat = nat_color[:,:,i]

        G = np.fft.fft2(nat) # pylint: disable=invalid-name

        W = np.conj(H) / (np.abs(H)**2 + K)                    # pylint: disable=invalid-name
        F_hat = W * G                                          # pylint: disable=invalid-name
        X_hat = np.real(np.fft.ifft2(F_hat))                   # pylint: disable=invalid-name

        channels.append(X_hat)

    # Merge into rgb
    rgb = np.stack(channels, axis=2)
    rgb = np.clip(rgb / rgb.max(), 0.0, 1.0)

    # Rotate images 
    rgb = utils.rotate_image(rgb)

    plt.imsave(f"{BONUS_PNG_PATH}/dec_color_bad.png", rgb)

    return rgb

def chromatics_per_channel() -> np.ndarray:
    """Bonus 2"""

    # Natural scene
    raw_path: str = "Lab3/nat_levin/0.dng"
    raw: np.ndarray =  utils.read_raw(raw_path)
    nat_color: np.ndarray = process_raw(raw, greyscale=False)
    nat_min = nat_color.min()
    nat_max = nat_color.max()
    nat_color = (nat_color - nat_min) / (nat_max - nat_min)

    raw_path: str = "Lab3/psf_levin/0.dng"
    raw: np.ndarray =  utils.read_raw(raw_path)
    psf_color: np.ndarray = process_raw(raw, greyscale=False)

    K = 5e-2 # pylint: disable=invalid-name

    channels: list[np.ndarray] = []

    for i, color_name in enumerate(["red","green","blue"]):
        nat = nat_color[:,:,i]
        psf = psf_color[:,:,i]

        G = np.fft.fft2(nat) # pylint: disable=invalid-name

        # PSF capture

        psf_crop = crop_from_brightest(psf, PSF_SIZE_2)
        fft: np.ndarray = get_fft(psf_crop)

        psf_padded: np.ndarray =  pad_image(psf_crop, nat.shape)
        psf_norm: np.ndarray = psf_padded / np.sum(psf_padded)
        H: np.ndarray = np.fft.fft2(np.fft.fftshift(psf_norm)) # pylint: disable=invalid-name
        W = np.conj(H) / (np.abs(H)**2 + K)                    # pylint: disable=invalid-name
        F_hat = W * G                                          # pylint: disable=invalid-name
        X_hat = np.real(np.fft.ifft2(F_hat))                   # pylint: disable=invalid-name

        channels.append(X_hat)

        utils.save_image(psf_crop, f"psf_{color_name}", BONUS_PNG_PATH, cmap=MONOCHROME_CMAPS[i])
        utils.save_image(fft, f"fft_{color_name}", BONUS_PNG_PATH, cmap=FFT_CMAP)

    # Merge into rgb
    rgb = np.stack(channels, axis=2)
    rgb = np.clip(rgb / rgb.max(), 0.0, 1.0)
    
    # Rotate images 
    rgb = utils.rotate_image(rgb)
    nat_color = utils.rotate_image(nat_color)

    plt.imsave(f"{BONUS_PNG_PATH}/dec_color.png", rgb)
    plt.imsave(f"{BONUS_PNG_PATH}/nat_color.png", nat_color)

    return rgb
def show_diff(rgb: np.ndarray, green: np.ndarray) -> None:
    """Display the difference between the two methods."""

    diff = np.mean(np.abs(green - rgb), axis=2)

    # Exclude 64-pixel border when finding maximum difference
    margin = 64
    search_diff = diff.copy()
    search_diff[:margin, :] = -np.inf
    search_diff[-margin:, :] = -np.inf
    search_diff[:, :margin] = -np.inf
    search_diff[:, -margin:] = -np.inf

    # Find largest difference within the valid region
    y, x = np.unravel_index(np.argmax(search_diff), search_diff.shape) # pylint: disable=unbalanced-tuple-unpacking

    print("Maximum difference:", diff[y, x])
    print("Location:", x, y)

    crop_size = 256

    x1 = max(0, x - crop_size // 2)
    x2 = min(diff.shape[1], x + crop_size // 2)

    y1 = max(0, y - crop_size // 2)
    y2 = min(diff.shape[0], y + crop_size // 2)

    crop_green = green[y1:y2, x1:x2]
    crop_rgb = rgb[y1:y2, x1:x2]
    crop_diff = diff[y1:y2, x1:x2]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(crop_green)
    ax[0].set_title("Green PSF for all channels")
    ax[0].axis("off")

    ax[1].imshow(crop_rgb)
    ax[1].set_title("Individual PSFs for each channel")
    ax[1].axis("off")

    ax[2].imshow(crop_diff, cmap=FFT_CMAP)
    ax[2].set_title("Absolute difference")
    ax[2].axis("off")

    fig.savefig(
        f"{BONUS_PNG_PATH}/dec_color_compare.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

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
    deconvole(LEVIN)
    deconvole(NAYAR)
    deconvole(NAYAR_EQUIVALENT)

    # Part 9
    # multiple_depths()

    # Bonus 1
    # angled_aperture()
    # straight_aperture()

    # Bonus 2
    # green_dec = chromatics_per_channel()
    # rgb_dec = chromatics_green_all()
    # show_diff(rgb_dec, green_dec)
