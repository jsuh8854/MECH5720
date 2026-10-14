"""
Lab 4
"""

import os
# import cv2
import utils
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve, windows
from scipy.ndimage import median_filter, gaussian_filter
from ADMM import reconstruct_image, downsample, gradient


# 24 mm diameter

# 39 mm long tube
# 18 mm short tube

# ===================================
# Paths
# ===================================

PART_1_FOLDER: str = "part1"
PART_2_FOLDER: str = "part2"
PART_3_FOLDER: str = "part3"

PART_3_IMAGE_COUNT: int = 8

ADMM_SNAPSHOTS = (5, 20, 100, 300, 600)

PART_1_OUT: str = "part1_out"
PART_2_OUT: str = "part2_out"
PART_3_OUT: str = "part3_out"
PART_4_OUT: str = "part4_out"
PART_5_OUT: str = "part5_out"

# ===================================
# Functions yippie
# ===================================

def find_hot_pixels(dark: np.ndarray):
    """Find hot pixels in a dark frame."""
    dark = dark.astype(np.float32)

    n_sigma: float = 4.0 # Factor a pixel needs to exceed to qualify as a hot pixel.
    size: int = 9 # Box size of neighbours to check against
    zoom_size: int = 96 # Crop view size in px

    local = median_filter(dark, size=size)
    resid = dark - local

    # Robust noise estimate (MAD -> sigma), floored at one ADC step
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    sigma = max(sigma, 1.0 / utils.MAX_PIXEL)
    thresh = n_sigma * sigma

    mask = resid > thresh
    rows, cols = np.nonzero(mask)
    excess = resid[rows, cols]
    order = np.argsort(excess)[::-1]
    coords = np.column_stack([rows[order], cols[order], excess[order]])

    # Histogram of residuals with threshold shown
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(resid.ravel() * utils.MAX_PIXEL, bins=200, log=True)
    ax.axvline(thresh * utils.MAX_PIXEL, color="r", ls="--", label=f"{n_sigma} sigma")
    ax.set_xlabel("Pixel minus local median (DN)")
    ax.set_ylabel("Count")
    ax.set_title("Dark frame residuals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{PART_3_OUT}/hot_pixel_hist.png")
    plt.close(fig)

    # Annotated dark frame circling hot pixels
    fig, ax = plt.subplots(figsize=(8, 5))
    
    lo, hi = np.percentile(dark, [1, 99.5])
    ax.imshow(dark, cmap="gray", vmin=lo, vmax=hi)
    ax.scatter(cols, rows, s=60, facecolors="none", edgecolors="r", linewidths=0.5)
    ax.set_title(f"Hot pixels: {len(coords)} found")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{PART_3_OUT}/hot_pixel_map.png", dpi=200)
    plt.close(fig)

    # Zoomed top left corner, annotated as a separate plot
    crop = dark[:zoom_size, :zoom_size]
    in_crop = (rows < zoom_size) & (cols < zoom_size)

    fig, ax = plt.subplots(figsize=(6, 6))

    zoom_lo, zoom_hi = np.percentile(crop, [1, 99.5])
    ax.imshow(crop, cmap="gray", vmin=zoom_lo, vmax=zoom_hi, interpolation="nearest")
    ax.scatter(
        cols[in_crop], rows[in_crop],
        s=250, facecolors="none", edgecolors="r", linewidths=1.0,
    )
    ax.set_title(f"Top left {zoom_size}x{zoom_size}: {int(in_crop.sum())} hot pixels")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{PART_3_OUT}/hot_pixel_zoom.png", dpi=200)
    plt.close(fig)

def prep(img, factor=1/4):
    h, w = img.shape
    img = img[:h - h % 4, :w - w % 4]          # dims divisible by 4
    return downsample(img, factor)

def taper(img, a=0.1):
    wy = windows.tukey(img.shape[0], a); wx = windows.tukey(img.shape[1], a)
    return img * np.outer(wy, wx)

def crop_and_pad_psf(psf: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    """Centres the psf and removes unwanted signals outside the PSF range."""
    height, width = psf.shape

    if size <= 0:
        raise ValueError("size must be positive")

    # Define crop bounds around (x, y).
    half = size // 2
    x_start = x - half
    y_start = y - half
    x_end = x_start + size
    y_end = y_start + size

    # Crop while handling image boundaries.
    crop_x_start = max(0, x_start)
    crop_y_start = max(0, y_start)
    crop_x_end = min(width, x_end)
    crop_y_end = min(height, y_end)

    cropped = psf[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    # Pad the crop if it extends beyond the original image.
    pad_left = crop_x_start - x_start
    pad_top = crop_y_start - y_start
    pad_right = x_end - crop_x_end
    pad_bottom = y_end - crop_y_end

    cropped = np.pad(
        cropped,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    # Place the crop at the centre of an array matching the original dimensions.
    result = np.zeros_like(psf)
    y_offset = (height - size) // 2
    x_offset = (width - size) // 2

    # Handle size larger than the original dimensions.
    src_y_start = max(0, -y_offset)
    src_x_start = max(0, -x_offset)
    dst_y_start = max(0, y_offset)
    dst_x_start = max(0, x_offset)

    copy_height = min(size, height)
    copy_width = min(size, width)

    result[
        dst_y_start:dst_y_start + copy_height,
        dst_x_start:dst_x_start + copy_width,
    ] = cropped[
        src_y_start:src_y_start + copy_height,
        src_x_start:src_x_start + copy_width,
    ]

    return result

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

def prepare_psf(psf: np.ndarray) -> np.ndarray:
    """Subtract background (no clipping), downsample, normalise."""
    border = np.concatenate([psf[:50].ravel(), psf[-50:].ravel()])
    psf_d = prep(psf - np.median(border))
    return psf_d / np.linalg.norm(psf_d)


def prepare_data(img: np.ndarray) -> np.ndarray:
    """Downsample and normalise a scene capture."""
    data_d = prep(img)
    return data_d / np.linalg.norm(data_d)


def run_and_save_admm(name, data_d, psf_used, tau, snapshots=ADMM_SNAPSHOTS):
    """Run ADMM once, save and show the snapshot series for this tau."""
    recons = reconstruct_image(psf_used, data_d, tau=tau, snapshots=snapshots)

    tau_str = f"{tau:.0e}".replace("-", "m").replace("+", "p")   # 1e-07 -> 1em07

    fig, axes = plt.subplots(
        1, len(snapshots), figsize=(4 * len(snapshots), 7), squeeze=False
    )
    for ax, n in zip(axes.flat, snapshots):
        recon = recons[n]
        ax.imshow(
            utils.rotate_image(recon), cmap="gray",
            vmin=0, vmax=np.percentile(recon, 99.5),
        )
        ax.set_title(f"{n} iterations")
        ax.axis("off")

    fig.suptitle(f"{name}: tau = {tau:.1e}")
    fig.tight_layout()
    fig.savefig(f"{PART_5_OUT}/admm_{name.lower()}_tau_{tau_str}.png")
    # plt.show()
    plt.close(fig)

    return recons

def tv(img):
    """Get total variation"""
    return np.abs(gradient(img)).sum()

def tau_effect(psf, star, iters=100):
    """Debugging what values of tau to sweep"""
    psf_d = prepare_psf(psf)
    data_d = prepare_data(star)

    ref = reconstruct_image(psf_d, data_d, tau=1e-30, snapshots=(iters,))[iters] # effectively no TV

    for tau in np.logspace(-16, 4, 11):
        r = reconstruct_image(psf_d, data_d, tau=tau, snapshots=(iters,))[iters]
        print(
            f"tau={tau:.0e}  "
            f"rel diff vs no-TV={np.linalg.norm(r - ref) / np.linalg.norm(ref):.2e}  "
            f"TV ratio={tv(r) / tv(ref):.3f}")

def wiener_deconvolve(psf_d, data_d, k_rel, taper_a=0.2):
    """Wiener deconvolution on a 2x padded grid, matching ADMM's conventions
    """
    h, w = data_d.shape

    # Remove pedestals (DC is the biggest edge-step source), soften the edges
    psf_z = taper(psf_d - psf_d.mean(), taper_a)
    dat_z = taper(data_d - data_d.mean(), taper_a)

    def pad(a):
        return np.pad(a, ((h // 2, h - h // 2), (w // 2, w - w // 2)))

    H = np.fft.fft2(np.fft.ifftshift(pad(psf_z)))   # pylint: disable=invalid-name
    G = np.fft.fft2(np.fft.ifftshift(pad(dat_z)))   # pylint: disable=invalid-name

    K = k_rel * np.mean(np.abs(H) ** 2)             # pylint: disable=invalid-name
    X = np.real(np.fft.ifft2(np.conj(H) * G / (np.abs(H) ** 2 + K)))  # pylint: disable=invalid-name
    X = np.fft.fftshift(X)

    return X[h // 2:h // 2 + h, w // 2:w // 2 + w]   # crop back to sensor size

# ===================================
# Sections
# ===================================

def part_4(psf, star, natural, best_k=None):
    """Part 4"""
    psf_d = prepare_psf(psf)

    k_rels=np.logspace(-3, 1.25, 18) # 4 per decade

    for name, scene in [("Star", star), ("Natural", natural)]:
        data_d = prepare_data(scene)

        n_cols = 6
        n_rows = int(np.ceil(len(k_rels) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(2 * n_cols, 3 * n_rows), squeeze=False)

        for ax, k in zip(axes.flat, k_rels):
            X = wiener_deconvolve(psf_d, data_d, k)  # pylint: disable=invalid-name
            shown = utils.rotate_image(X)
            lo, hi = np.percentile(X, [1, 99.5])

            k_str = f"{k:.1e}".replace("-", "m").replace("+", "p")
            plt.imsave(
                f"{PART_4_OUT}/wiener_{name.lower()}_k_{k_str}.png",
                shown, cmap="gray", vmin=lo, vmax=hi
            )

            ax.imshow(shown, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(f"K = {k:.1e} x mean|H|²")
            ax.axis("off")

        for ax in axes.flat[len(k_rels):]:
            ax.axis("off")

        fig.suptitle(f"{name}: Wiener deconvolution, K sweep")
        fig.tight_layout()
        fig.savefig(f"{PART_4_OUT}/wiener_{name.lower()}_sweep.png")
        plt.close(fig)

        if best_k is not None:
            X = wiener_deconvolve(psf_d, data_d, best_k)  # pylint: disable=invalid-name
            lo, hi = np.percentile(X, [1, 99.5])
            plt.imsave(
                f"{PART_4_OUT}/wiener_{name.lower()}_best.png",
                utils.rotate_image(X), cmap="gray", vmin=lo, vmax=hi
            )

def part_5_tau_sweep(
    psf: np.ndarray, 
    star: np.ndarray,
    taus=np.logspace(-12, -6, 11)
):
    """Sweep tau on the star target only."""

    psf_d = prepare_psf(psf)
    data_d = prepare_data(star)

    for tau in taus:
        run_and_save_admm("Star", data_d, psf_d, tau)


def part_5_best_tau(
    psf: np.ndarray,
    star: np.ndarray,
    natural: np.ndarray,
    best_tau: float
):

    """Rotated-PSF failure test (star) and natural image, at a chosen tau."""
    psf_d = prepare_psf(psf)
    psf_rot = psf_d[::-1, ::-1]                     # 180 degree rotation

    run_and_save_admm("Rotated", prepare_data(star), psf_rot, best_tau)
    run_and_save_admm("Natural", prepare_data(natural), psf_d, best_tau)

def part_1():
    """Part 1"""

    # Show cover close frame
    closed_frame = utils.read_and_process_raw(f"{PART_1_FOLDER}/blank.dng")
    plt.imsave(f"{PART_1_OUT}/blank_norm.png", closed_frame, cmap="gray")
    plt.imsave(f"{PART_1_OUT}/blank.png", closed_frame, cmap="gray", vmax=1)

    fig, ax = plt.subplots()
    for file_name in ["close", "sharp", "far"]:  # Add more filenames here
        print(f"Processing {file_name}")
        ambient = utils.read_and_process_raw(f"{PART_1_FOLDER}/ambient_{file_name}.dng")

        psf = utils.read_and_process_raw(f"{PART_1_FOLDER}/{file_name}.dng") - ambient

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

def part_2():
    """Part 2"""

    ambient = utils.read_and_process_raw(f"{PART_2_FOLDER}/ambient.dng").astype(np.float32)

    positions = []
    psf_stds = []
    shifts = []

    psf_0 = utils.read_and_process_raw(f"{PART_2_FOLDER}/0.dng").astype(np.float32) - ambient

    for i in range(-5, 6):
        psf = utils.read_and_process_raw(f"{PART_2_FOLDER}/{i}.dng").astype(np.float32) - ambient

        # get psf std and plot
        psf_std = np.std(psf)

        positions.append(i)
        psf_stds.append(psf_std)

        # Cross-correlate with on-axis PSF
        corr = fftconvolve(psf, psf_0[::-1, ::-1], mode="full")
        peak_x, _ = np.unravel_index(np.argmax(corr), corr.shape) # pylint: disable=unbalanced-tuple-unpacking

        shift_x = peak_x - (psf.shape[1] - 1)
        shifts.append(shift_x)

    plt.figure(figsize=(8, 5))
    plt.plot(positions, psf_stds, "o-")
    plt.xlabel("Lateral position")
    plt.ylabel("PSF standard deviation (pixel intensity)")
    plt.title("PSF Standard Deviation vs Lateral Position")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PART_2_OUT}/lateral_sweep.png")
    plt.close()

    # Fit line through origin using only on-screen positions (-4 to 4)
    fit_positions = np.array(positions[1:-1], dtype=float)
    fit_shifts = np.array(shifts[1:-1], dtype=float)

    # Plot correlation shift against lateral position
    plt.figure(figsize=(8, 5))

    plt.plot(positions, shifts, "o-")

    slope = np.sum(fit_positions * fit_shifts) / np.sum(fit_positions**2)
    plt.plot(fit_positions, slope * fit_positions, "k:", label=f"Fit (slope={slope:.2f})")
    plt.legend()

    plt.xlabel("Lateral position")
    plt.ylabel("Horizontal PSF shift (pixels)")
    plt.title("PSF Shift vs Lateral Position")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PART_2_OUT}/psf_shifts.png")
    plt.close()

    print(f"Shift slope: {slope:.4f} pixels per position unit")

def part_3():
    """Part 3"""

    fixed_pattern = np.mean([
        utils.read_and_process_raw(
            f"{PART_3_FOLDER}/fixed_pattern/fixed_pattern_{i}.dng"
        ).astype(np.float32)
        for i in range(PART_3_IMAGE_COUNT)
    ], axis=0)

    ambient = np.mean([
        utils.read_and_process_raw(
            f"{PART_3_FOLDER}/ambient_{i}.dng"
        ).astype(np.float32)
        for i in range(PART_3_IMAGE_COUNT)
    ], axis=0)

    psf = np.mean([
        utils.read_and_process_raw(
            f"{PART_3_FOLDER}/psf_{i}.dng"
        ).astype(np.float32)
        for i in range(PART_3_IMAGE_COUNT)
    ], axis=0) - ambient

    star = np.mean([
        utils.read_and_process_raw(
            f"{PART_3_FOLDER}/star_{i}.dng"
        ).astype(np.float32)
        for i in range(PART_3_IMAGE_COUNT)
    ], axis=0) - fixed_pattern

    natural = np.mean([
        utils.read_and_process_raw(
            f"{PART_3_FOLDER}/natural_{i}.dng"
        ).astype(np.float32)
        for i in range(PART_3_IMAGE_COUNT)
    ], axis=0) - fixed_pattern

    psf_crop = crop_and_pad_psf(psf, 1010, 540, 1080) # (psf, 261, 269, 450)
    psf_crop /= psf_crop.sum()

    lo, hi = np.percentile(psf_crop, [1, 99.8])
    plt.imsave(
        f"{PART_3_OUT}/psf.png",
        utils.rotate_image(psf_crop), vmin=lo, vmax=hi, cmap="gray"
    )

    lo, hi = np.percentile(fixed_pattern, [1, 99.8])
    plt.imsave(
        f"{PART_3_OUT}/fixed_pattern.png",
        utils.rotate_image(fixed_pattern), vmin=lo, vmax=hi, cmap="gray"
    )

    lo, hi = np.percentile(star, [1, 99.8])
    plt.imsave(
        f"{PART_3_OUT}/star_raw.png",
        utils.rotate_image(star), vmin=lo, vmax=hi, cmap="gray"
    )

    lo, hi = np.percentile(natural, [1, 99.8])
    plt.imsave(
        f"{PART_3_OUT}/natural_raw.png",
        utils.rotate_image(natural), vmin=lo, vmax=hi, cmap="gray"
    )

    find_hot_pixels(fixed_pattern)


    # --------------------
    # Weiner
    # --------------------

    # part_4(psf_crop, star, natural) # first pass: look at the sweep

    # --------------------
    # ADMM
    # --------------------

    # tau_effect(psf, star)

    # part_5_tau_sweep(psf, star)

    # best_tau = 1e-8
    # part_5_best_tau(psf, star, natural, best_tau)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # So it doesn't cry
    os.makedirs(PART_1_OUT, exist_ok=True)
    os.makedirs(PART_2_OUT, exist_ok=True)
    os.makedirs(PART_3_OUT, exist_ok=True)
    os.makedirs(PART_4_OUT, exist_ok=True)
    os.makedirs(PART_5_OUT, exist_ok=True)

    # part_1()
    # part_2()
    # test = utils.read_and_process_raw(f"{PART_3_FOLDER}/psf_0.dng")
    # print(np.max(test))

    part_3()
