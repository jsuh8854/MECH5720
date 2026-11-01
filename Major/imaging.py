"""Imaging code for Background Orientated Schlieren Imaging"""
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
import cv2
import utils
import numpy as np
from matplotlib import pyplot as plt

IMAGE_DIRECTORY: str = "data"
OUT_DIRECTORY: str = "out"
METHOD: str = "temp"                   # "tvl1", "wdef", "both" or "temp"
# Temperature estimate (METHOD = "temp")

# ----------------------------------------
# Temperature analysis constants
# ----------------------------------------

AMBIENT_TEMP_C: float = 30.0           # ambient temperature
MAX_TEMP_C:     float = 280.0          # pylint:disable=no-member # hot plate maximum, from the manual
# Positions [start, stop) in the *sorted* heat image list, one entry per hot-plate setting.
# The first group is the calibration setting that is assumed to be at MAX_TEMP_C.
HEAT_GROUPS: dict = {
    "max": (0, 4),
    "medium": (4, 8),
    "low": (8, 12)
}
TEMP_METHOD: str = "wdef"              # solver used for the temperature analysis: "wdef" or "tvl1"
STD_ROI = None                         # None = whole image, or (y0, y1, x0, x1) to measure inside
# Signal ~ (dT)^p. p = 1 is linear. For turbulent flow, Nu prop Ra^(1/3), exponent approx 1.33
SIGNAL_EXPONENT: float = 1.33
# Other "known" point from manual
KNOWN_TEMPS_C: dict = {"low": 150.0}
NOISE_TEST_FRAMES: int = 4 # first N ambient frames are the noise-floor test frames;
                           # they are compared against the mean of the remaining ambient frames

@dataclass
class TVL1Settings:
    """Settings for the TV-L1 solver."""
    l: float = 0.05 # Lower for smoother, higher for higher trust in data
    """Lambda"""
    tau: float = 0.25
    scales_number: int = 5
    warpings_number: int = 5
    epsilon: float = 0.01
    inner_iterations: int = 30
    outer_iterations: int = 10
    blur_sigma: float = 2.0  # Gaussian pre-blur applied to BOTH images (0 = disabled)

@dataclass
class WDefSettings:
    """Settings for the Window Deformation Solver"""
    window_sizes: tuple = (64, 32, 16)  # one pass per entry, coarse -> fine
    overlap: float = 0.5                # fractional window overlap
    iterations: int = 2                 # deformation iterations per pass
    max_shift_frac: float = 0.4         # peak search limited to +-frac*window
    median_thresh: float = 2.0          # normalised median test threshold
    min_std: float = 1e-3               # skip windows with almost no texture


# ---------------------------------------------------------------------------
# TV-L1
# ---------------------------------------------------------------------------

def _create_tvl1(settings: TVL1Settings):
    """Create and initialise tv-l1 from provided settings."""
    # pylint:disable=no-member
    tv_l1 = cv2.optflow.createOptFlow_DualTVL1()

    tv_l1.setLambda(settings.l)
    tv_l1.setTau(settings.tau)
    tv_l1.setScalesNumber(settings.scales_number)
    tv_l1.setWarpingsNumber(settings.warpings_number)
    tv_l1.setEpsilon(settings.epsilon)
    tv_l1.setInnerIterations(settings.inner_iterations)
    tv_l1.setOuterIterations(settings.outer_iterations)

    return tv_l1

def perform_tv_l1(
    ambient_images: list[np.ndarray],
    heat_images: list[np.ndarray],
    tv_l1_settings: TVL1Settings
) -> list[np.ndarray]:
    """
    Takes in stacks of ambient and heat source images and returns a list of displacement images.
    
    Each displacement image has shape (H, W, 2) having displacement data of (dx, dy) in pixels.
    """
    # pylint:disable=no-member

    # Average ambient to reduce noise
    ref = np.mean(ambient_images, axis=0)

    # TV-L1's data/smoothness balance depends on the intensity range, so scale both
    # images to [0, 1] with ONE shared factor (applied identically to every image).
    scale = 1.0 / max(float(ref.max()), 1e-6)
    ref = ref * scale

    # Optional pre-blur, always applied to both images or neither.
    if tv_l1_settings.blur_sigma > 0:
        ref = cv2.GaussianBlur(ref, (0, 0), tv_l1_settings.blur_sigma)

    # Create the solver from settings
    tv_l1 = _create_tvl1(tv_l1_settings)

    results: list[np.ndarray] = [] # To return

    for i, img in enumerate(heat_images):
        print(f"Processing image {i+1}...")

        cur = img * scale

        if tv_l1_settings.blur_sigma > 0:
            cur = cv2.GaussianBlur(cur, (0, 0), tv_l1_settings.blur_sigma)

        # Match overall brightness to the reference (lamp drift, exposure)
        cur = cur * (ref.mean() / max(cur.mean(), 1e-6))

        flow = tv_l1.calc(ref, cur, None)  # (H, W, 2) with dx, dy

        # Remove global rigid shift from vibration/drift
        flow[..., 0] -= np.median(flow[..., 0])
        flow[..., 1] -= np.median(flow[..., 1])

        results.append(flow.astype(np.float32))

    print("Done!")

    return results

# ---------------------------------------------------------------------------
# Window-deformation cross-correlation
# ---------------------------------------------------------------------------

def _grid_axes(h: int, w: int, win: int, overlap: float):
    """Window start coordinates for one pass."""
    step = max(1, int(round(win * (1 - overlap))))
    y0 = np.arange(0, h - win + 1, step)
    x0 = np.arange(0, w - win + 1, step)
    return step, y0, x0

def _subpixel_peak(corr: np.ndarray, max_shift: int) -> np.ndarray:
    """corr: (n, w, w) fftshifted correlation. Returns (n, 2) residual (dx, dy)."""
    n, w, _ = corr.shape
    c0 = w // 2
    lo, hi = c0 - max_shift, c0 + max_shift + 1
    sub = corr[:, lo:hi, lo:hi]
    flat = sub.reshape(n, -1)
    idx = np.argmax(flat, axis=1)
    py, px = np.divmod(idx, sub.shape[2])
    ar = np.arange(n)

    # 3-point Gaussian fit on log of the (normalised, clipped) correlation
    peak = np.maximum(flat[ar, idx], 1e-12)
    cn = np.clip(corr / peak[:, None, None], 1e-3, None)
    gy = np.clip(py + lo, 1, w - 2)
    gx = np.clip(px + lo, 1, w - 2)
    ln = np.log(cn)
    c = ln[ar, gy, gx]
    dx_sub = (ln[ar, gy, gx - 1] - ln[ar, gy, gx + 1]) / (
        2 * (ln[ar, gy, gx - 1] - 2 * c + ln[ar, gy, gx + 1]) - 1e-12)
    dy_sub = (ln[ar, gy - 1, gx] - ln[ar, gy + 1, gx]) / (
        2 * (ln[ar, gy - 1, gx] - 2 * c + ln[ar, gy + 1, gx]) - 1e-12)
    dx_sub = np.clip(dx_sub, -1, 1)
    dy_sub = np.clip(dy_sub, -1, 1)

    return np.stack([px + lo - c0 + dx_sub, py + lo - c0 + dy_sub], axis=1)

def _correlate_grid(
    a: np.ndarray, b: np.ndarray,
    win: int,
    y0: np.ndarray, x0: np.ndarray,
    settings: WDefSettings
) -> np.ndarray:
    """Cross-correlate all windows; returns residual (ny, nx, 2), NaN where no texture."""
    out = np.full((len(y0), len(x0), 2), np.nan, np.float32)
    hann = np.hanning(win).astype(np.float32)
    taper = np.outer(hann, hann)
    max_shift = max(1, int(settings.max_shift_frac * win))

    for iy, y in enumerate(y0):  # one row of windows at a time to limit memory
        wa = np.stack([a[y:y + win, x:x + win] for x in x0])
        wb = np.stack([b[y:y + win, x:x + win] for x in x0])
        wa = wa - wa.mean(axis=(1, 2), keepdims=True)
        wb = wb - wb.mean(axis=(1, 2), keepdims=True)
        textured = ((wa.std(axis=(1, 2)) > settings.min_std)
                    & (wb.std(axis=(1, 2)) > settings.min_std))

        fa = np.fft.rfft2(wa * taper)
        fb = np.fft.rfft2(wb * taper)
        corr = np.fft.irfft2(np.conj(fa) * fb, s=(win, win))
        corr = np.fft.fftshift(corr, axes=(1, 2))

        res = _subpixel_peak(corr, max_shift)
        res[~textured] = np.nan
        out[iy] = res
    return out

def _validate_and_fill(field: np.ndarray, thresh: float) -> np.ndarray:
    """Normalised median test (Westerweel & Scarano); outliers/NaNs -> neighbour median."""
    ny, nx, _ = field.shape
    out = field.copy()
    pad = np.pad(field, ((1, 1), (1, 1), (0, 0)), mode="edge")
    for c in range(2):
        nb = np.stack([
            pad[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx, c]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)
        ])
        med = np.nanmedian(nb, axis=0)
        rm = np.nanmedian(np.abs(nb - med), axis=0)
        r = np.abs(field[..., c] - med) / (rm + 0.1)
        bad = np.isnan(field[..., c]) | (r > thresh)
        out[..., c] = np.where(bad, med, field[..., c])
    # anything still NaN (large untextured regions) is set to 0
    return np.nan_to_num(out, nan=0.0)

def _grid_to_dense(
    grid: np.ndarray,
    win: int,
    step: int,
    y0: np.ndarray,
    x0: np.ndarray,
    h: int,
    w: int
) -> np.ndarray:
    """Cubic-interpolate the vector grid to one vector per pixel."""
    # pylint:disable=no-member
    cy0 = y0[0] + (win - 1) / 2
    cx0 = x0[0] + (win - 1) / 2
    gy = np.clip((np.arange(h, dtype=np.float32) - cy0) / step, 0, len(y0) - 1)
    gx = np.clip((np.arange(w, dtype=np.float32) - cx0) / step, 0, len(x0) - 1)
    mapx, mapy = np.meshgrid(gx.astype(np.float32), gy.astype(np.float32))
    return cv2.remap(
        grid.astype(np.float32),
        mapx, mapy,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def window_deform_piv(ref: np.ndarray, cur: np.ndarray, settings: WDefSettings) -> np.ndarray:
    """Dense (H, W, 2) displacement (dx, dy) of cur relative to ref."""
    # pylint:disable=no-member, invalid-name
    h, w = ref.shape
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)

    disp = np.zeros((h, w, 2), np.float32)  # current dense estimate

    for win in settings.window_sizes:
        step, y0, x0 = _grid_axes(h, w, win, settings.overlap)
        for it in range(settings.iterations):
            print(f"  window {win}px, iteration {it + 1}/{settings.iterations}")
            # Symmetric deformation: both images are moved by half the estimate
            half = 0.5 * disp
            a = cv2.remap(
                ref, X - half[..., 0], Y - half[..., 1],
                cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            b = cv2.remap(
                cur, X + half[..., 0], Y + half[..., 1],
                cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

            resid = _correlate_grid(a, b, win, y0, x0, settings)
            resid = _validate_and_fill(resid, settings.median_thresh)
            disp = disp + _grid_to_dense(resid, win, step, y0, x0, h, w)

    return disp


def perform_window_deform(
    ambient_images: list[np.ndarray],
    heat_images: list[np.ndarray],
    wdef_settings: WDefSettings
) -> list[np.ndarray]:
    """
    Same inputs and outputs as perform_tv_l1, using window-deformation cross-correlation.

    Each displacement image has shape (H, W, 2) having displacement data of (dx, dy) in pixels.
    """
    # Average ambient to reduce noise
    ref = np.mean(ambient_images, axis=0)

    results: list[np.ndarray] = []  # To return

    for i, img in enumerate(heat_images):
        print(f"Processing image {i+1}...")

        # Match overall brightness to the reference (lamp drift, exposure)
        cur = img * (ref.mean() / max(img.mean(), 1e-6))

        flow = window_deform_piv(ref, cur, wdef_settings)  # (H, W, 2) with dx, dy

        # Remove global rigid shift from vibration/drift
        flow[..., 0] -= np.median(flow[..., 0])
        flow[..., 1] -= np.median(flow[..., 1])

        results.append(flow.astype(np.float32))

    print("Done!")

    return results

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _natural_key(path: Path):
    """Sort heat_2 before heat_10 (plain sorted() would put heat_10 before heat_2)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def load_images(
    directory: Path # pylint:disable=redefined-outer-name
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Load the ambient and heat image stacks (natural filename order) and the heat file names."""
    files = sorted(directory.iterdir(), key=_natural_key)
    ambient_files = [
        f for f in files
        if f.is_file() and f.name.startswith("ambient_")
    ]
    heat_files = [f for f in files if f.is_file() and f.name.startswith("heat_")]

    ambient = [utils.process_raw(str(f)) for f in ambient_files]
    heat = [utils.process_raw(str(f)) for f in heat_files]

    return ambient, heat, [f.name for f in heat_files]

# ---------------------------------------------------------------------------
# Plotting and analysis
# ---------------------------------------------------------------------------


def plot_displacements(
    heat_flow: np.ndarray, ambient_flow: np.ndarray,
    heat_image: np.ndarray, ambient_image: np.ndarray,
    method: str, save_path: Path,
    central_fraction: float = 0.99
) -> None:
    """
    4x2 plot: x displacement, y displacement and displacement magnitude for the
    heat image and the ambient noise floor, with the debayered source images underneath.

    The shared colour scales span only the middle `central_fraction` (default 99%) of all
    displacement values, so extreme outliers don't flatten the rest of the image.
    """
    img1 = heat_flow[:, :, 0]
    img2 = ambient_flow[:, :, 0]
    img3 = heat_flow[:, :, 1]
    img4 = ambient_flow[:, :, 1]
    mag_heat = np.hypot(heat_flow[:, :, 0], heat_flow[:, :, 1])
    mag_ambient = np.hypot(ambient_flow[:, :, 0], ambient_flow[:, :, 1])

    # Shared signed colour scale for dx/dy from the central fraction of all values
    tail = 100 * (1 - central_fraction) / 2
    all_vals = np.concatenate([img.ravel() for img in (img1, img2, img3, img4)])
    vmin, vmax = np.percentile(all_vals, [tail, 100 - tail])

    # Shared magnitude scale: 0 up to the upper percentile (magnitude is never negative)
    all_mag = np.concatenate([mag_heat.ravel(), mag_ambient.ravel()])
    mmax = np.percentile(all_mag, 100 - tail)

    # Shared grey scale for the two source images (1st to 99th percentile)
    both = np.concatenate([heat_image.ravel(), ambient_image.ravel()])
    gmin, gmax = np.percentile(both, [1, 99])

    # Third column is reserved for colorbars, so every row of image axes has the
    # same width and the bottom (source image) row lines up with the rows above.
    fig = plt.figure(figsize=(12, 20), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, width_ratios=[1, 1, 0.04])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(4)])
    cax_disp = fig.add_subplot(gs[0:2, 2])
    cax_mag = fig.add_subplot(gs[2, 2])

    im1 = axes[0][0].imshow(img1, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0][0].set_title("Heat (x displacement)")

    axes[0][1].imshow(img2, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0][1].set_title("Ambient Comparison (x displacement)")

    axes[1][0].imshow(img3, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1][0].set_title("Heat (y displacement)")

    axes[1][1].imshow(img4, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1][1].set_title("Ambient Comparison (y displacement)")

    im_mag = axes[2][0].imshow(mag_heat, cmap="inferno", vmin=0, vmax=mmax)
    axes[2][0].set_title("Heat (displacement magnitude)")

    axes[2][1].imshow(mag_ambient, cmap="inferno", vmin=0, vmax=mmax)
    axes[2][1].set_title("Ambient Comparison (displacement magnitude)")

    axes[3][0].imshow(heat_image, cmap="gray", vmin=gmin, vmax=gmax)
    axes[3][0].set_title("Heat (debayered)")

    axes[3][1].imshow(ambient_image, cmap="gray", vmin=gmin, vmax=gmax)
    axes[3][1].set_title("Ambient (debayered)")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(method)
    fig.colorbar(im1, cax=cax_disp, label="Displacement (px)")
    fig.colorbar(im_mag, cax=cax_mag, label="Magnitude (px)")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_collage(
    magnitudes: dict,
    heat_names: list[str],
    method: str,
    save_path: Path,
    image_shape: tuple[int, int],
    central_fraction: float = 0.95
) -> None:
    """
    Collage of displacement-magnitude images.
    One column per HEAT_GROUPS setting, one row per repeat within the setting.
    All panels share the pixel axes and one colour scale
    (0 up to the upper percentile of all panels pooled, so outliers don't wash everything out).

    magnitudes: {heat image index: 2D magnitude array, possibly downsampled}
    image_shape: (H, W) of the original images, so the axes stay in original pixel units.
    """
    h, w = image_shape
    ranges = {g: range(a, min(b, len(heat_names))) for g, (a, b) in HEAT_GROUPS.items()}
    ncols = len(ranges)
    nrows = max(len(r) for r in ranges.values())

    tail = 100 * (1 - central_fraction) / 2
    pooled = np.concatenate([m.ravel() for m in magnitudes.values()])
    vmax = float(np.percentile(pooled, 100 - tail))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.2 * nrows + 0.6),
        sharex=True, sharey=True,
        constrained_layout=True, squeeze=False
    )
    im = None
    for col, (gname, idx_range) in enumerate(ranges.items()):
        for row in range(nrows):
            ax = axes[row][col]
            if row < len(idx_range) and idx_range[row] in magnitudes:
                i = idx_range[row]
                im = ax.imshow(
                    magnitudes[i],
                    cmap="inferno", vmin=0, vmax=vmax,
                    extent=(0, w, h, 0), interpolation="nearest"
                )
                ax.set_title(f"{gname}: {Path(heat_names[i]).stem}", fontsize=9)
            else:
                ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("x (px)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (px)")

    fig.suptitle(f"{method}: displacement magnitude, shared colour scale")
    if im is not None:
        fig.colorbar(im, ax=axes, label="Displacement magnitude (px)", shrink=0.7)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyse_all(
    name: str, out_subdir: str, perform, settings,
    ambient: list[np.ndarray], heat: list[np.ndarray],
    heat_names: list[str] | None = None
) -> None:
    """Run `perform` on every heat image and save one plot per image.

    The ambient-vs-ambient noise floor is computed once and reused for all plots.
    Plots are saved in OUT_DIRECTORY/<out_subdir>/<out_subdir>_<index>.png, where the index
    follows the sorted order of the heat_* files.
    A collage of all heat images (one column per HEAT_GROUPS setting, shared axes and colour scale)
    is saved as <out_subdir>_collage.png.
    """
    # pylint:disable=no-member
    heat_names = heat_names or [f"heat_{i + 1}" for i in range(len(heat))]

    out_dir = Path(OUT_DIRECTORY) / out_subdir
    width = max(3, len(str(len(heat) - 1)))

    print(f"[{name}] computing ambient noise floor...")

    ambient_flow = perform(ambient, [ambient[0]], settings)[0]

    magnitudes = {}

    for i, heat_image in enumerate(heat):
        print(f"[{name}] heat image {i + 1}/{len(heat)}")

        flow = perform(ambient, [heat_image], settings)[0]

        plot_displacements(
            flow, ambient_flow,
            heat_image, ambient[0],
            name, out_dir / f"{out_subdir}_{i:0{width}d}.png"
        )

        mag = np.hypot(flow[..., 0], flow[..., 1]).astype(np.float32)
        magnitudes[i] = mag

    plot_collage(
        magnitudes,
        heat_names,
        name,
        out_dir / f"{out_subdir}_collage.png",
        heat[0].shape[:2]
    )

    print(f"[{name}] saved {len(heat)} plots and the collage to {out_dir}")

# ---------------------------------------------------------------------------
# Create the settings for the solvers
# ---------------------------------------------------------------------------

def make_tvl1_settings() -> TVL1Settings:
    """Settings used by the TV-L1 analyses."""
    return TVL1Settings(
        l=0.15,
        tau=0.25,
        scales_number=5,
        warpings_number=5,
        epsilon=0.01,
        inner_iterations=30,
        outer_iterations=10
    )

def make_wdef_settings() -> WDefSettings:
    """Settings used by the window-deformation analyses."""
    return WDefSettings(
        window_sizes=(64, 32, 16),
        overlap=0.5,
        iterations=8,
        max_shift_frac=0.4,
        median_thresh=2.0,
        min_std=1e-3
    )

# --------------------------------------------------------------------------- #
# Temperature estimate from the spread (std) of the displacement magnitude
# --------------------------------------------------------------------------- #

def magnitude_std(flow: np.ndarray) -> float:
    """Standard deviation of the displacement magnitude image (optionally inside STD_ROI)."""
    mag = np.hypot(flow[..., 0], flow[..., 1])
    if STD_ROI is not None:
        y0, y1, x0, x1 = STD_ROI
        mag = mag[y0:y1, x0:x1]
    return float(mag.std())

def temp_linear(r: np.ndarray) -> np.ndarray:
    """Signal proportional to (T - T_ambient)."""
    return AMBIENT_TEMP_C + (MAX_TEMP_C - AMBIENT_TEMP_C) * r

def temp_density(r: np.ndarray) -> np.ndarray:
    """Signal proportional to the air density deficit, 1 - T_ambient/T (ideal gas, Kelvin)."""
    ta = AMBIENT_TEMP_C + 273.15
    tm = MAX_TEMP_C + 273.15
    r = np.clip(r, 0.0, 2.0)  # keep the denominator positive when extrapolating
    return ta / (1.0 - r * (1.0 - ta / tm)) - 273.15

def temp_power(r: np.ndarray, p: float) -> np.ndarray:
    """Signal proportional to (T - T_ambient)^p, so dT = dT_max * r^(1/p)."""
    r = np.clip(r, 0.0, None)
    return AMBIENT_TEMP_C + (MAX_TEMP_C - AMBIENT_TEMP_C) * r ** (1.0 / p)


def ambient_noise_stds(
    perform,
    settings,
    ambient: list[np.ndarray]
) -> tuple[list[float], int]:
    """
    Calculate the noise floor for the ambient frames:

    std of |d| for the first NOISE_TEST_FRAMES ambient frames,
    each measured against the mean of the remaining ambient frames
    (the test frames are removed from the reference).

    Returns (stds, number of frames in the reference).
    """
    # pylint:disable=line-too-long

    test_frames, reference = ambient[:NOISE_TEST_FRAMES], ambient[NOISE_TEST_FRAMES:]
    print(
        f"[temp] noise floor: {NOISE_TEST_FRAMES} ambient test frames vs reference of {len(reference)} frames"
    )
    flows = perform(reference, test_frames, settings)
    return [magnitude_std(f) for f in flows], len(reference)

# ---------------------------------------------------------------------------
# Analysis calls for each algorithm
# ---------------------------------------------------------------------------

def temperature_analysis(
    ambient: list[np.ndarray],
    heat: list[np.ndarray],
    heat_names: list[str]
) -> None:
    """
    Estimates the hot-plate temperatures from the std. of the displacement magnitude images.

    Calibration: ambient (std = noise floor) is AMBIENT_TEMP_C and the first group in
    HEAT_GROUPS is MAX_TEMP_C. The noise floor is removed in quadrature, the remaining
    signal is normalised by the max-setting signal (r), and r is mapped to temperature
    with two simple models (linear in T, and linear in air density deficit).
    """
    if TEMP_METHOD == "tvl1":
        perform, settings = perform_tv_l1, make_tvl1_settings()
    else:
        perform, settings = perform_window_deform, make_wdef_settings()

    # Which file is in which group (printed so the grouping can be checked by eye)
    group_of: dict[int, str] = {}
    for gname, (start, stop) in HEAT_GROUPS.items():
        for i in range(start, min(stop, len(heat))):
            group_of[i] = gname

    print("\nHeat images (natural sort order) and assigned group:")

    for i, name in enumerate(heat_names):
        print(f"  [{i + 1:2d}] {name:<30s} -> {group_of.get(i, '(unassigned, skipped)')}")

    # --------------------
    # Measure stds
    # --------------------

    noise_stds, n_noise_ref = ambient_noise_stds(perform, settings, ambient)
    # Heat images use all ambient frames as the reference, the noise test frames use fewer.
    # Residual noise scales as sqrt(1 + 1/n_ref), so correct for the difference.
    n_heat_ref = len(ambient)
    factor = float(np.sqrt((1 + 1 / n_heat_ref) / (1 + 1 / n_noise_ref)))
    noise_stds = [v * factor for v in noise_stds]
    noise = float(np.mean(noise_stds))
    noise_spread = float(np.std(noise_stds))

    idxs = sorted(group_of)
    stds = {}
    for n, i in enumerate(idxs):
        print(f"[temp] heat image {n + 1}/{len(idxs)} ({heat_names[i]})")
        stds[i] = magnitude_std(perform(ambient, [heat[i]], settings)[0])

    # --------------------
    # Calibrate, convert std. to temperature estimates.
    # --------------------

    def signal(std: float) -> float:
        return float(np.sqrt(max(std ** 2 - noise ** 2, 0.0)))

    # Remove background noise (assumed independent) to leave only displacement from heat.
    cal_group = next(iter(HEAT_GROUPS))
    cal_signal = float(np.mean([signal(stds[i]) for i in idxs if group_of[i] == cal_group]))
    r_img = {i: signal(stds[i]) / cal_signal for i in idxs}

    # Exponent for the power-law model: fitted to datasheet temperatures if given, else assumed
    p_used = SIGNAL_EXPONENT
    p_note = f"assumed exponent {p_used:g}"
    fit_lines = []
    num = den = 0.0
    for gname, t_known in KNOWN_TEMPS_C.items():
        if gname not in HEAT_GROUPS or gname == cal_group or t_known is None:
            continue
        r_g = float(np.mean([r_img[i] for i in idxs if group_of[i] == gname]))
        x = (t_known - AMBIENT_TEMP_C) / (MAX_TEMP_C - AMBIENT_TEMP_C)
        if r_g > 0 and 0 < x < 1:
            lr, lx = np.log(r_g), np.log(x)
            fit_lines.append(
                f"  {gname}: datasheet {t_known:.0f} C -> r = {r_g:.3f}, "
                f"implied exponent {lr / lx:.2f}"
            )
            num += lr * lx
            den += lx * lx
    if den > 0:
        p_used = num / den
        p_note = f"exponent fitted to the datasheet values: {p_used:.2f}"

    # Smallest signal distinguishable from the ambient noise (noise mean + 2 sd), as a temperature
    det_signal = float(np.sqrt(4 * noise * noise_spread + 4 * noise_spread ** 2))
    r_det = det_signal / cal_signal
    t_det_lin = float(temp_linear(r_det))
    t_det_dens = float(temp_density(r_det))
    t_det_pow = float(temp_power(r_det, p_used))

    rows = []
    for i in idxs:
        r = r_img[i]
        rows.append({
            "index": i + 1,
            "file": heat_names[i],
            "group": group_of[i],
            "std_px": stds[i],
            "signal_px": signal(stds[i]),
            "r": r,
            "T_linear_C": float(temp_linear(r)),
            "T_density_C": float(temp_density(r)),
            "T_power_C": float(temp_power(r, p_used))
        })

    # --------------------
    # Console output for report
    # --------------------
    print(
        f"\nAmbient noise floor: std(|d|) = {noise:.4f} px "
        f"(mean of {len(noise_stds)} test frames vs a {n_noise_ref}-frame reference, "
        f"spread {noise_spread:.4f})"
    )
    print(
        f"Calibration: ambient = {AMBIENT_TEMP_C:.0f} C,'{cal_group}' = {MAX_TEMP_C:.0f} C "
        f"(noise-subtracted signal {cal_signal:.4f} px)"
    )
    print(f"Power-law model: {p_note}")
    for line in fit_lines:
        print(line)
    print(
        f"Detection limit (noise + 2 sd): r = {r_det:.2f}  ->  about {t_det_lin:.0f} C (linear) / "
        f"{t_det_dens:.0f} C (density) / {t_det_pow:.0f} C (power). "
        f"Settings below this cannot be separated from noise.\n"
    )

    print("Per image")
    print(
        f"{'#':>3} {'file':<28} {'group':<8} {'std px':>8} {'signal':>8} {'r':>6} "
        f"{'T lin C':>8} {'T dens C':>9} {'T pow C':>8}"
    )
    for row in rows:
        print(
            f"{row['index']:>3} {row['file']:<28} {row['group']:<8} {row['std_px']:8.4f} "
            f"{row['signal_px']:8.4f} {row['r']:6.2f} {row['T_linear_C']:8.1f} "
            f"{row['T_density_C']:9.1f} {row['T_power_C']:8.1f}"
        )

    groups = {}
    for gname in HEAT_GROUPS:
        sel = [r_ for r_ in rows if r_["group"] == gname]
        if sel:
            groups[gname] = sel

    def rng_str(a: np.ndarray) -> str:
        return f"{a.mean():6.1f} ({a.min():.0f}-{a.max():.0f})".ljust(17)

    print("\nPer setting (mean, with min-max across the images in the group)")
    print(
        f"{'group':<8} {'n':>2} {'std px (mean+-sd)':>19} {'r':>6}  {'T linear C':<17} "
        f"{'T density C':<17} {'T power C':<17}"
    )
    summary = {}
    for gname, sel in groups.items():
        sd = np.array([r_["std_px"] for r_ in sel])
        rr = np.array([r_["r"] for r_ in sel])
        tl = np.array([r_["T_linear_C"] for r_ in sel])
        td = np.array([r_["T_density_C"] for r_ in sel])
        tp = np.array([r_["T_power_C"] for r_ in sel])
        summary[gname] = {"std": sd, "r": rr, "tl": tl, "td": td, "tp": tp}
        flag = ""
        if gname != cal_group and rr.mean() < r_det:
            flag = "  <- below detection limit: upper bound only"
        print(
            f"{gname:<8} {len(sel):>2} {sd.mean():>11.4f} +- {sd.std():.4f} {rr.mean():6.2f}  "
            f"{rng_str(tl)} {rng_str(td)} {rng_str(tp)}{flag}"
        )

    # --------------------
    # Plot results
    # --------------------

    out_dir = Path(OUT_DIRECTORY) / "temperature"

    colours = {
        name: c for name, c in zip(
            HEAT_GROUPS,
            ["tab:red", "tab:orange", "tab:blue", "tab:green"]
        )}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # ------------
    # Left image: std of |d| for every image, ambient first for comparison.
    # ------------

    xs, labels = [], []
    for k, v in enumerate(noise_stds):
        ax1.bar(k, v, color="tab:gray", label="Ambient (noise floor)" if k == 0 else None)
        xs.append(k)
        labels.append(f"amb {k + 1}")
    pos = len(noise_stds) + 1
    for gname, sel in groups.items():
        start = pos
        for j, row in enumerate(sel):
            ax1.bar(pos, row["std_px"], color=colours[gname], label=gname if j == 0 else None)
            xs.append(pos)
            labels.append(Path(row["file"]).stem)
            pos += 1
        mean = np.mean([r_["std_px"] for r_ in sel])
        ax1.hlines(mean, start - 0.4, pos - 0.6, colors="black", linestyles="-", linewidth=2)
        pos += 1
    ax1.axhline(noise, color="tab:gray", linestyle="--", linewidth=1)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, rotation=60, ha="right")
    ax1.set_ylabel("std of displacement magnitude (px)")
    ax1.set_title("Spread of displacement magnitude per image (black bar = group mean)")
    ax1.legend()

    # ------------
    # Right: calibration curves with the group estimates
    # ------------

    rgrid = np.linspace(0, 1.3, 200)
    ax2.plot(
        rgrid,
        temp_linear(rgrid),
        color="tab:purple",
        label="linear in T"
    )
    ax2.plot(
        rgrid,
        temp_density(rgrid),
        color="tab:green",
        label="linear in density deficit"
    )
    ax2.plot(
        rgrid,
        temp_power(rgrid, p_used),
        color="tab:brown",
        label=f"power law, exponent {p_used:.2f}"
    )
    ax2.scatter(
        [0, 1],
        [AMBIENT_TEMP_C, MAX_TEMP_C],
        color="black",
        zorder=5,
        label="calibration points"
    )

    # Place markers of estimates from std.
    for gname, vals in summary.items():
        if gname == cal_group:
            continue
        rm = vals["r"].mean()
        for t_mean, colr, dy in (
            (vals["tl"].mean(), "tab:purple", 8),
            (vals["tp"].mean(), "tab:brown", -2),
            (vals["td"].mean(), "tab:green", -14)
        ):
            ax2.scatter(rm, t_mean, color=colr, zorder=6)
            tag = " (upper bound)" if rm < r_det else ""
            ax2.annotate(
                f"{gname}: {t_mean:.0f} C{tag}",
                (rm, t_mean),
                textcoords="offset points",
                xytext=(8, dy),
                color=colr
            )

    ax2.set_xlabel("normalised signal r = sqrt(std^2 - noise^2) / max-setting signal")
    ax2.set_ylabel("estimated temperature (C)")
    ax2.set_title("Temperature estimate from displacement spread")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left")

    fig.savefig(out_dir / "temperature_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_dir / 'temperature_analysis.png'} and {out_dir / 'per_image_stats.csv'}")


def tv_l1_analysis(
    ambient: list[np.ndarray], heat: list[np.ndarray],
    heat_names: list[str] | None = None
) -> None:
    """
    Run TV-L1 on every heat image,
    with an ambient-vs-ambient noise-floor check to the side for comparison.
    """
    analyse_all(
        "TV-L1", "tvl1",
        perform_tv_l1, make_tvl1_settings(),
        ambient, heat, heat_names
    )

def wdef_analysis(
    ambient: list[np.ndarray],heat: list[np.ndarray],
    heat_names: list[str] | None = None
) -> None:
    """
    Run window-deformation PIV on every heat image,
    with an ambient-vs-ambient noise-floor check to the side for comparison.
    """
    analyse_all(
        "Window deformation", "wdef",
        perform_window_deform, make_wdef_settings(),
        ambient, heat, heat_names
    )



if __name__ == "__main__":
    os.chdir(sys.path[0])

    # Load images to stack
    directory = Path(IMAGE_DIRECTORY)
    ambient_stack, heat_stack, heat_file_names = load_images(directory)

    # Pick processing method so don't have to run all every time.
    if METHOD in ("tvl1", "both"):
        tv_l1_analysis(ambient_stack, heat_stack, heat_file_names)
    if METHOD in ("wdef", "both"):
        wdef_analysis(ambient_stack, heat_stack, heat_file_names)
    if METHOD == "temp":
        temperature_analysis(ambient_stack, heat_stack, heat_file_names)
