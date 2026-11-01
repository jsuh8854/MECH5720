"""Imaging code"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import cv2
import utils
import numpy as np
from matplotlib import pyplot as plt
from debug import flow_stats

# cv2.calcOpticalFlowFarneback

IMAGE_DIRECTORY: str = "data"
OUT_DIRECTORY: str = "out"
FILE_PREFIX: str = ""   # images are <prefix>ambient_* and <prefix>heat_*
METHOD: str = "both"                   # "tvl1", "wdef" or "both"


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


# --------------------------------------------------------------------------- #
# TV-L1
# --------------------------------------------------------------------------- #

def create_tvl1(settings: TVL1Settings):
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

    # Create the solver
    tv_l1 = create_tvl1(tv_l1_settings)

    results: list[np.ndarray] = [] # To return

    for i, img in enumerate(heat_images):
        print(f"Processing image {i+1}...")
        cur = img * scale
        if tv_l1_settings.blur_sigma > 0:
            cur = cv2.GaussianBlur(cur, (0, 0), tv_l1_settings.blur_sigma)

        # Match overall brightness to the reference (lamp drift, exposure)
        cur = cur * (ref.mean() / max(cur.mean(), 1e-6))

        flow = tv_l1.calc(ref, cur, None)  # (H, W, 2) with dx, dy

        # 3. Remove global rigid shift from vibration/drift
        flow[..., 0] -= np.median(flow[..., 0])
        flow[..., 1] -= np.median(flow[..., 1])

        results.append(flow.astype(np.float32))

    print("Done!")

    return results


# --------------------------------------------------------------------------- #
# Window-deformation cross-correlation
#
# Inputs are assumed to be single-channel float32 arrays already.
# Output convention matches TV-L1: (H, W, 2) = (dx, dy) in pixels, where a
# feature at x in the reference appears at x + d in the measurement image.
# --------------------------------------------------------------------------- #

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


def _correlate_grid(a: np.ndarray, b: np.ndarray, win: int,
                    y0: np.ndarray, x0: np.ndarray, settings: WDefSettings) -> np.ndarray:
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
        nb = np.stack([pad[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx, c]
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
        med = np.nanmedian(nb, axis=0)
        rm = np.nanmedian(np.abs(nb - med), axis=0)
        r = np.abs(field[..., c] - med) / (rm + 0.1)
        bad = np.isnan(field[..., c]) | (r > thresh)
        out[..., c] = np.where(bad, med, field[..., c])
    # anything still NaN (large untextured regions) -> 0
    return np.nan_to_num(out, nan=0.0)


def _grid_to_dense(grid: np.ndarray, win: int, step: int, y0: np.ndarray,
                   x0: np.ndarray, h: int, w: int) -> np.ndarray:
    """Cubic-interpolate the vector grid to one vector per pixel."""
    cy0 = y0[0] + (win - 1) / 2
    cx0 = x0[0] + (win - 1) / 2
    gy = np.clip((np.arange(h, dtype=np.float32) - cy0) / step, 0, len(y0) - 1)
    gx = np.clip((np.arange(w, dtype=np.float32) - cx0) / step, 0, len(x0) - 1)
    mapx, mapy = np.meshgrid(gx.astype(np.float32), gy.astype(np.float32))
    return cv2.remap(grid.astype(np.float32), mapx, mapy, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)


def window_deform_piv(ref: np.ndarray, cur: np.ndarray, settings: WDefSettings) -> np.ndarray:
    """Dense (H, W, 2) displacement (dx, dy) of cur relative to ref."""
    # pylint:disable=no-member
    h, w = ref.shape
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)

    disp = np.zeros((h, w, 2), np.float32)  # current dense estimate

    for win in settings.window_sizes:
        step, y0, x0 = _grid_axes(h, w, win, settings.overlap)
        for it in range(settings.iterations):
            print(f"  window {win}px, iteration {it + 1}/{settings.iterations}")
            # Symmetric deformation: both images are moved by half the estimate
            half = 0.5 * disp
            a = cv2.remap(ref, X - half[..., 0], Y - half[..., 1],
                          cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            b = cv2.remap(cur, X + half[..., 0], Y + half[..., 1],
                          cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

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


# --------------------------------------------------------------------------- #
# Loading, plotting, analyses
# --------------------------------------------------------------------------- #

def load_images(directory: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load the ambient and heat image stacks."""
    ambient = [
        utils.process_raw(str(file))
        for file in sorted(directory.iterdir())
        if file.is_file() and file.name.startswith(f"{FILE_PREFIX}ambient_")
    ]
    heat = [
        utils.process_raw(str(file))
        for file in sorted(directory.iterdir())
        if file.is_file() and file.name.startswith(f"{FILE_PREFIX}heat_")
    ]
    return ambient, heat


def plot_displacements(
    heat_flow: np.ndarray, ambient_flow: np.ndarray,
    heat_image: np.ndarray, ambient_image: np.ndarray,
    method: str, save_path: Path,
    central_fraction: float = 0.925
) -> None:
    """4x2 plot: x displacement, y displacement and displacement magnitude for the
    heat image and the ambient noise floor, with the debayered source images underneath.

    The shared colour scales span only the central `central_fraction` of all
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
    # vmin = -2.0
    # vmax = 2.0
    # mmax = 3.0

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
    plt.close(fig)  # free memory; important when saving many plots


def analyse_all(name: str, out_subdir: str, perform, settings,
                ambient: list[np.ndarray], heat: list[np.ndarray]) -> None:
    """Run `perform` on every heat image and save one plot per image.

    The ambient-vs-ambient noise floor is computed once and reused for every plot.
    Images are processed one at a time so memory stays flat and finished plots are
    already on disk if a long run is interrupted.
    Plots go to OUT_DIRECTORY/<out_subdir>/<out_subdir>_<index>.png, where the index
    follows the sorted order of the heat_* files.
    """
    if not heat:
        print(f"[{name}] no heat images found, nothing to do.")
        return

    out_dir = Path(OUT_DIRECTORY) / out_subdir
    width = max(3, len(str(len(heat) - 1)))

    print(f"[{name}] computing ambient noise floor...")
    ambient_flow = perform(ambient, [ambient[0]], settings)[0]

    for i, heat_image in enumerate(heat):
        print(f"[{name}] heat image {i + 1}/{len(heat)}")
        flow = perform(ambient, [heat_image], settings)[0]
        plot_displacements(flow, ambient_flow, heat_image, ambient[0], name,
                           out_dir / f"{out_subdir}_{i:0{width}d}.png")

    print(f"[{name}] saved {len(heat)} plots to {out_dir}")


def tv_l1_analysis(ambient: list[np.ndarray], heat: list[np.ndarray]) -> None:
    """Run TV-L1 on every heat image, with an ambient-vs-ambient noise-floor check."""
    tv_settings = TVL1Settings(
        l=0.15,
        tau=0.25,
        scales_number=5,
        warpings_number=5,
        epsilon=0.01,
        inner_iterations=30,
        outer_iterations=10
    )
    analyse_all("TV-L1", "tvl1", perform_tv_l1, tv_settings, ambient, heat)


def wdef_analysis(ambient: list[np.ndarray], heat: list[np.ndarray]) -> None:
    """Run window-deformation PIV on every heat image, with an ambient noise-floor check."""
    wdef_settings = WDefSettings(
        window_sizes=(64, 32, 16),
        overlap=0.5,
        iterations=2,
        max_shift_frac=0.4,
        median_thresh=2.0,
        min_std=1e-3
    )
    analyse_all("Window deformation", "wdef", perform_window_deform, wdef_settings,
                ambient, heat)


if __name__ == "__main__":
    os.chdir(sys.path[0])

    directory = Path(IMAGE_DIRECTORY)
    ambient_stack, heat_stack = load_images(directory)

    if METHOD in ("tvl1", "both"):
        tv_l1_analysis(ambient_stack, heat_stack)
    if METHOD in ("wdef", "both"):
        wdef_analysis(ambient_stack, heat_stack)
