"""Sensitivity test for a BOS solver: can it recover a *known* small displacement
from your own pattern, at your own noise level?

Usage (inside imaging_1.py, after loading ambient/heat):

    from bos_diagnostics import recovery_test, flow_stats
    tv = create_tvl1(TVL1Settings())
    ref = np.mean(ambient, axis=0)
    recovery_test(lambda r, c: tv.calc(r, c, None), ref, ambient)
"""
from typing import Callable
import cv2
import numpy as np


def inject_shift(image: np.ndarray, amp_px: float, sigma_px: float = 150.0):
    """Warp `image` by a smooth Gaussian-bump displacement with peak `amp_px` (dx=amp, dy=-amp/2).

    Returns (shifted_image, true_flow) with true_flow shaped (H, W, 2) = (dx, dy),
    using the convention cur(x) = ref(x - d).
    """
    h, w = image.shape
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-(((X - w / 2) ** 2 + (Y - h / 2) ** 2) / (2 * sigma_px ** 2))).astype(np.float32)
    flow = np.stack([amp_px * g, -0.5 * amp_px * g], axis=-1)
    shifted = cv2.remap(image.astype(np.float32), X - flow[..., 0], Y - flow[..., 1],
                        cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return shifted, flow


def estimate_noise(ambient: list[np.ndarray]) -> float:
    """Per-pixel temporal noise sigma, from the spread of the ambient frames."""
    return float(np.std(np.stack(ambient), axis=0).mean())


def recovery_test(solver: Callable[[np.ndarray, np.ndarray], np.ndarray],
                  ref: np.ndarray, ambient: list[np.ndarray],
                  amps=(0.05, 0.1, 0.2, 0.5, 1.0), sigma_px: float | None = None) -> None:
    """Inject known shifts (plus realistic noise) and report how well `solver` recovers them.

    solver(ref, cur) must return an (H, W, 2) array of (dx, dy) in pixels.
    """
    h, w = ref.shape
    sigma_px = sigma_px or min(h, w) / 6
    noise = estimate_noise(ambient)
    rng = np.random.default_rng(0)
    print(f"ambient temporal noise sigma = {noise:.4g} (image units), "
          f"image mean = {ref.mean():.4g}")
    print(f"{'true core dx':>12} {'recovered core dx':>18} {'ratio':>7} {'rmse dx':>9}")

    for amp in amps:
        shifted, truth = inject_shift(ref, amp, sigma_px)
        cur = shifted + rng.normal(0, noise, shifted.shape).astype(np.float32)
        est = solver(ref, cur)
        est = est - np.median(est.reshape(-1, 2), axis=0)  # same rigid-shift removal as the pipeline
        # Average over the bump's core (not the max, which would just pick out noise spikes)
        core = truth[..., 0] > 0.9 * amp
        offset = np.median(truth[..., 0])
        peak_true = truth[..., 0][core].mean() - offset
        peak_est = est[..., 0][core].mean()
        rmse = np.sqrt(np.mean((est[..., 0] - (truth[..., 0] - offset)) ** 2))
        print(f"{peak_true:12.3f} {peak_est:18.3f} {peak_est / peak_true:7.2f} {rmse:9.3f}")


def flow_stats(flow: np.ndarray, name: str = "") -> None:
    """Print percentile statistics so you can see real numbers instead of colour-scaled plots."""
    mag = np.hypot(flow[..., 0], flow[..., 1])
    p = np.percentile(mag, [50, 90, 99, 99.9])
    print(f"{name}: |d| median={p[0]:.4f} p90={p[1]:.4f} p99={p[2]:.4f} p99.9={p[3]:.4f} "
          f"max={mag.max():.4f} px")
