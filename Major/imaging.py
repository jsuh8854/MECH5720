"""Imaging code"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import cv2
import utils
import numpy as np
from matplotlib import pyplot as plt

# cv2.calcOpticalFlowFarneback

IMAGE_DIRECTORY: str = "data"
OUT_DIRECTORY: str = "out"

@dataclass
class TVL1Settings:
    """Settings for the TV-L1 solver."""
    l: float = 0.15 # Lower for smoother, higher for higher trust in data
    """Lambda"""
    tau: float = 0.25
    scales_number: int = 5
    warpings_number: int = 5
    epsilon: float = 0.01
    inner_iterations: int = 30
    outer_iterations: int = 10

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
    # Optional, test if it actually helps.
    # ref = cv2.GaussianBlur(ref, (0, 0), 1.0)

    # Create the solver
    tv_l1 = create_tvl1(tv_l1_settings)

    results: list[np.ndarray] = [] # To return

    for i, img in enumerate(heat_images):
        print(f"Processing image {i+1}...")
        cur = cv2.GaussianBlur(img, (0, 0), 1.0)

        # Match overall brightness to the reference (lamp drift, exposure)
        cur *= ref.mean() / max(cur.mean(), 1e-6)

        flow = tv_l1.calc(ref, cur, None)  # (H, W, 2) with dx, dy

        # 3. Remove global rigid shift from vibration/drift
        flow[..., 0] -= np.median(flow[..., 0])
        flow[..., 1] -= np.median(flow[..., 1])

        results.append(flow.astype(np.float32))

    print("Done!")

    return results


if __name__ == "__main__":
    os.chdir(sys.path[0])

    directory = Path(IMAGE_DIRECTORY)
    ambient = [
        utils.process_raw(str(file))
        for file in directory.iterdir()
        if file.is_file() and file.name.startswith("second_stripes_ambient_")
    ]
    heat = [
        utils.process_raw(str(file))
        for file in directory.iterdir()
        if file.is_file() and file.name.startswith("second_stripes_heat_")
    ]
    tv_settings = TVL1Settings(
        l=0.15,
        tau=0.25,
        scales_number=5,
        warpings_number=5,
        epsilon=0.01,
        inner_iterations=30,
        outer_iterations=10
    )

    images = perform_tv_l1(ambient, [heat[1]], tv_settings)
    ambient_comparison = perform_tv_l1(ambient, [ambient[1]], tv_settings)

    img1 = images[0][:, :, 0]
    img2 = ambient_comparison[0][:, :, 0]

    # Shared colour scale
    vmin = min(img1.min(), img2.min())
    vmax = max(img1.max(), img2.max())

    fig, axes = plt.subplots(1, 2, figsize=(8, 12), constrained_layout=True)

    im1 = axes[0].imshow(img1, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Heat")
    axes[0].axis("off")

    im2 = axes[1].imshow(img2, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Ambient Test")
    axes[1].axis("off")

    fig.colorbar(im1, ax=axes, label="Intensity", location="right", pad=0.04, shrink=0.8)
    plt.show()
