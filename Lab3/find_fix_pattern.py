"""
find_fixed_pattern.py

Finds the fixed pattern noise and saves it in a pickle file.
"""

import pickle
import utils
import numpy as np
from matplotlib import pyplot as plt

BLACK_LEVEL: int = 16
"""The black level of the camera, known to be 16."""

IMG_DIR: str = "Lab3_cap"
"""Directory containing the images. Image names should be in the format 1.dng, 2.dng, etc."""

IMG_COUNT: int = 16
"""Number of images in IMG_DIR to average."""

SAVE_NAME: str = "fixed_pattern.pkl"
"""File to save pickle as."""

CROP_SIZE: int = 64
"""Size of crop to use of the fixed pattern noise."""

if __name__ == "__main__":
    # Variable to dump
    fixed_pattern: np.ndarray

    # Load images into numpy array as a stack, subtracting black level while doing so
    raw_image_list: list[np.ndarray] = []
    raw_images: np.ndarray

    for i in range(IMG_COUNT):
        img_path: str = f"{IMG_DIR}/{i+1}.dng"
        black_image: np.ndarray =  utils.read_raw(img_path) - BLACK_LEVEL
        raw_image_list.append(black_image)

    raw_images = np.stack(raw_image_list)

    # Calculate average as fixed-pattern noise, clipping bottom values
    fixed_pattern = np.clip(np.mean(raw_images, axis=0), 0, utils.MAX_PIXEL)

    # Sanity checks
    min_val: float = np.min(fixed_pattern)
    max_val: float = np.max(fixed_pattern)
    ave_val: float = np.mean(fixed_pattern)

    print(f"Shape: {fixed_pattern.shape}")
    print(
        f"Min: {min_val}, Max: {max_val}, Ave: {ave_val}"
    )

    # Save as pickle
    with open(SAVE_NAME, "wb") as f:
        pickle.dump(fixed_pattern, f)

    # Find brightest spot and crop around it to show most interesting
    max_y, max_x = np.unravel_index( np.argmax(fixed_pattern), fixed_pattern.shape ) # pylint: disable=unbalanced-tuple-unpacking

    y_start: int = max_y - CROP_SIZE // 2
    x_start: int = max_x - CROP_SIZE // 2
    y_end: int = max_y + CROP_SIZE // 2
    x_end: int = max_x + CROP_SIZE // 2

    crop: np.ndarray = fixed_pattern[y_start:y_end, x_start:x_end]

    plt.imshow(crop, cmap="viridis")
    plt.axis("off")
    plt.show()

    utils.save_image(crop, "fixed_pattern", "p0_fixed_pattern")
