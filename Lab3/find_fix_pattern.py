"""
find_fixed_pattern.py

Finds the fixed pattern noise and saves it in a pickle file.
"""

import pickle
import utils
import numpy as np

BLACK_LEVEL: int = 16
"""The black level of the camera, known to be 16."""

IMG_DIR: str = "black_images"
"""Directory containing the images. Image names should be in the format 01.dng, 02.dng, etc."""

IMG_COUNT: int = 16
"""Number of images in IMG_DIR to average."""

SAVE_NAME: str = "fixed_pattern.pickle"
"""File to save pickle as."""

if __name__ == "__main__":
    # Variable to dump
    fixed_pattern: np.ndarray

    # Load images into numpy array as a stack, subtracting black level while doing so
    raw_image_list: list[np.ndarray] = []
    raw_images: np.ndarray

    for i in range(IMG_COUNT):
        img_path: str = f"{IMG_DIR}/{i+1:02d}.dng"
        raw_images.append( utils.read_raw(img_path) - BLACK_LEVEL )
    
    raw_images = np.stack(raw_image_list)

    # Calculate average as fixed-pattern noise
    fixed_pattern = np.mean(raw_Images, axis=0)

    # Sanity check prints
    print(f"Min: {np.min(fixed_pattern)}, Max: {np.max(fixed_pattern)}, Ave: {np.mean(fixed_pattern)}")

    # Save as pickle
    pickle.dump(fixed_pattern, SAVE_NAME)
