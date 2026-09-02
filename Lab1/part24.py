"""
Lab 1 Part 2.4 code
"""

# ====================
# Part2.py
# ====================
from pathlib import Path
import numpy as np
import utils

# Change to match actual locations
OUTPUT_FOLDER = "./generated_debayers/"
large_iris_folder : str = "./Part2/2_4/big_iris"
small_iris_folder : str = "./Part2/2_4/small_iris"

BLACK_LEVEL : float = 16.0

def calculate_mean_brightness(p_images: list[np.ndarray]) -> float:
    """arst"""
    images = np.stack(p_images)
    return np.mean(images) - BLACK_LEVEL

def calculate_mean_noise(p_images: list[np.ndarray]) -> float:
    """arst"""
    images = np.stack(p_images)
    noise = np.std(images, axis=0, ddof=1)
    return float(np.mean(noise))

def batch_load_raws(folder: str) -> list[np.ndarray]:
    """arst"""
    raws : list[np.ndarray] = []

    for file in Path(folder).iterdir():
        if file.is_file():
            raw = utils.read_raw(str(file))
            raws.append(raw)
    return raws

def save_processed_raw(raw: np.ndarray, filename: str, amplified: bool = False) -> None:
    """arst"""
    rgb =  utils.debayer_raw(raw)
    rgb_rotated = utils.rotate_image(rgb)
    rgb_balanced = utils.white_balance_rgb(rgb_rotated)
    if amplified:
        rgb_balanced = utils.brighten_image(rgb_balanced)

    utils.save_image(rgb_balanced, filename, OUTPUT_FOLDER)

if __name__ == "__main__":
    large_iris_raws = batch_load_raws(large_iris_folder)
    small_iris_raws = batch_load_raws(small_iris_folder)

    # Output example images
    save_processed_raw(large_iris_raws[0], "large_iris_example")
    save_processed_raw(small_iris_raws[0], "small_iris_example")
    save_processed_raw(small_iris_raws[0], "small_iris_boosted_example", True)

    # 2.4 outputs

    large_iris_mean_brightness = calculate_mean_brightness(large_iris_raws)
    small_iris_mean_brightness = calculate_mean_brightness(small_iris_raws)
    brightness_ratio = large_iris_mean_brightness / small_iris_mean_brightness
    estimate_diameter_ratio = np.sqrt(brightness_ratio)

    print("====================\nPart 2.4 Output\n====================")
    print("Black offset:", str(BLACK_LEVEL))
    print("Large iris mean brightness:", str(large_iris_mean_brightness))
    print("Small iris mean brightness:", str(small_iris_mean_brightness))
    print("Brightness ratio:", str(brightness_ratio))
    print("Estimate diameter ratio:", str(estimate_diameter_ratio))

    # 2.5 outputs

    large_iris_mean_noise = calculate_mean_noise(large_iris_raws)
    small_iris_mean_noise = calculate_mean_noise(small_iris_raws)
    large_iris_snr = 20 * np.log10(large_iris_mean_brightness / large_iris_mean_noise)
    small_iris_snr = 20 * np.log10(small_iris_mean_brightness / small_iris_mean_noise)
    snr_delta = large_iris_snr - small_iris_snr

    print("====================\nPart 2.5 Output\n====================")
    print("Large iris mean brightness:", str(large_iris_mean_brightness))
    print("Large iris mean noise:", str(large_iris_mean_noise))
    print("Small iris mean brightness:", str(small_iris_mean_brightness))
    print("Small iris mean noise:", str(small_iris_mean_noise))
    print("--------------------------------")
    print("Large iris SNR (dB):", str(large_iris_snr))
    print("Small iris SNR (dB):", str(small_iris_snr))
    print("--------------------------------")
    print("Change in SNR (dB):", str(snr_delta))
