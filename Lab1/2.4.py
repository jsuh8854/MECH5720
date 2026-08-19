import numpy as np
import utils
from pathlib import Path

output_folder = "./generated_debayers/"
big_iris_folder : str = "./Part2/2_4/big_iris"
small_iris_folder : str = "./Part2/2_4/small_iris"

BLACK_LEVEL : float = 16.0

def calculate_mean_brightness(p_images: List[np.ndarray]) -> float:
    images = np.stack(p_images)
    return np.mean(images) - BLACK_LEVEL


def batch_load_raws(folder: str) -> List[np.ndarray]:
    raws : List[np.ndarray] = []

    for file in Path(folder).iterdir():
        if file.is_file():
            raw = utils.read_raw(str(file))
            raws.append(raw)
    return raws

def save_processed_raw(raw: np.ndarray, filename: str) -> None:
    rgb =  utils.debayer_raw(raw)
    rgb_rotated = utils.rotate_image(rgb)
    rgb_balanced = utils.white_balance_rgb(rgb_rotated)
    
    utils.save_image(rgb_balanced, filename, output_folder)


if __name__ == "__main__":
    big_iris_raws = batch_load_raws(big_iris_folder)
    small_iris_raws = batch_load_raws(small_iris_folder)

    # Output example images
    save_processed_raw(big_iris_raws[0], "big_iris_example")
    save_processed_raw(small_iris_raws[0], "small_iris_example")

    # Required outputs

    print("Black offset:", str(BLACK_LEVEL))

    big_iris_mean_brightness = calculate_mean_brightness(big_iris_raws)
    small_iris_mean_brightness = calculate_mean_brightness(small_iris_raws)

    print("Large iris mean brightness:", str(big_iris_mean_brightness))
    print("Small iris mean brightness:", str(small_iris_mean_brightness))

    brightness_ratio = big_iris_mean_brightness / small_iris_mean_brightness
    print("Brightness ratio:", str(brightness_ratio))

    estimate_diameter_ratio = np.sqrt(brightness_ratio)
    print("Estimate diameter ratio:", str(estimate_diameter_ratio))

