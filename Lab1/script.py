import utils
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

output_folder = "./generated_debayers/"

def process_image(path: str, brighten: bool = False) -> np.ndarray:
    raw = utils.read_raw(path)
    rgb = utils.debayer_raw(raw)
    rgb_rotated = utils.rotate_image(rgb)
    rgb_balanced = utils.white_balance_rgb(rgb_rotated)
    if brighten:
        rgb_balanced = utils.brighten_image(rgb_balanced)

    return rgb_balanced

def save_image(image: np.ndarray, path: str) -> None:
    utils.save_image(image, Path(path).stem, output_folder)

def batch_debayer(paths: List[str], brighten: bool = False) -> List[np.ndarray]:
    images : List[np.ndarray] = []
    for path in paths:
        img = process_image(path, brighten)

        images.append(img)

        utils.save_image(img, Path(path).stem, output_folder)
        # utils.show_image(rgb_balanced)

    return images


# Get line-trace plot of images and plot
def create_line_trace(paths: List[str], output_file: str = "gain_line_trace") -> None:
    MARGIN : int = 20 # Margin to only read the bar

    names : List[str] = []
    
    plt.figure()

    for path in paths:
        name = Path(path).stem
        names.append(name)
        raw = utils.read_raw(path)
        green = raw[0::2, 1::2]
        row = green[MARGIN:-MARGIN, 64]
        print(name + " Min: " + str(np.min(row)) + ", Max: " + str(np.max(row)))

        x = np.arange(green.shape[0] - MARGIN * 2)

        plt.plot(x, row)

    plt.xlabel("X Position")
    plt.ylabel("Intensity")
    plt.title("Green Channel Line-trace")
    plt.grid(True)
    plt.legend(names)
    plt.savefig(output_folder + output_file +  ".png")
    # plt.show()


if __name__ == "__main__":
    utils.update_working_directory()
    # 1.2
    # part_1_2_paths = ["./pics/balanced_raw.dng", "./pics/high_gain_raw.dng", "./pics/low_gain_raw.dng", ]
    # batch_debayer(part_1_2_paths) # Generate images
    # create_line_trace(part_1_2_paths) # Generate plot
    # create_line_trace(["./pics/low_gain_raw.dng"], "low_gain_trace") # Generate plot alone of small

    # 1.3
    # part_1_3_paths = ["./Part1/1_3/raw/short_exposure_raw.dng", "./Part1/1_3/raw/log_exposure_raw.dng"]
    # batch_debayer(part_1_3_paths, True) # Generate images

    # 2.1
    # part_2_1_paths = ["./Part2/2_1/raw/focused_lens_front_raw.dng", "./Part2/2_1/raw/focused_lens_reverse.dng"]
    # batch_debayer(part_2_1_paths, True)

    # 2.2
    # part_2_2_paths = ["./Part2/2_2/raw/small_iris_front_raw.dng", "./Part2/2_2/raw/small_iris_back_raw.dng"]
    # part_2_2_images = batch_debayer(part_2_2_paths, True)

    # 2.3
    # part_2_3_paths = [
    #     "./Part2/2_3/raw/no_iris_close_raw.dng", 
    #     "./Part2/2_3/raw/no_iris_center_raw.dng", 
    #     "./Part2/2_3/raw/no_iris_far_raw.dng",
    #     "./Part2/2_3/raw/iris_close_raw.dng", 
    #     "./Part2/2_3/raw/iris_center_raw.dng", 
    #     "./Part2/2_3/raw/iris_far_raw.dng",
    # ]
    # part_2_3_images = batch_debayer(part_2_3_paths, True)

