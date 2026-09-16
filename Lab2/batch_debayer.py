"""
Convert all dngs in a folder to debayered pngs.
Also save line trace of each image (from Lab 1) to find black level.
"""

from pathlib import Path
import utils
import numpy as np

CROP_IMAGE: bool = False

# raw_folder = Path("./data/captured/")
raw_folder = Path("./Bonus/")
png_folder = Path("./data/debayered/")
trace_folder = Path("./data/trace/")

if __name__ == "__main__":
    for file in raw_folder.iterdir():
        if file.is_file():
            print("="*40)
            print(file.name)
            raw = utils.read_raw(str(file))
            rgb16 = utils.debayer_raw(raw)
            if CROP_IMAGE:
                rgb16 = utils.crop_square(rgb16, 960, 768, 1024)
            img_rot = utils.rotate_image(rgb16)

            utils.create_line_trace_of_raw(raw, str(trace_folder), file.stem)

            max_val = np.max(img_rot)
            print(f"Max Value: {max_val}")
            if max_val == utils.MAX_PIXEL - 1.0:
                print("WARNING: Image is saturated!")

            utils.save_image(img_rot, file.stem, png_folder)
