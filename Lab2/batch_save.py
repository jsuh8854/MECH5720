"""
Convert all dngs in a folder to not debayered pngs.
"""

from pathlib import Path
import utils
import numpy as np

raw_folder = Path("./data/captured/")
png_folder = Path("./data/bayered/")
trace_folder = Path("./data/trace/")

if __name__ == "__main__":
    for file in raw_folder.iterdir():
        if file.is_file():
            print("="*40)
            print(file.name)
            raw = utils.read_raw(str(file))
            raw_normalised = utils.normalise_raw(raw)
            img_rot = utils.rotate_image(raw_normalised)
            img_crop = utils.crop_square(img_rot, 256, 900, 1024)

            max_val = np.max(img_crop)
            print(f"Max Value: {max_val}")
            if max_val == utils.MAX_PIXEL - 1.0:
                print("WARNING: Image is saturated!")

            utils.save_image(img_crop, file.stem, png_folder)
