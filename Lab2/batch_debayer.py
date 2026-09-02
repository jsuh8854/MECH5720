"""
Convert all dngs in a folder to pngs.
"""

from pathlib import Path
import utils
import numpy as np

raw_folder = Path("./data/captured/")
png_folder = Path("./data/debayered/")

if __name__ == "__main__":
    for file in raw_folder.iterdir():
        if file.is_file():
            print("="*40)
            print(file.name)
            raw = utils.read_raw(str(file))
            rgb16 = utils.debayer_raw(raw)

            max_val = np.max(rgb16)
            print(f"Max Value: {max_val}")
            if max_val == utils.MAX_PIXEL - 1.0:
                print("WARNING: Image is saturated!")

            rgb16_rot = utils.rotate_image(rgb16)
            reduced_image = utils.reduce_image(rgb16_rot, 3)
            utils.save_image(reduced_image, file.stem, png_folder)
