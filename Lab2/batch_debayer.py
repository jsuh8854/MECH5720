import utils
import numpy as np
from pathlib import Path

raw_folder = Path("./data/captured/")
png_folder = Path("./data/debayered/")

if __name__ == "__main__":
    for file in raw_folder.iterdir():
        if file.is_file():
            raw = utils.read_raw(str(file))
            rgb16 = utils.debayer_raw(raw)
            rgb16_rot = utils.rotate_image(rgb16)
            reduced_image = utils.reduce_image(rgb16_rot, 3)
            utils.save_image(reduced_image, file.stem, png_folder)
    

