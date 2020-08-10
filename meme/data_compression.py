from typing import Tuple

import os
from PIL import Image
import imagehash
import math
import random

INPUT_DATA_DIR = "d:\\ml\\goat-data\\goat-train-small"
OUTPUT_DATA_DIR = "d:\\ml\\goat-data\\goat-train-compressed"

if __name__ == '__main__':

    files = os.listdir(INPUT_DATA_DIR)

    for i in range(len(files)):
        im = Image.open(INPUT_DATA_DIR + "\\" + files[i])
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[3])  # 3 is the alpha channel
        background.save(OUTPUT_DATA_DIR + "\\" + files[i].replace(".png", ".jpg"), "jpeg", quality=30)
