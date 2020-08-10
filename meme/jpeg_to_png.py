import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

INPUT_DATA_DIR = "d:\\ml\\goat-data\\goat-train-compressed"
OUTPUT_DATA_DIR = "d:\\ml\\goat-data\\goat-train-input"


def convert(file):
    im = Image.open(INPUT_DATA_DIR + "\\" + file)
    im.save(OUTPUT_DATA_DIR + "\\" + file.replace(".jpg", ".png"), "png")


if __name__ == '__main__':
    files = os.listdir(INPUT_DATA_DIR)

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(convert, files)
