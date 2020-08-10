import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

INPUT_DATA_DIR = "d:\\ml\\goat-data\\DIV2K_HR"
OUTPUT_DATA_DIR = "d:\\ml\\goat-data\\DIV2K_LR"

SIZE = 270


def convert(file):
    im = Image.open(INPUT_DATA_DIR + "\\" + file)
    if im.size[0] > im.size[1]:
        result_size = (int(im.size[0] * (im.size[1] / SIZE)), SIZE)
    else:
        result_size = (SIZE, int(im.size[1] * (im.size[0] / SIZE)))
    im.thumbnail(result_size, Image.BICUBIC)
    im.save(OUTPUT_DATA_DIR + "\\" + file, "png")


if __name__ == '__main__':
    files = os.listdir(INPUT_DATA_DIR)

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(convert, files)
