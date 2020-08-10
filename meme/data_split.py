import os

from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from meme.utils import split_image_by_size, save_images

INPUT_DATA_DIR = "c:\\ml\\photo\\raw"
OUTPUT_DATA_DIR = "c:\\ml\\photo\\input"

SIZE = (400, 300)


def convert(file: str):
    im = Image.open(INPUT_DATA_DIR + "\\" + file)
    images = split_image_by_size(im, SIZE)
    save_images(images, OUTPUT_DATA_DIR, name_pattern=file[:file.index('.')] + '_{:0>2d}')


if __name__ == '__main__':
    files = os.listdir(INPUT_DATA_DIR)

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(convert, files)
