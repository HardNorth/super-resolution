import os
from typing import List

from PIL import Image

INPUT_DATA_DIR = "c:\\ml\\photo\\output"
OUTPUT_DATA_DIR = "c:\\ml\\photo"

OUTPUT_SIZE = (6400, 4800)


def convert(files: List[str]):
    i = 0
    filename = files[i]
    im = Image.open(INPUT_DATA_DIR + os.path.sep + files[i])
    result = Image.new(im.mode, OUTPUT_SIZE)
    while i < len(files):
        for w in range(0, OUTPUT_SIZE[0], im.size[0]):
            for h in range(0, OUTPUT_SIZE[1], im.size[1]):
                result.paste(im, (w, h, w + im.size[0], h + im.size[1]))
                i += 1
                if i < len(files):
                    im = Image.open(INPUT_DATA_DIR + os.path.sep + files[i])
        result.save(OUTPUT_DATA_DIR + os.path.sep + filename)
        if i < len(files):
            filename = files[i]


if __name__ == '__main__':
    files = os.listdir(INPUT_DATA_DIR)
    convert(files)
