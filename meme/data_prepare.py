from typing import Tuple

import os
from PIL import Image
import imagehash
import math
import random

INPUT_DATA_DIR = "d:\\ml\\goat-data\\goat-train-compressed"
OUTPUT_DATA_DIR = "d:\\ml\\goat-data\\goat-train-input"


def to_ratio(image: Image, ratio: Tuple[int, int] = (4, 3)) -> Image:
    width = image.size[0]
    height = image.size[1]
    wanted_ratio = ratio[0] / ratio[1]
    new_width = int(height * wanted_ratio)
    if new_width == width:
        return image
    if width > new_width:  # cut edges
        left_corner = (width - new_width) // 2
        right_corner = left_corner + new_width
        part = image.crop((left_corner, 0, right_corner, height))
        result = Image.new(image.mode, (width, height))
        result.paste(part, (left_corner, 0, right_corner, height))
        return result
    else:  # add edges
        new_image = Image.new(image.mode, (new_width, height))
        left_corner = (new_width - width) // 2
        right_corner = left_corner + width
        new_image.paste(image, (left_corner, 0, right_corner, height))
        return new_image


def rotate(image: Image, angle: int = 0) -> Image:
    return image.rotate(angle, expand=False, fillcolor=0)


def puzzle(image: Image, size: Tuple[int, int], rotate_parts: bool = False, random_seed: int = 0):
    seed = random_seed
    if seed == 0:
        seed = hash(imagehash.dhash(image))
    width = image.size[0]
    height = image.size[1]
    square_length = math.gcd(size[0], size[1])
    redundant_width = (width - size[0])
    start_point_width = redundant_width // 2
    start_point_height = (height - size[1]) // 2
    parts = list()
    for h in range(size[1] // square_length):
        h_current = (h * square_length) + start_point_height
        for w in range(size[0] // square_length):
            w_current = (w * square_length) + start_point_width
            square = (w_current, h_current, w_current + square_length, h_current + square_length)
            parts.append(image.crop(square))

    rnd = random.Random(seed)
    rnd.shuffle(parts)
    if rotate_parts:
        parts = [i.rotate(rnd.sample([0, 90, 180, 270], 1)[0], expand=False, fillcolor=0) for i in parts]
    new_image = Image.new(image.mode, (width, height))
    h = w = 0
    for i in parts:
        left_corner = (w * square_length) + start_point_width
        left_height = (h * square_length) + start_point_height
        square = (left_corner, left_height, left_corner + square_length, left_height + square_length)
        new_image.paste(i, square)
        w += 1
        if (w * square_length) + redundant_width >= width:
            w = 0
            h += 1
    return new_image


if __name__ == '__main__':
    actions = [lambda x: x, lambda x: to_ratio(x, (4, 3)), lambda x: rotate(x, 90), lambda x: rotate(x, 180),
               lambda x: rotate(x, 270), lambda x: puzzle(x, (450, 270)), lambda x: puzzle(x, (450, 270), True)]

    files = os.listdir(INPUT_DATA_DIR)

    for i in range(len(files)):
        action_idx = i % len(actions)
        action = actions[action_idx]
        im = Image.open(INPUT_DATA_DIR + "\\" + files[i])
        result = action(im)
        result.save(OUTPUT_DATA_DIR + "\\" + files[i].replace(".jpg", ".png"))
        im.close()
        result.close()
