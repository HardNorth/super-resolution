import os
import numpy as np
from super_resolution.model.srgan import generator
from super_resolution.model import resolve, resolve_single
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# INPUT_DIR = "c:\\ml\\goat-data\\frames-raw"
INPUT_DIR = "c:\\ml\\photo\\input"
INPUT_FILES = os.listdir(INPUT_DIR)
WEIGHTS_FILE = "d:\\ml\\goat-data\\goat-train-weights9_gan.h5"
# OUTPUT_DIR = "c:\\ml\\goat-data\\frames-super"
OUTPUT_DIR = "c:\\ml\\photo\\output"
NUM_FILTERS = 128

SKIP_FIRST = 0
BATCH_SIZE = 1


def save_image(source, file_name):
    Image.fromarray(source.numpy(), 'RGB').save(OUTPUT_DIR + "\\" + file_name)


def load_image(path):
    result = np.array(Image.open(path))[:, :, : 3]
    return result


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

gan_generator = generator(NUM_FILTERS)
gan_generator.load_weights(WEIGHTS_FILE)

images = list()
files = list()
with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
    for i in range(SKIP_FIRST, len(INPUT_FILES)):
        images.append(load_image(INPUT_DIR + "\\" + INPUT_FILES[i]))
        files.append(INPUT_FILES[i])
        if len(images) >= BATCH_SIZE:
            if BATCH_SIZE == 1:
                gan_sr = resolve_single(gan_generator, images[0])
                executor.map(save_image, [gan_sr], files)
            else:
                gan_sr = resolve(gan_generator, images)
                executor.map(save_image, gan_sr, files)
            images.clear()
            files.clear()

if len(images) > 0:
    gan_sr = resolve(gan_generator, images)
    for j in range(len(images)):
        Image.fromarray(gan_sr[j].numpy(), 'RGB').save(OUTPUT_DIR + "\\" + INPUT_FILES[len(INPUT_FILES)-len(images)+j])
