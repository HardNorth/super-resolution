import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


INPUT_DATA_DIR = "d:\\ml\\goat-data\\goat-train-hi_bits"
OUTPUT_DATA_DIR = "d:\\ml\\goat-data\\goat-train-output"


def convert_file(source: str, dest: str):
    im = Image.open(source)
    result = im.convert("RGB", palette=Image.ADAPTIVE)
    result.save(dest)


if __name__ == '__main__':

    files = os.listdir(INPUT_DATA_DIR)

    argument_1 = list()
    argument_2 = list()
    for i in range(len(files)):
        argument_1.append(INPUT_DATA_DIR + "\\" + files[i])
        argument_2.append(OUTPUT_DATA_DIR + "\\" + files[i])

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(convert_file, argument_1, argument_2)
