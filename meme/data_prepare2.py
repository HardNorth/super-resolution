import os
import tempfile
from PIL import Image, ImageFilter
from concurrent.futures import ThreadPoolExecutor

INPUT_DATA_DIR = "d:\\ml\\goat-data\\DIV2K_LR"
OUTPUT_DATA_DIR = "d:\\ml\\goat-data\\DIV2K_LR_compressed"
BATCH_SIZE = 8


def compress(image: Image, quality: int) -> Image:
    channels = image.split()
    tmp_file = tempfile.mktemp("compression")
    if len(channels) > 3:
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        background.save(tmp_file, "jpeg", quality=quality)
        background.close()
    else:
        image.save(tmp_file, "jpeg", quality=quality)
    image.close()
    return Image.open(tmp_file)


def blur(image: Image, radius: float) -> Image:
    blurred = image.filter(ImageFilter.GaussianBlur(radius))
    return blurred


def image_action(action, file):
    im = Image.open(INPUT_DATA_DIR + "\\" + file)
    result = action(im)
    result.save(OUTPUT_DATA_DIR + "\\" + file.replace(".jpg", ".png"))
    im.close()
    result.close()


if __name__ == '__main__':
    actions = [lambda x: compress(blur(x, 0.5), 30)]

    files = os.listdir(INPUT_DATA_DIR)
    image_actions = list()

    for i in range(len(files)):
        action_idx = i % len(actions)
        action = actions[action_idx]
        image_actions.append(action)

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        executor.map(image_action, image_actions, files)
