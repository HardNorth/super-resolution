from super_resolution.model.srgan import generator
from super_resolution.model import resolve_single
from super_resolution.utils import load_image
from PIL import Image

DATA_DIR = "d:\\ml\\goat-data"
WEIGHTS_FILE = DATA_DIR + '\\' + "gan_generator.h5"


gan_generator = generator()
gan_generator.load_weights(WEIGHTS_FILE)

gan_sr = resolve_single(gan_generator, load_image('d:\\ml\\goat-data\\goat-train-input\\goat-train_010647.png'))
Image.fromarray(gan_sr.numpy(), 'RGB').save(DATA_DIR + '\\' + "test.png")
