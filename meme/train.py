import os

import tensorflow as tf
import random

from meme.utils import read_images
from train import SrganGeneratorTrainer, SrganTrainer
from data import random_crop, random_rotate, random_flip
from meme.super_resolution.model.srgan import generator, discriminator

from tensorflow.python.data.experimental import AUTOTUNE

# To restart
# Change random seed
# Delete cockpit
# Delete cache

VALIDATION_DATASET_SIZE = 16
TRAIN_DATASET_SIZE = 8000

CACHE_DIR = "c:\\ml\\cache"
LOW_RES_IMAGES = "d:\\ml\\goat-data\\goat-train-input"
HI_RES_IMAGES = "d:\\ml\\goat-data\\goat-train-output"

VALIDATION_LR = "d:\\ml\\goat-data\\validation_lr"
VALIDATION_HR = "d:\\ml\\goat-data\\validation_hr"

INPUT_WEIGHTS_FILE = "d:\\ml\\goat-data\\goat-train-weights9.h5"
OUTPUT_WEIGHTS_FILE = "d:\\ml\\goat-data\\goat-train-weights9_gan.h5"

INPUT_WEIGHTS_DISCR_FILE = "d:\\ml\\goat-data\\goat-train-discr-weights4.h5"
OUTPUT_WEIGHTS_DISCR_FILE = "d:\\ml\\goat-data\\goat-train-discr-weights4.h5"

# RANDOM_SEED = int(random.random() * 1000)
RANDOM_SEED = 147
HR_FRAGMENT_SIZE = 35 * 4
NUM_FILTERS = 128


def prepare_dataset(name, files, image_type):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    ds = read_images(files, image_type).cache(os.path.join(CACHE_DIR, name + ".image.cache"))
    for _ in ds:
        pass
    return ds


def dataset(low_res, hi_res, batch_size=16, repeat_count=None):
    ds = tf.data.Dataset.zip((low_res, hi_res))
    ds = ds.map(lambda lr, hr: random_crop(lr, hr, HR_FRAGMENT_SIZE, scale=4), num_parallel_calls=AUTOTUNE)
    ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
    ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.repeat(repeat_count)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


INPUT_FILES = [os.path.join(LOW_RES_IMAGES, f) for f in os.listdir(LOW_RES_IMAGES)]

random.Random(RANDOM_SEED).shuffle(INPUT_FILES)
valid_files = [os.path.join(VALIDATION_LR, f) for f in os.listdir(VALIDATION_LR)]
train_files = INPUT_FILES[:TRAIN_DATASET_SIZE]

train_ds_low = prepare_dataset("train-low", train_files, "png")
train_ds_hi = prepare_dataset("train-hi", [f.replace(LOW_RES_IMAGES, HI_RES_IMAGES) for f in train_files], "png")

valid_ds_low = prepare_dataset("valid-low", valid_files, "png")
valid_ds_hi = prepare_dataset("valid-hi", [f.replace(VALIDATION_LR, VALIDATION_HR) for f in valid_files], "png")

train_ds = dataset(train_ds_low, train_ds_hi)
valid_ds = dataset(valid_ds_low, valid_ds_hi, repeat_count=1)

# Generator trainer
# model = generator(NUM_FILTERS, training=True)
# if os.path.exists(INPUT_WEIGHTS_FILE):
#     model.load_weights(INPUT_WEIGHTS_FILE)
#
# pre_trainer = SrganGeneratorTrainer(model=model, checkpoint_dir=f'.ckpt/goat_generator')
# pre_trainer.train(train_ds, valid_ds, steps=300000, evaluate_every=1000, save_best_only=False)
#
# pre_trainer.model.save_weights(OUTPUT_WEIGHTS_FILE)
# print(RANDOM_SEED)


# Generator + discriminator trainer
model = generator(NUM_FILTERS)
if os.path.exists(INPUT_WEIGHTS_FILE):
    model.load_weights(INPUT_WEIGHTS_FILE)

discriminator = discriminator(NUM_FILTERS, HR_FRAGMENT_SIZE)
if os.path.exists(INPUT_WEIGHTS_DISCR_FILE):
    discriminator.load_weights(INPUT_WEIGHTS_DISCR_FILE)

trainer = SrganTrainer(generator=model, discriminator=discriminator)
trainer.train(train_ds, steps=200000)

trainer.generator.save_weights(OUTPUT_WEIGHTS_FILE)
trainer.discriminator.save_weights(OUTPUT_WEIGHTS_DISCR_FILE)
