import os
import tensorflow as tf

from super_resolution.data import DIV2K
from super_resolution.model.srgan import generator
from super_resolution.train import SrganGeneratorTrainer

WEIGHTS_FILE = "d:\\ml\\goat-data\\gan_generator.h5"

gan_generator = generator()
gan_generator.load_weights(WEIGHTS_FILE)

div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')
div2k_valid = DIV2K(scale=4, subset='valid', downgrade='bicubic')

train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1)

pre_trainer = SrganGeneratorTrainer(model=gan_generator, checkpoint_dir=f'.ckpt/pre_generator')
pre_trainer.train(train_ds,
                  valid_ds.take(10),
                  steps=1000000,
                  evaluate_every=1000,
                  save_best_only=False)

pre_trainer.model.save_weights("d:\\ml\\goat-data\\gan_generator_1.h5")
