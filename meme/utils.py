import os
import tensorflow as tf
from typing import Tuple, List

from PIL import Image
import imagehash
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE


def split_image_by_size(image: Image.Image, size: Tuple[int, int]) -> List[Image.Image]:
    result: List[Image.Image] = list()
    for w in range(0, image.size[0], size[0]):
        for h in range(0, image.size[1], size[1]):
            im = image.crop((w, h, w + size[0], h + size[1]))
            result.append(im)
    return result


def save_images(images: List[Image.Image], output_dir: str, image_type: str = 'png', name_pattern: str = '{:0>4d}'):
    for i in range(len(images)):
        images[i].save(output_dir + os.path.sep + name_pattern.format(i) + '.' + image_type, image_type)


def image_difference(image1: Image.Image, image2: Image.Image, hash_size: int = 8) -> int:
    return abs(imagehash.dhash(image1, hash_size) - imagehash.dhash(image2, hash_size))


def max_image_difference(images: List[Image.Image], hash_size: int = 8) -> int:
    size = len(images)
    if size < 2:
        raise ValueError("There should be more than 1 image in the input list")
    max_diff = 0
    for i in range(1, len(images)):
        diff = image_difference(images[i - 1], images[i], hash_size)
        if diff > max_diff:
            max_diff = diff
    return max_diff


def random_crop_3d(lr_3d_img, hr_img, hr_crop_size=96, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_3d_img[0])[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_3d_img_cropped = [img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size] for img in lr_3d_img]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return tf.data.Dataset.from_tensor_slices(lr_3d_img_cropped), hr_img_cropped


def random_rotate_3d(lr_3d_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.data.Dataset.from_tensor_slices([tf.image.rot90(img, rn) for img in lr_3d_img]), tf.image.rot90(hr_img,
                                                                                                              rn)


def random_flip_3d(lr_3d_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_3d_img, hr_img),
                   lambda: (
                       tf.data.Dataset.from_tensor_slices([tf.image.flip_left_right(img, rn) for img in lr_3d_img]),
                       tf.image.flip_left_right(hr_img)))


def read_images(image_files: List[str], image_type: str) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(image_files)
    ds = ds.map(tf.io.read_file)
    if image_type == 'png':
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
    elif image_type == 'jpeg':
        ds = ds.map(lambda x: tf.image.decode_jpeg(x, channels=3), num_parallel_calls=AUTOTUNE)
    else:
        raise ValueError()
    return ds
