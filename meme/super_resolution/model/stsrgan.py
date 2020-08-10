import tensorflow as tf
from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Conv3D, Dense, Flatten, Input, LeakyReLU, PReLU, \
    Lambda, Dropout, TimeDistributed
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg19 import VGG19


from super_resolution.model.common import pixel_shuffle, normalize_01, normalize_m11, denormalize_m11


def upsample(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)


def res_block(x_in, num_filters, training=False, momentum=0.8):
    x = Conv3D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=momentum)(x)
    if training:
        x = Dropout(0.1)(x, training=training)
    x = PReLU(shared_axes=[1, 2, 3])(x)
    x = Conv3D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    if training:
        x = Dropout(0.1)(x, training=training)
    x = Add()([x_in, x])
    return x


def st_sr_resnet(num_filters=64, num_res_blocks=16, training=False):
    x_in = Input(shape=(None, None, None, 3))
    x = Lambda(normalize_01)(x_in)

    x = Conv3D(num_filters, kernel_size=9, padding='same')(x)
    if training:
        x = Dropout(0.1)(x, training=training)
    x = x_1 = PReLU(shared_axes=[1, 2, 3])(x)

    for _ in range(num_res_blocks):
        x = res_block(x, num_filters, training)

    x = Conv3D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    if training:
        x = Dropout(0.1)(x, training=training)
    x = Add()([x_1, x])

    x2d = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding='same')
    x = TimeDistributed(x2d)(x)

    x = upsample(x, num_filters * 4)
    if training:
        x = Dropout(0.1)(x, training=training)
    x = upsample(x, num_filters * 4)
    if training:
        x = Dropout(0.1)(x, training=training)

    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    x = Lambda(denormalize_m11)(x)

    return Model(x_in, x)


generator = st_sr_resnet