import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from tensorflow.compat.v1.layers import flatten

import src.layers as ly


def network(
    x: tf.placeholder,
    grayscale: bool,
    normalize: bool,
    low_keep_prob: float,
    high_keep_prob: float,
):
    """
    Multilayer network to classify traffic sign images.
    @param x: input images
    @param grayscale: whether the images should be converted to grayscale
    @param normalize: whether the converted images should be normalized
    @param low_keep_prob: a lower probability of keeping values for the dropout regularization
    @param high_keep_prob: a higher probability of keeping values for the dropout regularization
    """
    depth = 3
    if grayscale:
        x = tf.image.rgb_to_grayscale(x)
        depth = 1
        if normalize:
            x = ly.normalize_grayscale(x)

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    layer_1 = ly.convolutional_network(x, 32, 1, 5, 6)

    # Activation.
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, high_keep_prob)

    # Layer 2: Convolutional. Input = 28x28x6. Output = 10x10x16.
    layer_2 = ly.convolutional_network(layer_1, 28, 6, 5, 16)

    # Activation.
    layer_2 = tf.nn.relu(layer_2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    k = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    layer_2 = tf.nn.max_pool(layer_2, k, strides, padding)

    # Layer 3: Convolutional. Input = 5x5x16, Output = 8x8x412.
    layer_3 = ly.convolutional_network(layer_2, 5, 16, 5, 512)

    # Flatten. Input = 8x8x1024. Output = 26368.
    fc = flatten(layer_3)
    fc = tf.nn.dropout(fc, high_keep_prob)

    # Layer 4: Fully Connected. Input = 65536. Output = 512.
    layer_4 = ly.linear_network(fc, 32768, 256)

    # Activation.
    layer_4 = tf.nn.relu(layer_4)
    layer_4 = tf.nn.dropout(layer_4, low_keep_prob)

    # Layer 5: Fully Connected. Input = 512. Output = 86.
    layer_5 = ly.linear_network(layer_4, 256, 128)

    # Activation.
    layer_5 = tf.nn.relu(layer_5)
    layer_5 = tf.nn.dropout(layer_5, low_keep_prob)

    # Layer 6: Fully Connected. Input = 86. Output = 43.
    logits = ly.linear_network(layer_5, 128, 43)

    return logits
