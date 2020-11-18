import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

mu = 0
sigma = 0.1


def normalize_grayscale(image_data: tf.Tensor):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    max_value = 0.9
    min_value = 0.1
    X_std = (image_data - tf.reduce_min(image_data)) / (
        tf.reduce_max(image_data) - tf.reduce_min(image_data)
    )
    X_scaled = X_std * (max_value - min_value) + min_value
    return X_scaled


def linear_network(x_in: int, in_dim: int, out_dim: int, mu_in=mu, sigma_in=sigma):
    """
    Create a linear network layer with the input parameters provided.
    """
    W = tf.Variable(
        tf.truncated_normal(shape=(in_dim, out_dim), mean=mu_in, stddev=sigma_in)
    )
    B = tf.Variable(tf.zeros(shape=(1, out_dim)))
    y_out = tf.matmul(x_in, W) + B
    return y_out


def convolutional_network(
    x_in: int, in_h_w: int, in_depth: int, filter_h_w: int, out_depth: int
):
    """
    Create a convolutional network layer with the input parameters provided.
    """
    out_h_w = (in_h_w - filter_h_w) + 1  # no padding, stride = 1
    W = tf.Variable(
        tf.truncated_normal(
            shape=(filter_h_w, filter_h_w, in_depth, out_depth), mean=mu, stddev=sigma
        )
    )
    B = tf.Variable(tf.zeros(shape=(1, out_h_w, out_h_w, out_depth)))
    strides = [1, 1, 1, 1]
    padding = "VALID"
    y_out = tf.nn.conv2d(x_in, W, strides=strides, padding=padding) + B
    return y_out
